import copy
import json

import numpy as np
import pytest
import torch

from t2l.boundary.inference import (CHECKPOINT_SCHEMA, load_boundary_checkpoint,
                                    refine_boundaries, resolve_device)
from t2l.boundary.model import BoundaryTCN
from t2l.boundary.neural_features import (BoundaryConfig, PHONE_TO_ID, build_candidate_windows,
                                         extract_neural_features, phone_id)


@pytest.fixture(autouse=True, scope="module")
def limited_cpu_threads():
    previous = torch.get_num_threads()
    torch.set_num_threads(2)
    yield
    torch.set_num_threads(previous)


def candidate(start=12000, *, length=48000, rate=48000, **kwargs):
    result = {"candidate_id": "word", "text": "word", "status": "candidate", "phonemes": ["W", "ER1", "D"],
        "original_sample_rate": rate, "original_num_samples": length,
        "start_sample": start, "start_seconds": start / rate if start is not None else None,
        "end_sample": length, "end_seconds": length / rate,
        "raw_frames": {"start": 4, "end": 9, "convention": "legacy"},
        "confidence": None, "confidence_status": "uncalibrated", "sample_coordinate_kind": "coarse"}
    result.update(kwargs)
    if "end_sample" in kwargs and "end_seconds" not in kwargs:
        result["end_seconds"] = kwargs["end_sample"] / rate if kwargs["end_sample"] is not None else None
    return result


@pytest.fixture
def feature_data():
    return extract_neural_features(np.zeros(48000), 48000)


def test_default_configuration_roundtrip_and_strict_schema():
    config = BoundaryConfig()
    assert BoundaryConfig.from_dict(json.loads(json.dumps(config.to_dict()))) == config
    with pytest.raises(ValueError, match="exactly"):
        BoundaryConfig.from_dict({"channels": 128})
    with pytest.raises(ValueError, match="exactly"):
        BoundaryConfig.from_dict(dict(config.to_dict(), extra="unsupported"))


@pytest.mark.parametrize("kwargs", [{"mel_bins": 47}, {"mel_bins": 48.0}, {"radius_seconds": .4},
    {"hop_seconds": .005}, {"window_frames": 280}, {"channels": 0}, {"norm_groups": 3},
    {"dropout": float("nan")}, {"dropout": True}, {"dilations": (1, 2)}, {"schema_version": "legacy"}])
def test_incompatible_configuration_rejected(kwargs):
    with pytest.raises(ValueError):
        BoundaryConfig(**kwargs)


@pytest.mark.parametrize("rate,expected_hop", [(48000, 120), (44100, 110), (24000, 60)])
def test_neural_features_preserve_original_grid_and_57_channel_contract(rate, expected_hop):
    audio = np.zeros(rate // 10 + 1)
    result = extract_neural_features(audio, rate)
    assert np.array_equal(result["sample_positions"], np.arange(0, len(audio), expected_hop))
    assert result["features"].shape == (57, len(result["sample_positions"]))
    assert result["features"].dtype == np.float32
    assert np.isfinite(result["features"]).all()
    assert np.all(result["features"][:49] == -12)  # 48 Mel channels and log_energy.
    assert np.all(result["features"][49:] == 0)
    assert result["metadata"]["resampled"] is False
    assert result["metadata"]["mel_filter_edges_hz"][0] == pytest.approx(40)
    assert result["metadata"]["mel_filter_edges_hz"][-1] == pytest.approx(12000)
    assert result["metadata"]["latency_correction_samples"] == 0
    json.dumps(result["metadata"], allow_nan=False)


def test_low_rate_rejected_instead_of_changing_fixed_hz_filterbank():
    with pytest.raises(ValueError, match="12kHz"):
        extract_neural_features(np.zeros(22050), 22050)


def test_neural_feature_first_channel_and_amplitude_normalization():
    signal = np.sin(2 * np.pi * 200 * np.arange(5000) / 48000)
    mono = extract_neural_features(signal, 48000)
    stereo = extract_neural_features(np.vstack([signal, np.ones_like(signal)]), 48000)
    scaled = extract_neural_features(signal * 1e-100, 48000)
    assert np.array_equal(mono["features"], stereo["features"])
    assert np.allclose(mono["features"], scaled["features"], atol=1e-6)
    assert np.max(mono["features"][:48]) > -5


def test_coarse_only_fixed_window_ignores_source_and_human_boundary_fields(feature_data):
    row = candidate(12001)
    window = build_candidate_windows(feature_data, [row])
    changed = dict(row, source_truth_sample=40000, boundary={"candidate_sample": 40000, "human_truth_sample": 1})
    second = build_candidate_windows(feature_data, [changed])
    assert window["features"].shape == (1, 58, 281)
    assert window["phones"].shape == (1, 3)
    assert window["mask"].dtype == bool
    assert np.array_equal(window["features"], second["features"])
    assert np.array_equal(window["sample_positions"], second["sample_positions"])
    assert window["sample_positions"][0, 140] == 12000
    assert window["features"][0, 57, 140] == pytest.approx(-1 / 48000 / .35)
    assert np.all(np.abs(window["features"][:, 57]) <= 1)
    assert "no source or human" in window["metadata"]["center_source"]


def test_window_padding_mask_audio_edges_and_coarse_end_are_safe(feature_data):
    rows = [candidate(0, end_sample=600), candidate(47999)]
    result = build_candidate_windows(feature_data, rows)
    for index in range(2):
        valid = result["mask"][index]
        positions = result["sample_positions"][index]
        assert np.all(result["features"][index, :57, ~valid] == 0)
        assert np.all((positions[valid] >= 0) & (positions[valid] < 48000))
        assert np.all(np.abs(positions[valid] - rows[index]["start_sample"]) <= 16800)
    assert result["sample_positions"][0][result["mask"][0]].max() <= 600
    assert not result["mask"][0, :140].any()
    assert not result["mask"][1, 141:].any()


def test_phone_context_is_arpabet_stress_stripped_and_uses_real_neighbor_units(feature_data):
    rows = [candidate(6000, phonemes=["S", "AH0"]), candidate(12000, phonemes=["ER1", "D"]),
            candidate(18000, phonemes=["SH", "IY2"])]
    result = build_candidate_windows(feature_data, rows)
    assert result["phones"].tolist() == [[PHONE_TO_ID["S"], 0, PHONE_TO_ID["ER"]],
        [PHONE_TO_ID["ER"], PHONE_TO_ID["AH"], PHONE_TO_ID["SH"]],
        [PHONE_TO_ID["SH"], PHONE_TO_ID["D"], 0]]
    assert phone_id("native-symbol") == phone_id(None) == 0
    assert phone_id({"label": "AH2"}) == PHONE_TO_ID["AH"]


def test_missing_coarse_is_not_filled_from_label(feature_data):
    row = candidate(None, status="unaligned", source_truth_sample=12000)
    result = build_candidate_windows(feature_data, [row])
    assert result["features"].shape == (0, 58, 281)
    assert result["phones"].shape == (0, 3)
    assert result["skipped"] == [{"candidate_index": 0, "reason": "coarse_alignment_missing"}]


@pytest.mark.parametrize("changes", [{"original_sample_rate": 44100}, {"original_num_samples": 47999},
    {"start_sample": -1}, {"start_sample": True}, {"start_seconds": float("nan")},
    {"start_seconds": .9}, {"end_sample": 100}, {"end_sample": 48001}, {"end_seconds": .5}])
def test_bad_candidate_source_grid_or_time_rejected(feature_data, changes):
    with pytest.raises(ValueError):
        build_candidate_windows(feature_data, [candidate(**changes)])


def test_bad_feature_coordinates_values_and_config_rejected(feature_data):
    corrupted = dict(feature_data, sample_positions=feature_data["sample_positions"] + 1)
    with pytest.raises(ValueError, match="source coordinates"):
        build_candidate_windows(corrupted, [candidate()])
    corrupted = copy.deepcopy(feature_data)
    corrupted["features"][0, 0] = np.nan
    with pytest.raises(ValueError, match="source coordinates"):
        build_candidate_windows(corrupted, [candidate()])
    with pytest.raises(ValueError, match="config differ"):
        build_candidate_windows(feature_data, [candidate()], config=BoundaryConfig(channels=8, norm_groups=2))


def test_default_model_parameter_budget():
    model = BoundaryTCN()
    count = sum(parameter.numel() for parameter in model.parameters())
    assert 500_000 <= count <= 2_000_000


def test_model_heads_mask_and_all_losses_are_differentiable(feature_data):
    config = BoundaryConfig(channels=8, norm_groups=2, dropout=0.)
    data = extract_neural_features(np.zeros(48000), 48000, config=config)
    windows = build_candidate_windows(data, [candidate(0), candidate(12000)], config=config)
    model = BoundaryTCN(config)
    features = torch.from_numpy(windows["features"]).requires_grad_()
    phones = torch.from_numpy(windows["phones"])
    mask = torch.from_numpy(windows["mask"])
    result = model(features, phones, mask)
    assert {key: tuple(value.shape) for key, value in result.items()} == {
        "heatmap_logits": (2, 281), "delta_seconds": (2,), "confidence_logit": (2,)}
    mass = result["heatmap_logits"].softmax(1)
    assert torch.all(mass[~mask] == 0)
    assert torch.allclose(result["delta_seconds"], (mass * features[:, 57]).sum(1) * .35)
    loss = result["heatmap_logits"][mask].mean() + result["delta_seconds"].sum() + result["confidence_logit"].sum()
    loss.backward()
    assert features.grad is not None and torch.isfinite(features.grad).all()
    assert all(parameter.grad is not None and torch.isfinite(parameter.grad).all() for parameter in model.parameters())


def test_masked_acoustic_padding_cannot_change_model_outputs():
    config = BoundaryConfig(channels=8, norm_groups=2, dropout=0.)
    model = BoundaryTCN(config).eval()
    features = torch.zeros(1, 58, 281)
    features[:, 57] = torch.linspace(-1, 1, 281)
    mask = torch.arange(281)[None] >= 140
    phones = torch.zeros(1, 3, dtype=torch.long)
    with torch.no_grad():
        before = model(features, phones, mask)
        changed = features.clone()
        changed[:, :57, :140] = 10000
        after = model(changed, phones, mask)
    assert all(torch.equal(before[key], after[key]) for key in before)


def test_real_cpu_autocast_heads_and_backward_are_finite():
    model = BoundaryTCN(BoundaryConfig(channels=8, norm_groups=2, dropout=0.))
    features = torch.randn(2, 58, 281, requires_grad=True)
    with torch.no_grad():
        features[:, 57] = torch.linspace(-1, 1, 281)
    phones = torch.zeros(2, 3, dtype=torch.long)
    mask = torch.ones(2, 281, dtype=torch.bool)
    mask[0, :140] = False
    with torch.autocast("cpu", dtype=torch.bfloat16):
        result = model(features, phones, mask)
        loss = (torch.nn.functional.binary_cross_entropy_with_logits(
                    result["heatmap_logits"][mask], torch.zeros_like(result["heatmap_logits"][mask]))
                + result["delta_seconds"].square().mean()
                + torch.nn.functional.binary_cross_entropy_with_logits(
                    result["confidence_logit"], torch.ones_like(result["confidence_logit"])))
    assert all(value.dtype == torch.float32 and torch.isfinite(value).all() for value in result.values())
    assert torch.all(result["heatmap_logits"].softmax(1)[~mask] == 0)
    assert torch.isfinite(loss)
    loss.backward()
    assert torch.isfinite(features.grad).all()
    assert all(parameter.grad is not None and torch.isfinite(parameter.grad).all() for parameter in model.parameters())


@pytest.mark.parametrize("kind", ["nan", "phone_shape", "phone_id", "mask_empty", "bad_relative"])
def test_model_invalid_input_rejected(kind):
    model = BoundaryTCN(BoundaryConfig(channels=8, norm_groups=2))
    features, phones, mask = torch.zeros(1, 58, 281), torch.zeros(1, 3, dtype=torch.long), torch.ones(1, 281, dtype=torch.bool)
    if kind == "nan":
        features[0, 0, 0] = float("nan")
    elif kind == "phone_shape":
        phones = phones[:, :2]
    elif kind == "phone_id":
        phones[0, 0] = 100
    elif kind == "mask_empty":
        mask[:] = False
    else:
        features[0, 57, 0] = 2
    with pytest.raises(ValueError):
        model(features, phones, mask)


class FixedModel(BoundaryTCN):
    def __init__(self, positions, *, nonfinite=False):
        super().__init__(BoundaryConfig(channels=8, norm_groups=2, dropout=0.))
        self.positions = positions
        self.nonfinite = nonfinite

    def forward(self, features, phones, mask):
        logits = torch.zeros(features.shape[0], 281, device=features.device)
        for index in range(len(features)):
            logits[index, self.positions[index % len(self.positions)]] = float("nan") if self.nonfinite else 12
        mass = logits.masked_fill(~mask, torch.finfo(features.dtype).min).softmax(1)
        return {"heatmap_logits": logits, "delta_seconds": (mass * features[:, 57]).sum(1) * .35,
                "confidence_logit": torch.full((len(features),), 100., device=features.device)}


def runtime(model):
    return {"model": model, "config": model.config, "device": "cpu", "checkpoint_sha256": "a" * 64}


def test_inference_argmax_metadata_uncalibrated_and_non_destructive():
    original = candidate(12000)
    before = copy.deepcopy(original)
    result = refine_boundaries(np.zeros(48000), 48000, [original], loaded=runtime(FixedModel([160])))
    row = result[0]
    assert original == before
    assert row["start_sample"] == 14400
    assert row["start_seconds"] == .3
    assert row["raw_frames"] == original["raw_frames"]
    assert row["refinement"]["coarse_start_sample"] == 12000
    assert row["refinement"]["delta_seconds"] == .05
    assert row["refinement"]["top_candidates"][0]["sample"] == 14400
    assert row["refinement"]["confidence_logit"] == 100
    assert row["confidence"] is None
    assert row["confidence_status"] == "uncalibrated_neural"
    assert row["refinement"]["heatmap_standard_deviation_seconds"] >= 0
    json.dumps(result, allow_nan=False)


def test_inference_original_edges_and_coarse_end_mask_predictions():
    rows = [candidate(0, end_sample=120), candidate(47999)]
    results = refine_boundaries(np.zeros(48000), 48000, rows, loaded=runtime(FixedModel([0, 280])))
    assert 0 <= results[0]["start_sample"] <= 120
    assert 0 <= results[1]["start_sample"] < 48000
    assert results[1]["start_sample"] <= results[1]["end_sample"]


def test_inference_missing_and_crossing_fall_back_to_coarse_without_source_truth():
    rows = [candidate(None, status="unaligned", source_truth_sample=5000), candidate(10000), candidate(11000)]
    results = refine_boundaries(np.zeros(48000), 48000, rows, loaded=runtime(FixedModel([200, 80])))
    assert results[0]["start_sample"] is None
    assert results[0]["refinement"]["reason"] == "coarse_alignment_missing"
    assert [row["start_sample"] for row in results[1:]] == [10000, 11000]
    assert all(row["refinement"]["reason"] == "crossing_proposals" for row in results[1:])
    assert results[1]["refinement"]["predicted_sample"] != results[1]["start_sample"]


def test_inference_rejects_nonmonotonic_coarse_and_nonfinite_outputs():
    with pytest.raises(ValueError, match="monotonic"):
        refine_boundaries(np.zeros(48000), 48000, [candidate(12000), candidate(11000)], loaded=runtime(FixedModel([140])))
    with pytest.raises(ValueError, match="nonfinite"):
        refine_boundaries(np.zeros(48000), 48000, [candidate()], loaded=runtime(FixedModel([140], nonfinite=True)))


def test_checkpoint_roundtrip_explicit_cpu_and_exact_config(tmp_path):
    model = BoundaryTCN(BoundaryConfig(channels=8, norm_groups=2))
    path = tmp_path / "new-boundary.pt"
    torch.save({"schema_version": CHECKPOINT_SCHEMA, "config": model.config.to_dict(),
                "model_state_dict": model.state_dict()}, path)
    loaded = load_boundary_checkpoint(path, device="cpu")
    assert loaded["device"] == "cpu"
    assert len(loaded["checkpoint_sha256"]) == 64
    assert loaded["model"].training is False
    assert all(torch.equal(value, loaded["model"].state_dict()[key]) for key, value in model.state_dict().items())


@pytest.mark.parametrize("kind", ["old_schema", "missing_config", "unknown_config", "nan_state", "wrong_state", "precision_overflow"])
def test_checkpoint_rejects_old_incompatible_or_nonfinite_state(tmp_path, kind):
    model = BoundaryTCN(BoundaryConfig(channels=8, norm_groups=2))
    checkpoint = {"schema_version": CHECKPOINT_SCHEMA, "config": model.config.to_dict(), "model_state_dict": model.state_dict()}
    if kind == "old_schema":
        checkpoint["schema_version"] = 1
    elif kind == "missing_config":
        checkpoint.pop("config")
    elif kind == "unknown_config":
        checkpoint["config"]["extra"] = 1
    elif kind == "nan_state":
        checkpoint["model_state_dict"]["phone_embedding.weight"][0, 0] = float("nan")
    elif kind == "precision_overflow":
        checkpoint["model_state_dict"]["phone_embedding.weight"] = torch.full_like(
            checkpoint["model_state_dict"]["phone_embedding.weight"], 1e100, dtype=torch.float64)
    else:
        checkpoint["model_state_dict"]["unrecognized"] = torch.zeros(1)
    path = tmp_path / "bad.pt"
    torch.save(checkpoint, path)
    with pytest.raises((ValueError, RuntimeError)):
        load_boundary_checkpoint(path, device="cpu")


def test_no_default_checkpoint_or_device_fallback(monkeypatch):
    with pytest.raises(ValueError, match="explicitly"):
        load_boundary_checkpoint(None)
    with pytest.raises(ValueError, match="exactly one"):
        refine_boundaries(np.zeros(48000), 48000, [candidate()])
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    assert str(resolve_device("auto")) == "cpu"
    with pytest.raises(ValueError, match="no CPU fallback"):
        resolve_device("cuda")
    with pytest.raises(ValueError):
        resolve_device("xpu")


def test_device_indices_match_actual_parameter_device_without_allocating_gpu(monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "device_count", lambda: 2)
    monkeypatch.setattr(torch.cuda, "current_device", lambda: 1)
    assert str(resolve_device("auto")) == "cuda:1"
    assert str(resolve_device("cuda")) == "cuda:1"
    assert str(resolve_device("cuda:0")) == "cuda:0"
    assert str(resolve_device("cpu:0")) == "cpu"
    with pytest.raises(ValueError, match="no CPU fallback"):
        resolve_device("cuda:2")
    monkeypatch.setattr(torch.backends.mps, "is_available", lambda: True)
    assert str(resolve_device("mps")) == "mps:0"
    with pytest.raises(ValueError, match="no CPU fallback"):
        resolve_device("mps:1")
