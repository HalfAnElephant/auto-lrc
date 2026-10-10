import math
import warnings
import librosa
import numpy as np
from time import time
import torch
import torch.nn as nn
import torch.nn.functional as F

from . import utils
from .model import train_audio_transforms, AcousticModel, BoundaryDetection

np.random.seed(7)

SAMPLE_RATE = 22050
HOP_LENGTH = 256
FRAME_STRIDE = 3
FRAME_RESOLUTION_SECONDS = HOP_LENGTH / SAMPLE_RATE * FRAME_STRIDE


def preprocess_from_file(audio_file, lyrics_file, word_file=None):
    y, sr = preprocess_audio(audio_file)

    words, lyrics_p, idx_word_p, idx_line_p = preprocess_lyrics(
        lyrics_file, word_file)

    return y, words, lyrics_p, idx_word_p, idx_line_p


def _alignment_options(options):
    if options is None:
        return {"mode": "legacy"}
    if not isinstance(options, dict):
        raise ValueError("alignment_options must be a mapping")
    allowed = {"mode", "fallback_to_legacy", "word_windows", "onset_windows", "silence_intervals"}
    if set(options) - allowed:
        raise ValueError("Unknown alignment options: " + str(sorted(set(options) - allowed)))
    result = dict(options)
    result.setdefault("mode", "constrained")
    if result["mode"] not in ("legacy", "constrained"):
        raise ValueError("Alignment mode must be legacy or constrained")
    if result["mode"] == "legacy" and set(result) != {"mode"}:
        raise ValueError("Legacy mode does not apply alignment constraints")
    if not isinstance(result.get("fallback_to_legacy", True), bool):
        raise ValueError("fallback_to_legacy must be a boolean")
    return result


def align(audio, words, lyrics_p, idx_word_p, idx_line_p, method="Baseline", cuda=True, verbose=True,
          return_details=False, alignment_options=None):
    """Align on the legacy frame grid; opt-in details do not calibrate onset time."""

    options = _alignment_options(alignment_options)
    # start timer
    t = time()

    # constants
    alpha = 0.8

    # decode method
    if isinstance(method, str):
        method = load_mtl_model(method=method, cuda=cuda, verbose=verbose)
    ac_model, bdr_model, model_type, bdr_flag, device = method

    if not isinstance(audio, torch.Tensor):
        audio = torch.as_tensor(audio)
    if audio.ndim == 1:
        audio = audio.unsqueeze(0)
    if audio.ndim != 2 or audio.shape[0] < 1 or audio.shape[1] < 1:
        raise ValueError(
            "Audio must have shape [samples] or [channels, samples] with at "
            "least one sample."
        )
    audio = audio.to(dtype=torch.float32)

    # Keep the first channel for compatibility with the previous flattened input,
    # whose retained output corresponded most closely to the left channel.
    waveform = audio[:1]

    with torch.inference_mode():
        # reshape input, prepare mel
        x = utils.move_data_to_device(waveform, device)
        x = train_audio_transforms.to(device)(x)
        mel_frames = x.shape[-1]
        x = nn.utils.rnn.pad_sequence(x, batch_first=True).unsqueeze(1)

        # predict
        all_outputs = ac_model(x)
        if model_type == "MTL":
            all_outputs = torch.sum(all_outputs, dim=3)

        all_outputs = F.log_softmax(all_outputs, dim=2)

        _, _, num_classes = all_outputs.shape
        song_pred = all_outputs.reshape(-1, num_classes)

        # smoothing
        P_noise = torch.empty_like(song_pred).uniform_(1e-11, 1e-10)
        song_pred = torch.log(torch.exp(song_pred) + P_noise)

        verbose and print("Computing phoneme posteriorgram...")
        if bdr_flag:
            verbose and print("Computing boundary probability curve...")
            bdr_outputs = bdr_model(x).reshape(-1)
            bdr_outputs = torch.log(bdr_outputs) * alpha

    constrained_result, fallback = None, []
    if options["mode"] == "constrained":
        from t2l.alignment.constrained_dp import constrained_alignment, NoAlignmentPathError
        silence = np.zeros(song_pred.shape[0], dtype=bool)
        intervals = options.get("silence_intervals", [])
        if not isinstance(intervals, (list, tuple)):
            raise ValueError("silence_intervals must contain [start_seconds, end_seconds] pairs")
        nominal_times = np.arange(song_pred.shape[0]) * FRAME_RESOLUTION_SECONDS
        duration = waveform.shape[-1] / SAMPLE_RATE
        for interval in intervals:
            if (not isinstance(interval, (list, tuple)) or len(interval) != 2 or
                    any(isinstance(v, bool) or not isinstance(v, (float, int)) or
                        not math.isfinite(v) for v in interval) or
                    not 0 <= interval[0] < interval[1] <= duration):
                raise ValueError("Silence intervals must be finite, ordered and inside decoded audio")
            silence |= (nominal_times >= interval[0]) & (nominal_times < interval[1])
        try:
            constrained_result = constrained_alignment(
                song_pred.detach().cpu().numpy(), utils.phone2seq(lyrics_p), idx_word_p,
                word_windows=options.get("word_windows"),
                onset_windows=options.get("onset_windows"), silence_mask=silence,
                pause_phone_ids=(utils.phone2int[' '],),
                boundary_log_scores=bdr_outputs.detach().cpu().numpy() if bdr_flag else None,
                line_start_phones=np.asarray(idx_line_p)[:, 0] if bdr_flag else ())
        except NoAlignmentPathError as error:
            if not options.get("fallback_to_legacy", True):
                raise
            fallback.append("Constrained path unavailable; legacy path returned without satisfying constraints: " + str(error))
        if constrained_result is not None:
            word_align, score = constrained_result["word_frames"], constrained_result["score"]
    if constrained_result is None:
        if bdr_flag:
            line_start = idx_line_p[:, 0]
            verbose and print("Aligning...It might take a few minutes..., FIXME optimize perf.")
            word_align, score = utils.alignment_bdr(
                song_pred.detach().cpu().numpy(), lyrics_p, idx_word_p,
                bdr_outputs.detach().cpu().numpy(), line_start)
        else:
            verbose and print("Aligning...It might take a few minutes...")
            word_align, score = utils.alignment(song_pred, lyrics_p, idx_word_p)

    t = time() - t
    verbose and print("Alignment Score:\t{}\tTime:\t{}".format(score, t))

    if return_details:
        if not math.isfinite(float(score)):
            raise utils.AlignmentValueError(
                "Alignment path score is not finite; cannot export diagnostics."
            )
        details = {
            "sample_rate": SAMPLE_RATE,
            "hop_length": HOP_LENGTH,
            "frame_stride": FRAME_STRIDE,
            "frame_resolution_seconds": FRAME_RESOLUTION_SECONDS,
            "mel_frames": int(mel_frames),
            "output_frames": int(song_pred.shape[0]),
            "input_samples": int(waveform.shape[-1]),
            "method": model_type + ("_BDR" if bdr_flag else ""),
            "model_type": model_type,
            "bdr": bool(bdr_flag),
            "device": str(device),
            "score": float(score),
            "score_kind": ("whole_path_log_score_with_bdr_uncalibrated" if bdr_flag
                           else "whole_path_log_score_uncalibrated"),
            "legacy_frame_origin_seconds": 0.0,
            "physical_origin_seconds": None,
            "physical_origin_calibrated": False,
            "word_frames": [[int(start), int(end)] for start, end in word_align],
            "phonemes": list(lyrics_p),
            "word_phoneme_indices": np.asarray(idx_word_p).tolist(),
            "line_phoneme_indices": np.asarray(idx_line_p).tolist(),
        }
        if alignment_options is not None:
            details["constraints"] = dict(options,
                executed_mode="constrained" if constrained_result is not None else "legacy",
                satisfied=constrained_result is not None if options["mode"] == "constrained" else None,
                silence_frame_mapping="nominal_legacy_grid; physical origin uncalibrated")
            details["fallback"] = fallback
            details["frame_interval_convention"] = ("half_open_phone_occupancy" if constrained_result is not None
                                                     else "legacy_next_word_or_terminal_frame")
        if constrained_result is not None:
            details["score_kind"] = ("ctc_path_log_score_with_entry_bdr_uncalibrated" if bdr_flag
                                     else "ctc_path_log_score_uncalibrated")
            details["constrained_dp"] = {key: value for key, value in constrained_result.items()
                                         if key not in ("state_path", "word_frames", "score")}
        return word_align, words, details

    return word_align, words


def load_mtl_model(method="Baseline", cuda=True, verbose=True):
    cuda =  cuda and torch.cuda.is_available()
    # decode method
    if "BDR" in method:
        model_type = method[:-4]
        bdr_flag = True
    else:
        model_type = method
        bdr_flag = False
    verbose and print("Model: {} BDR?: {}".format(model_type, bdr_flag))

    # prepare acoustic model params
    if model_type == "Baseline":
        n_class = 41
    elif model_type == "MTL":
        n_class = (41, 47)
    else:
        raise ValueError("Invalid model type.")

    hparams = {
        "n_cnn_layers": 1,
        "n_rnn_layers": 3,
        "rnn_dim": 256,
        "n_class": n_class,
        "n_feats": 32,
        "stride": 1,
        "dropout": 0.1
    }

    device = 'cuda' if cuda else 'cpu'

    ac_model = AcousticModel(
        hparams['n_cnn_layers'], hparams['rnn_dim'], hparams['n_class'],
        hparams['n_feats'], hparams['stride'], hparams['dropout']
    ).to(device)

    verbose and print("Loading acoustic model from checkpoint..., cuda:", cuda) # True may cause OOM
    utils.load_model(
        ac_model, "./checkpoints/checkpoint_{}".format(model_type), cuda=cuda)
    ac_model.eval()

    if bdr_flag:
        # boundary model: fixed
        bdr_hparams = {
            "n_cnn_layers": 1,
            "rnn_dim": 32,  # a smaller rnn dim than acoustic model
            "n_class": 1,  # binary classification
            "n_feats": 32,
            "stride": 1,
            "dropout": 0.1,
        }

        bdr_model = BoundaryDetection(
            bdr_hparams['n_cnn_layers'], bdr_hparams['rnn_dim'], bdr_hparams['n_class'],
            bdr_hparams['n_feats'], bdr_hparams['stride'], bdr_hparams['dropout']
        ).to(device)
        verbose and print("Loading BDR model from checkpoint...")
        utils.load_model(
            bdr_model, "./checkpoints/checkpoint_BDR", cuda=cuda)
        bdr_model.eval()
    else:
        bdr_model = None

    return ac_model, bdr_model, model_type, bdr_flag, device


def preprocess_audio(audio_file, sr=22050):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        y, curr_sr = librosa.load(
            audio_file, sr=sr, mono=True, res_type='kaiser_fast')

    if len(y.shape) == 1:
        y = y[np.newaxis, :]  # (channel, sample)

    return y, curr_sr


def preprocess_lyrics(lyrics_file, word_file=None):
    from string import ascii_lowercase
    d = {ascii_lowercase[i]: i for i in range(26)}
    d["'"] = 26
    d[" "] = 27
    d["~"] = 28

    # process raw
    with open(lyrics_file, 'r') as f:
        raw_lines = f.read().splitlines()

    raw_lines = ["".join([c for c in line.lower() if c in d.keys()]).strip()
                 for line in raw_lines]
    raw_lines = [" ".join(line.split()) for line in raw_lines if len(line) > 0]
    # concat
    full_lyrics = " ".join(raw_lines)

    if word_file:
        with open(word_file) as f:
            words_lines = f.read().splitlines()
    else:
        words_lines = full_lyrics.split()

    lyrics_p, words_p, idx_word_p, idx_line_p = utils.gen_phone_gt(
        words_lines, raw_lines)

    return words_lines, lyrics_p, idx_word_p, idx_line_p


def write_csv(pred_file, word_align, words):
    resolution = 256 / 22050 * 3

    with open(pred_file, 'w') as f:
        for j in range(len(word_align)):
            word_time = word_align[j]
            f.write("{},{},{}\n".format(
                word_time[0] * resolution, word_time[1] * resolution, words[j]))
