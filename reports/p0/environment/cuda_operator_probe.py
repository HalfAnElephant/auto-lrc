"""Small CUDA compatibility probe; does not load project models or training data."""
import json
import platform

import torch
import torchaudio


result = {
    "python": platform.python_version(),
    "torch": torch.__version__,
    "torchaudio": torchaudio.__version__,
    "cuda": torch.version.cuda,
    "cudnn": torch.backends.cudnn.version(),
    "cuda_available": torch.cuda.is_available(),
}
assert result["cuda_available"], "CUDA unavailable"
result.update({
    "gpu": torch.cuda.get_device_name(0),
    "compute_capability": list(torch.cuda.get_device_capability(0)),
    "vram_bytes": torch.cuda.get_device_properties(0).total_memory,
    "compiled_architectures": torch.cuda.get_arch_list(),
})
torch.manual_seed(7)
x = torch.randn(2, 16, 64, device="cuda")
with torch.inference_mode():
    conv = torch.nn.Conv1d(16, 32, 3).cuda()(x)
    seq, _ = torch.nn.LSTM(16, 32, batch_first=True).cuda()(x.transpose(1, 2))
    spectrum = torch.stft(
        x[0, 0], n_fft=32, hop_length=8,
        window=torch.hann_window(32, device="cuda"), return_complex=True,
    )
    for name, value in [("conv", conv), ("lstm", seq), ("stft", spectrum)]:
        assert value.is_cuda and torch.isfinite(value).all(), name
        result[name] = {"device": str(value.device), "shape": list(value.shape), "finite": True}
torch.cuda.synchronize()
result["status"] = "passed"
print(json.dumps(result, indent=2))
