"""Run a short L1 SCALED flow-field inference.

This is a small, local-friendly wrapper around the L1 tutorial. It keeps the
original tutorial file untouched, checks required files up front, and lets us
run only a few timesteps for quick visualization.
"""

from __future__ import annotations

import argparse
import os
import sys
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class RunConfig:
    repo_root: Path
    steps: int = 5
    device: str = "auto"
    precision: str = "fp32"
    output_dir: Path | None = None


def required_files(config: RunConfig) -> dict[str, Path]:
    return {
        "compression_weight": config.repo_root / "weight" / "compression.pth",
        "inference_weight": config.repo_root / "weight" / "inference.pth",
        "geometry": config.repo_root / "l1_regression_based_surrogate_model" / "geo.npy",
    }


def _check_required_files(config: RunConfig) -> None:
    missing = [str(path) for path in required_files(config).values() if not path.exists()]
    if missing:
        joined = "\n".join(f"- {path}" for path in missing)
        raise FileNotFoundError(f"Missing required files:\n{joined}")


def _select_device(torch_module, requested: str):
    if requested != "auto":
        return torch_module.device(requested)
    if torch_module.cuda.is_available():
        return torch_module.device("cuda")
    if hasattr(torch_module.backends, "mps") and torch_module.backends.mps.is_available():
        return torch_module.device("mps")
    return torch_module.device("cpu")


def run(config: RunConfig) -> Path:
    _check_required_files(config)

    import importlib.metadata

    original_version = importlib.metadata.version
    importlib.metadata.version = (
        lambda name: "1.24.4" if name == "numpy" else original_version(name)
    )

    import matplotlib.pyplot as plt
    import numpy as np
    import torch
    from tqdm import tqdm

    sys.path.insert(0, str(config.repo_root))
    from scaled.model.autoencoders.autoencoder3dv1 import AutoencoderKL
    from scaled.model.unets.unet_3ds import UNet3DsModel

    device = _select_device(torch, config.device)
    use_fp16 = config.precision == "fp16"
    if use_fp16 and device.type != "cuda":
        raise ValueError("FP16 inference is only supported on CUDA in this runner.")
    model_dtype = torch.float16 if use_fp16 else torch.float32
    output_dir = config.output_dir or (config.repo_root / "outputs" / "l1_flow_inference")
    output_dir.mkdir(parents=True, exist_ok=True)

    width = 128
    height = 128
    depth = 64
    files = required_files(config)

    compression_model = AutoencoderKL(
        in_channels=3,
        out_channels=3,
        down_block_types=["DownEncoderBlock3D", "DownEncoderBlock3D", "DownEncoderBlock3D"],
        up_block_types=["UpDecoderBlock3D", "UpDecoderBlock3D", "UpDecoderBlock3D"],
        block_out_channels=[128, 256, 384],
        latent_channels=4,
    )
    compression_model.load_state_dict(torch.load(files["compression_weight"], map_location="cpu"))
    compression_model.to(device=device, dtype=model_dtype).eval()

    inference_model = UNet3DsModel(
        in_channels=8,
        out_channels=4,
        down_block_types=("DownBlock3D", "DownBlock3D", "DownBlock3D", "DownBlock3D"),
        up_block_types=("UpBlock3D", "UpBlock3D", "UpBlock3D", "UpBlock3D"),
        block_out_channels=(128, 256, 384, 512),
        add_attention=False,
    )
    inference_model.load_state_dict(torch.load(files["inference_weight"], map_location="cpu"))
    inference_model.to(device=device, dtype=model_dtype).eval()

    x0 = torch.zeros(
        (1, 3, depth, height, width),
        device=device,
        dtype=model_dtype,
    ) / 3
    geometry_array = np.load(files["geometry"])
    geometry_ = torch.zeros(depth, height, width)
    geometry_[:, 8:-8, 8:-8] = torch.tensor(geometry_array)[:, 8:-8, 8:-8]
    geometry = geometry_.bool()

    xbc = torch.ones((1, 3, depth, height, width), dtype=model_dtype)
    xbc[:, :, geometry] = 0
    xbc = xbc.to(device)

    with torch.no_grad():
        latent_x0 = compression_model.encode(x0) / 10
        latent_xbc = compression_model.encode(xbc) / 10
        if use_fp16:
            compression_model.to(dtype=torch.float32)

        for step in tqdm(range(config.steps), desc=f"L1 flow inference on {device}"):
            model_input = torch.cat([latent_x0, latent_xbc], dim=1)
            output = inference_model(model_input).sample
            latent_x0 = output.clone()
            decode_input = latent_x0.float() if use_fp16 else latent_x0
            decoded = compression_model.decode(decode_input * 10)
            slice_data = decoded.detach().cpu().numpy()[0, 0, 4] * -3

            plt.figure(figsize=(5, 5))
            plt.imshow(slice_data, vmax=1, vmin=-0.5)
            plt.title(f"L1 flow channel u, z=4, timestep {step}")
            plt.colorbar()
            plt.tight_layout()
            plt.savefig(output_dir / f"flow_{step:03d}.png", dpi=200)
            plt.close()

    return output_dir


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", default=Path(__file__).resolve().parents[1])
    parser.add_argument("--steps", type=int, default=5)
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "mps", "cuda"])
    parser.add_argument("--precision", default="fp32", choices=["fp32", "fp16"])
    parser.add_argument("--output-dir", default=None)
    args = parser.parse_args()

    config = RunConfig(
        repo_root=Path(args.repo_root).resolve(),
        steps=args.steps,
        device=args.device,
        precision=args.precision,
        output_dir=Path(args.output_dir).resolve() if args.output_dir else None,
    )
    output_dir = run(config)
    print(output_dir)


if __name__ == "__main__":
    os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
    main()
