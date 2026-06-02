"""Compare optional PyTorch dense-matrix and FFT backends."""

from __future__ import annotations

import argparse
import tempfile
import time
from pathlib import Path

import numpy as np

from SimuSIGW import (
    EvolutionConfig,
    gaussian_random_fields,
    run_evolution_torch,
    run_evolution_torch_fft,
)


def parse_dtype(name: str):
    if name == "float32":
        return np.float32
    if name == "float64":
        return np.float64
    raise argparse.ArgumentTypeError("dtype must be float32 or float64")


def bench(fn, phi, n: int, steps: int, dtype, device: str) -> float:
    import torch

    with tempfile.TemporaryDirectory() as tmp:
        config = EvolutionConfig(
            n=n,
            max_steps=steps,
            output_every=0,
            output_path=Path(tmp),
            dtype=dtype,
            save_initial=False,
        )
        if device.startswith("cuda"):
            torch.cuda.synchronize()
        start = time.perf_counter()
        fn(phi, config, device=device, dtype=dtype, clear_cache=False)
        if device.startswith("cuda"):
            torch.cuda.synchronize()
        elapsed = time.perf_counter() - start
        return elapsed / steps


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dtype", type=parse_dtype, default=np.float32)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--skip-dense", action="store_true")
    args = parser.parse_args()

    dtype = args.dtype
    device = args.device

    try:
        n = 64
        steps = 5
        _, phi = gaussian_random_fields(n, seed=1234, workers=1, dtype=dtype)
        print(f"n={n}, steps={steps}, dtype={np.dtype(dtype).name}, device={device}")
        if not args.skip_dense:
            print(
                f"  torch dense: {bench(run_evolution_torch, phi, n, steps, dtype, device):.4f} s/step"
            )
        print(
            f"  torch fft:   {bench(run_evolution_torch_fft, phi, n, steps, dtype, device):.4f} s/step"
        )

        for n, steps in ((128, 10), (256, 2)):
            _, phi = gaussian_random_fields(n, seed=1234, workers=1, dtype=dtype)
            print(f"n={n}, steps={steps}, dtype={np.dtype(dtype).name}, device={device}")
            print(
                f"  torch fft:   {bench(run_evolution_torch_fft, phi, n, steps, dtype, device):.4f} s/step"
            )
    except ImportError as exc:
        print(exc)
    except RuntimeError as exc:
        print(exc)


if __name__ == "__main__":
    main()
