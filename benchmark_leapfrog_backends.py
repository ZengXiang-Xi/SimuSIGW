"""Compare FFT and finite-difference evolution backends."""

from __future__ import annotations

import tempfile
import time
from pathlib import Path

import numpy as np

from SimuSIGW import (
    EvolutionConfig,
    gaussian_random_fields,
    run_evolution,
    run_evolution_fd,
    warmup_fd_backend,
)


def bench_fft(phi, workers: int, n: int, steps: int, dtype) -> float:
    with tempfile.TemporaryDirectory() as tmp:
        config = EvolutionConfig(
            n=n,
            max_steps=steps,
            output_every=0,
            output_path=Path(tmp),
            workers=workers,
            fft_workers=1,
            dtype=dtype,
            save_initial=False,
        )
        start = time.perf_counter()
        run_evolution(phi, config)
        return (time.perf_counter() - start) / steps


def bench_fd(phi, workers: int, n: int, steps: int, order: int, dtype) -> float:
    warmup_fd_backend(n=8, workers=workers, order=order, dtype=dtype)
    with tempfile.TemporaryDirectory() as tmp:
        config = EvolutionConfig(
            n=n,
            max_steps=steps,
            output_every=0,
            output_path=Path(tmp),
            workers=workers,
            fd_order=order,
            dtype=dtype,
            save_initial=False,
        )
        start = time.perf_counter()
        run_evolution_fd(phi, config, warmup=False)
        return (time.perf_counter() - start) / steps


def main() -> None:
    n = 64
    steps = 5
    workers = -1
    print(f"n={n}, steps={steps}, workers={workers}")
    for dtype in (np.float64, np.float32):
        _, phi = gaussian_random_fields(n, seed=1234, workers=1, dtype=dtype)
        print(f"dtype={np.dtype(dtype).name}")
        print(f"  fft: {bench_fft(phi, workers, n, steps, dtype):.4f} s/step")
        print(f"  fd2: {bench_fd(phi, workers, n, steps, 2, dtype):.4f} s/step")
        print(f"  fd4: {bench_fd(phi, workers, n, steps, 4, dtype):.4f} s/step")


if __name__ == "__main__":
    main()
