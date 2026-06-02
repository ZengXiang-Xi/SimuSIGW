"""Benchmark internal CPU parallelism for the optimized leapfrog evolution."""

from __future__ import annotations

import tempfile
import time
from pathlib import Path

from SimuSIGW import EvolutionConfig, gaussian_random_fields, run_evolution


def bench(workers: int, n: int = 64, steps: int = 5) -> float:
    with tempfile.TemporaryDirectory() as tmp:
        config = EvolutionConfig(
            n=n,
            max_steps=steps,
            output_every=0,
            output_path=Path(tmp),
            workers=workers,
            fft_workers=1,
            save_initial=False,
        )
        _, phi = gaussian_random_fields(n, seed=1234, workers=1)
        start = time.perf_counter()
        run_evolution(phi, config)
        return (time.perf_counter() - start) / steps


def main() -> None:
    for workers in (1, 2, 4, 8, -1):
        seconds_per_step = bench(workers)
        print(f"workers={workers:>2}: {seconds_per_step:.4f} s/step")


if __name__ == "__main__":
    main()
