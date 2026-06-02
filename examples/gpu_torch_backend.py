"""Example: run the optional PyTorch GPU backend."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from SimuSIGW import (
    EvolutionConfig,
    gaussian_bump_power_spectrum,
    gaussian_random_fields,
    run_evolution_torch,
    run_evolution_torch_fft,
)


def main() -> None:
    config = EvolutionConfig(
        n=64,
        max_steps=10,
        output_every=10,
        output_path="log/example_gpu",
        workers=-1,
        save_initial=False,
    )
    _, phi_initial = gaussian_random_fields(
        config.n,
        power_spectrum=gaussian_bump_power_spectrum,
        seed=1234,
        workers=config.workers,
        dtype=np.float32,
    )
    final_state = run_evolution_torch_fft(
        phi_initial, config, device="cuda", dtype=np.float32
    )
    print(f"finished at t={final_state.t:.8g}")


if __name__ == "__main__":
    main()
