"""Example: run SIGW leapfrog with user-defined model functions."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from SimuSIGW import EvolutionConfig, gaussian_random_fields, run_evolution_fd


def power_spectrum(k: np.ndarray) -> np.ndarray:
    """Dimensional Gaussian-bump spectrum.

    Replace this function with any callable that accepts the full k-grid and
    returns an array of the same shape.
    """

    kstar = 20.0
    width = 0.1
    amplitude = 1.0e-2
    return (
        amplitude
        * (1.0 / kstar) ** 3
        / (np.sqrt(2.0 * np.pi) * width)
        * np.exp(-((k / kstar - 1.0) ** 2) / (2.0 * width**2))
        * (2.0 * np.pi**2)
    )


def non_gaussian_transform(zeta_g: np.ndarray) -> np.ndarray:
    """Example local-type non-Gaussian transform."""

    return zeta_g + 0.5 * (zeta_g**2 - np.mean(zeta_g**2))


def main() -> None:
    config = EvolutionConfig(
        n=64,
        max_steps=10,
        output_every=10,
        output_path="log/example_custom_model",
        workers=-1,
        fd_order=4,
        save_initial=False,
    )
    _, phi_initial = gaussian_random_fields(
        config.n,
        power_spectrum=power_spectrum,
        non_gaussian_transform=non_gaussian_transform,
        seed=1234,
        workers=config.workers,
        dtype=config.dtype,
    )
    final_state = run_evolution_fd(phi_initial, config)
    print(f"finished at t={final_state.t:.8g}")


if __name__ == "__main__":
    main()
