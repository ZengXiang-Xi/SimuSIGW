"""Example CPU-parallel run for one SIGW leapfrog parameter point."""

from __future__ import annotations

import time

from SimuSIGW import (
    EvolutionConfig,
    gaussian_bump_power_spectrum,
    gaussian_random_fields,
    load_gw_spectra,
    run_evolution,
)


def main() -> None:
    config = EvolutionConfig(
        n=128,
        space_length=2.0,
        max_steps=1000,
        output_every=1000,
        output_path="log/test_1",
        workers=-1,
        fft_workers=1,
        fd_order=4,
        save_initial=False,
    )

    _, phi_initial = gaussian_random_fields(
        config.n,
        power_spectrum=gaussian_bump_power_spectrum,
        non_gaussianity=0.1,
        seed=None,
        workers=config.workers,
        dtype=config.dtype,
    )

    start = time.perf_counter()
    final_state = run_evolution(phi_initial, config)
    elapsed_min = (time.perf_counter() - start) / 60.0
    print(f"finished at t={final_state.t:.8g} in {elapsed_min:.3f} min")

    load_gw_spectra(
        [config.max_steps],
        raw_path=config.output_path,
        output_dir="GW_data",
        name="test_1",
        dt=config.dt,
        workers=config.workers,
    )


if __name__ == "__main__":
    main()
