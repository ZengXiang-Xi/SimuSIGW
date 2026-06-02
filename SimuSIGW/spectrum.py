"""TT projection and gravitational-wave spectrum utilities."""

from __future__ import annotations

from pathlib import Path

import numpy as np
from scipy import fft


Array = np.ndarray


def tt_energy_spectrum(
    a11: Array,
    a12: Array,
    a13: Array,
    a22: Array,
    a23: Array,
    a33: Array,
    workers: int = -1,
) -> tuple[Array, Array]:
    """Return shell counts and the TT-projected GW energy spectrum.

    This is the vectorized version of ``TT_project2`` from the notebook.  The
    projection is unchanged, but the final ``j,l,m`` loop is replaced by
    ``np.bincount``.
    """

    n = int(a11.shape[0])
    k1 = np.fft.fftfreq(n) * n
    kx, ky, kz = np.meshgrid(k1, k1, k1, indexing="ij")
    k = np.sqrt(kx**2 + ky**2 + kz**2)
    k_safe = np.where(k == 0.0, 1.0e-10, k)

    p11 = 1.0 - (kx / k_safe) ** 2
    p12 = -(kx / k_safe) * (ky / k_safe)
    p13 = -(kx / k_safe) * (kz / k_safe)
    p22 = 1.0 - (ky / k_safe) ** 2
    p23 = -(ky / k_safe) * (kz / k_safe)
    p33 = 1.0 - (kz / k_safe) ** 2

    fa11 = fft.fftn(a11, workers=workers)
    fa12 = fft.fftn(a12, workers=workers)
    fa13 = fft.fftn(a13, workers=workers)
    fa22 = fft.fftn(a22, workers=workers)
    fa23 = fft.fftn(a23, workers=workers)
    fa33 = fft.fftn(a33, workers=workers)

    v11 = p11 * fa11 + p12 * fa12 + p13 * fa13
    v12 = p11 * fa12 + p12 * fa22 + p13 * fa23
    v13 = p11 * fa13 + p12 * fa23 + p13 * fa33
    v21 = p12 * fa11 + p22 * fa12 + p23 * fa13
    v22 = p12 * fa12 + p22 * fa22 + p23 * fa23
    v23 = p12 * fa13 + p22 * fa23 + p23 * fa33
    v31 = p13 * fa11 + p23 * fa12 + p33 * fa13
    v32 = p13 * fa12 + p23 * fa22 + p33 * fa23
    v33 = p13 * fa13 + p23 * fa23 + p33 * fa33

    tr_pupu = (
        np.abs(v11) ** 2
        + np.abs(v22) ** 2
        + np.abs(v33) ** 2
        + v12 * np.conj(v21)
        + v21 * np.conj(v12)
        + v13 * np.conj(v31)
        + v31 * np.conj(v13)
        + v23 * np.conj(v32)
        + v32 * np.conj(v23)
    )
    tr_pu = v11 + v22 + v33
    projected_power = (tr_pupu - 0.5 * tr_pu * np.conj(tr_pu)).real

    kmax = int((np.sqrt(3.0) / 2.0) * n) + 2
    bins = np.floor(k + 0.5).astype(np.intp).ravel()
    weights = ((np.pi / 12.0) * (k**3 / n**6) * projected_power).ravel()

    knumber = np.bincount(bins, minlength=kmax).astype(np.float64)[:kmax]
    spectrum_sum = np.bincount(bins, weights=weights, minlength=kmax)[:kmax]
    spectrum = np.divide(
        spectrum_sum,
        knumber,
        out=np.zeros_like(spectrum_sum, dtype=np.float64),
        where=knumber > 0,
    )
    return knumber, spectrum


def load_gw_spectra(
    time_steps: list[int] | tuple[int, ...],
    raw_path: str | Path,
    output_dir: str | Path = "GW_data",
    name: str | None = None,
    dt: float | None = None,
    workers: int = -1,
) -> dict[int, Array]:
    """Load saved ``aij`` snapshots, compute spectra, and save them."""

    raw_path = Path(raw_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    prefix = name or raw_path.name

    results: dict[int, Array] = {}
    for step in time_steps:
        components = [
            np.load(raw_path / f"{component}_{step}.npy")
            for component in ("a11", "a12", "a13", "a22", "a23", "a33")
        ]
        _, spectrum = tt_energy_spectrum(*components, workers=workers)
        if dt is not None:
            spectrum = spectrum * ((step + 5) * dt) ** 2
        np.save(output_dir / f"{prefix}_{step}.npy", spectrum)
        results[step] = spectrum

    return results
