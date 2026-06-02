"""Initial random fields and power spectra."""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
from scipy import fft


Array = np.ndarray
PowerSpectrum = Callable[[Array], Array]
NonGaussianTransform = Callable[[Array], Array]


def gaussian_bump_power_spectrum(
    k: Array,
    kstar: float = 10.0,
    width: float = 0.1,
    amplitude: float = 1.0e-2,
) -> Array:
    """Dimensional Gaussian-bump spectrum used in the original notebook."""

    bump = (
        amplitude
        * (1.0 / kstar) ** 3
        / (np.sqrt(2.0 * np.pi) * width)
        * np.exp(-((k / kstar - 1.0) ** 2) / (2.0 * width**2))
    )
    return bump * (2.0 * np.pi**2)


def logarithmic_non_gaussian_transform(
    zeta_gaussian: Array,
    non_gaussianity: float = 0.1,
) -> Array:
    """Default log-type non-Gaussian transform used in the notebook."""

    return -non_gaussianity * np.log(
        np.abs(1.0 - zeta_gaussian / non_gaussianity)
    )


def local_polynomial_non_gaussian_transform(
    zeta_gaussian: Array,
    f_nl: float = 0.0,
    g_nl: float = 0.0,
    subtract_variance: bool = True,
) -> Array:
    """A common local-type example transform.

    Users can pass their own callable to ``gaussian_random_fields`` instead;
    this helper is included only as a convenient template.
    """

    quadratic = zeta_gaussian**2
    if subtract_variance:
        quadratic = quadratic - np.mean(quadratic)
    return zeta_gaussian + f_nl * quadratic + g_nl * zeta_gaussian**3


def gaussian_random_fields(
    n: int,
    power_spectrum: PowerSpectrum = gaussian_bump_power_spectrum,
    non_gaussianity: float = 0.1,
    non_gaussian_transform: NonGaussianTransform | None = None,
    seed: int | None = None,
    workers: int = -1,
    dtype: np.dtype = np.float64,
) -> tuple[Array, Array]:
    """Generate Gaussian and transformed non-Gaussian scalar fields.

    The normalization follows the faster routine in ``leapfrog_version.ipynb``.
    The returned arrays are real ``(n, n, n)`` arrays.  ``power_spectrum`` must
    accept the dimensionless ``k`` grid and return the dimensional spectrum.
    ``non_gaussian_transform`` must accept ``zeta_gaussian`` and return the
    transformed ``zeta``.  If omitted, the original log transform is used.
    """

    rng = np.random.default_rng(seed)
    noise = fft.fftn(rng.normal(size=(n, n, n)).astype(dtype), workers=workers)

    k1 = np.fft.fftfreq(n) * n
    kx, ky, kz = np.meshgrid(k1, k1, k1, indexing="ij")
    k = np.sqrt(kx**2 + ky**2 + kz**2)
    k[k == 0.0] = 1.0e-12

    amplitude = np.sqrt(power_spectrum(k) * (n / (2.0 * np.pi)) ** 3)
    zeta_gaussian = fft.ifftn(noise * amplitude, workers=workers).real

    field_gaussian = (2.0 / 3.0) * zeta_gaussian
    if non_gaussian_transform is None:
        zeta_non_gaussian = logarithmic_non_gaussian_transform(
            zeta_gaussian, non_gaussianity
        )
    else:
        zeta_non_gaussian = non_gaussian_transform(zeta_gaussian)
    field_non_gaussian = (2.0 / 3.0) * zeta_non_gaussian

    return field_gaussian.astype(dtype, copy=False), field_non_gaussian.astype(
        dtype, copy=False
    )
