"""CPU-parallel finite-difference evolution backend.

This backend uses second- or fourth-order central finite differences with
periodic boundary conditions.  It trades spectral accuracy for simpler stencil
kernels that Numba can parallelize over grid points.
"""

from __future__ import annotations

import os

import numpy as np
from numba import get_num_threads, njit, prange, set_num_threads

from .fft_cpu import Array, EvolutionConfig, EvolutionState, Tensor6, save_snapshot


def _normalize_workers(workers: int) -> int:
    if workers == -1:
        return os.cpu_count() or 1
    return max(1, int(workers))


@njit(cache=True)
def _wrap_index(i: int, n: int) -> int:
    if i < 0:
        return i + n
    if i >= n:
        return i - n
    return i


@njit(cache=True)
def _d1x4(values: Array, i: int, j: int, k: int, dx: float) -> float:
    n = values.shape[0]
    im2 = _wrap_index(i - 2, n)
    im1 = _wrap_index(i - 1, n)
    ip1 = _wrap_index(i + 1, n)
    ip2 = _wrap_index(i + 2, n)
    return (
        values[im2, j, k]
        - 8.0 * values[im1, j, k]
        + 8.0 * values[ip1, j, k]
        - values[ip2, j, k]
    ) / (12.0 * dx)


@njit(cache=True)
def _d1y4(values: Array, i: int, j: int, k: int, dx: float) -> float:
    n = values.shape[0]
    jm2 = _wrap_index(j - 2, n)
    jm1 = _wrap_index(j - 1, n)
    jp1 = _wrap_index(j + 1, n)
    jp2 = _wrap_index(j + 2, n)
    return (
        values[i, jm2, k]
        - 8.0 * values[i, jm1, k]
        + 8.0 * values[i, jp1, k]
        - values[i, jp2, k]
    ) / (12.0 * dx)


@njit(cache=True)
def _d1z4(values: Array, i: int, j: int, k: int, dx: float) -> float:
    n = values.shape[0]
    km2 = _wrap_index(k - 2, n)
    km1 = _wrap_index(k - 1, n)
    kp1 = _wrap_index(k + 1, n)
    kp2 = _wrap_index(k + 2, n)
    return (
        values[i, j, km2]
        - 8.0 * values[i, j, km1]
        + 8.0 * values[i, j, kp1]
        - values[i, j, kp2]
    ) / (12.0 * dx)


@njit(cache=True)
def _d2x4(values: Array, i: int, j: int, k: int, dx: float) -> float:
    n = values.shape[0]
    im2 = _wrap_index(i - 2, n)
    im1 = _wrap_index(i - 1, n)
    ip1 = _wrap_index(i + 1, n)
    ip2 = _wrap_index(i + 2, n)
    return (
        -values[im2, j, k]
        + 16.0 * values[im1, j, k]
        - 30.0 * values[i, j, k]
        + 16.0 * values[ip1, j, k]
        - values[ip2, j, k]
    ) / (12.0 * dx * dx)


@njit(cache=True)
def _d2y4(values: Array, i: int, j: int, k: int, dx: float) -> float:
    n = values.shape[0]
    jm2 = _wrap_index(j - 2, n)
    jm1 = _wrap_index(j - 1, n)
    jp1 = _wrap_index(j + 1, n)
    jp2 = _wrap_index(j + 2, n)
    return (
        -values[i, jm2, k]
        + 16.0 * values[i, jm1, k]
        - 30.0 * values[i, j, k]
        + 16.0 * values[i, jp1, k]
        - values[i, jp2, k]
    ) / (12.0 * dx * dx)


@njit(cache=True)
def _d2z4(values: Array, i: int, j: int, k: int, dx: float) -> float:
    n = values.shape[0]
    km2 = _wrap_index(k - 2, n)
    km1 = _wrap_index(k - 1, n)
    kp1 = _wrap_index(k + 1, n)
    kp2 = _wrap_index(k + 2, n)
    return (
        -values[i, j, km2]
        + 16.0 * values[i, j, km1]
        - 30.0 * values[i, j, k]
        + 16.0 * values[i, j, kp1]
        - values[i, j, kp2]
    ) / (12.0 * dx * dx)


@njit(cache=True)
def _fd4_offset(idx: int) -> int:
    if idx == 0:
        return -2
    if idx == 1:
        return -1
    if idx == 2:
        return 1
    return 2


@njit(cache=True)
def _fd4_coeff(idx: int) -> float:
    if idx == 0:
        return 1.0
    if idx == 1:
        return -8.0
    if idx == 2:
        return 8.0
    return -1.0


@njit(cache=True)
def _dxy4(values: Array, i: int, j: int, k: int, dx: float) -> float:
    n = values.shape[0]
    total = 0.0
    for a in range(4):
        ia = _wrap_index(i + _fd4_offset(a), n)
        ca = _fd4_coeff(a)
        for b in range(4):
            jb = _wrap_index(j + _fd4_offset(b), n)
            total += ca * _fd4_coeff(b) * values[ia, jb, k]
    return total / (144.0 * dx * dx)


@njit(cache=True)
def _dxz4(values: Array, i: int, j: int, k: int, dx: float) -> float:
    n = values.shape[0]
    total = 0.0
    for a in range(4):
        ia = _wrap_index(i + _fd4_offset(a), n)
        ca = _fd4_coeff(a)
        for b in range(4):
            kb = _wrap_index(k + _fd4_offset(b), n)
            total += ca * _fd4_coeff(b) * values[ia, j, kb]
    return total / (144.0 * dx * dx)


@njit(cache=True)
def _dyz4(values: Array, i: int, j: int, k: int, dx: float) -> float:
    n = values.shape[0]
    total = 0.0
    for a in range(4):
        ja = _wrap_index(j + _fd4_offset(a), n)
        ca = _fd4_coeff(a)
        for b in range(4):
            kb = _wrap_index(k + _fd4_offset(b), n)
            total += ca * _fd4_coeff(b) * values[i, ja, kb]
    return total / (144.0 * dx * dx)


@njit(cache=True)
def _lap4(values: Array, i: int, j: int, k: int, dx: float) -> float:
    return _d2x4(values, i, j, k, dx) + _d2y4(values, i, j, k, dx) + _d2z4(
        values, i, j, k, dx
    )


@njit(cache=True)
def _psi_d1x4(phi: Array, pi: Array, t: float, i: int, j: int, k: int, dx: float) -> float:
    n = phi.shape[0]
    im2 = _wrap_index(i - 2, n)
    im1 = _wrap_index(i - 1, n)
    ip1 = _wrap_index(i + 1, n)
    ip2 = _wrap_index(i + 2, n)
    return (
        (t * pi[im2, j, k] + phi[im2, j, k])
        - 8.0 * (t * pi[im1, j, k] + phi[im1, j, k])
        + 8.0 * (t * pi[ip1, j, k] + phi[ip1, j, k])
        - (t * pi[ip2, j, k] + phi[ip2, j, k])
    ) / (12.0 * dx)


@njit(cache=True)
def _psi_d1y4(phi: Array, pi: Array, t: float, i: int, j: int, k: int, dx: float) -> float:
    n = phi.shape[0]
    jm2 = _wrap_index(j - 2, n)
    jm1 = _wrap_index(j - 1, n)
    jp1 = _wrap_index(j + 1, n)
    jp2 = _wrap_index(j + 2, n)
    return (
        (t * pi[i, jm2, k] + phi[i, jm2, k])
        - 8.0 * (t * pi[i, jm1, k] + phi[i, jm1, k])
        + 8.0 * (t * pi[i, jp1, k] + phi[i, jp1, k])
        - (t * pi[i, jp2, k] + phi[i, jp2, k])
    ) / (12.0 * dx)


@njit(cache=True)
def _psi_d1z4(phi: Array, pi: Array, t: float, i: int, j: int, k: int, dx: float) -> float:
    n = phi.shape[0]
    km2 = _wrap_index(k - 2, n)
    km1 = _wrap_index(k - 1, n)
    kp1 = _wrap_index(k + 1, n)
    kp2 = _wrap_index(k + 2, n)
    return (
        (t * pi[i, j, km2] + phi[i, j, km2])
        - 8.0 * (t * pi[i, j, km1] + phi[i, j, km1])
        + 8.0 * (t * pi[i, j, kp1] + phi[i, j, kp1])
        - (t * pi[i, j, kp2] + phi[i, j, kp2])
    ) / (12.0 * dx)


@njit(parallel=True, fastmath=True, cache=True)
def _laplacian_fd(values: Array, dx: float) -> Array:
    n = values.shape[0]
    out = np.empty_like(values)
    inv_dx2 = 1.0 / (dx * dx)

    for i in prange(n):
        ip = 0 if i == n - 1 else i + 1
        im = n - 1 if i == 0 else i - 1
        for j in range(n):
            jp = 0 if j == n - 1 else j + 1
            jm = n - 1 if j == 0 else j - 1
            for k in range(n):
                kp = 0 if k == n - 1 else k + 1
                km = n - 1 if k == 0 else k - 1
                out[i, j, k] = (
                    values[ip, j, k]
                    + values[im, j, k]
                    + values[i, jp, k]
                    + values[i, jm, k]
                    + values[i, j, kp]
                    + values[i, j, km]
                    - 6.0 * values[i, j, k]
                ) * inv_dx2
    return out


@njit(parallel=True, fastmath=True, cache=True)
def _scalar_rhs_fd(phi: Array, pi: Array, t: float, dx: float) -> tuple[Array, Array]:
    n = phi.shape[0]
    dphi = np.empty_like(phi)
    dpi = np.empty_like(phi)
    inv_dx2 = 1.0 / (dx * dx)

    for i in prange(n):
        ip = 0 if i == n - 1 else i + 1
        im = n - 1 if i == 0 else i - 1
        for j in range(n):
            jp = 0 if j == n - 1 else j + 1
            jm = n - 1 if j == 0 else j - 1
            for k in range(n):
                kp = 0 if k == n - 1 else k + 1
                km = n - 1 if k == 0 else k - 1
                lap = (
                    phi[ip, j, k]
                    + phi[im, j, k]
                    + phi[i, jp, k]
                    + phi[i, jm, k]
                    + phi[i, j, kp]
                    + phi[i, j, km]
                    - 6.0 * phi[i, j, k]
                ) * inv_dx2
                dphi[i, j, k] = pi[i, j, k]
                dpi[i, j, k] = (1.0 / 3.0) * lap - (4.0 / t) * pi[i, j, k]
    return dphi, dpi


@njit(parallel=True, fastmath=True, cache=True)
def _scalar_rhs_fd4(phi: Array, pi: Array, t: float, dx: float) -> tuple[Array, Array]:
    n = phi.shape[0]
    dphi = np.empty_like(phi)
    dpi = np.empty_like(phi)

    for i in prange(n):
        for j in range(n):
            for k in range(n):
                dphi[i, j, k] = pi[i, j, k]
                dpi[i, j, k] = (1.0 / 3.0) * _lap4(phi, i, j, k, dx) - (
                    4.0 / t
                ) * pi[i, j, k]
    return dphi, dpi


@njit(parallel=True, fastmath=True, cache=True)
def _rk4_combine(
    phi: Array,
    pi: Array,
    dphi1: Array,
    dpi1: Array,
    dphi2: Array,
    dpi2: Array,
    dphi3: Array,
    dpi3: Array,
    dphi4: Array,
    dpi4: Array,
    dt: float,
) -> tuple[Array, Array]:
    n = phi.shape[0]
    phi_next = np.empty_like(phi)
    pi_next = np.empty_like(phi)
    dt6 = dt / 6.0

    for i in prange(n):
        for j in range(n):
            for k in range(n):
                phi_next[i, j, k] = phi[i, j, k] + dt6 * (
                    dphi1[i, j, k]
                    + 2.0 * dphi2[i, j, k]
                    + 2.0 * dphi3[i, j, k]
                    + dphi4[i, j, k]
                )
                pi_next[i, j, k] = pi[i, j, k] + dt6 * (
                    dpi1[i, j, k]
                    + 2.0 * dpi2[i, j, k]
                    + 2.0 * dpi3[i, j, k]
                    + dpi4[i, j, k]
                )
    return phi_next, pi_next


@njit(parallel=True, fastmath=True, cache=True)
def _rk4_scalar_step_fd2_fast(
    phi: Array,
    pi: Array,
    t: float,
    dt: float,
    dx: float,
) -> tuple[Array, Array]:
    n = phi.shape[0]
    phi2 = np.empty_like(phi)
    pi2 = np.empty_like(phi)
    dpi1 = np.empty_like(phi)
    phi3 = np.empty_like(phi)
    pi3 = np.empty_like(phi)
    dpi2 = np.empty_like(phi)
    phi4 = np.empty_like(phi)
    pi4 = np.empty_like(phi)
    dpi3 = np.empty_like(phi)
    phi_next = np.empty_like(phi)
    pi_next = np.empty_like(phi)

    inv_dx2 = 1.0 / (dx * dx)
    half_dt = 0.5 * dt

    for i in prange(n):
        ip = 0 if i == n - 1 else i + 1
        im = n - 1 if i == 0 else i - 1
        for j in range(n):
            jp = 0 if j == n - 1 else j + 1
            jm = n - 1 if j == 0 else j - 1
            for k in range(n):
                kp = 0 if k == n - 1 else k + 1
                km = n - 1 if k == 0 else k - 1
                lap = (
                    phi[ip, j, k]
                    + phi[im, j, k]
                    + phi[i, jp, k]
                    + phi[i, jm, k]
                    + phi[i, j, kp]
                    + phi[i, j, km]
                    - 6.0 * phi[i, j, k]
                ) * inv_dx2
                k1 = (1.0 / 3.0) * lap - (4.0 / t) * pi[i, j, k]
                dpi1[i, j, k] = k1
                phi2[i, j, k] = phi[i, j, k] + half_dt * pi[i, j, k]
                pi2[i, j, k] = pi[i, j, k] + half_dt * k1

    t_half = t + half_dt
    for i in prange(n):
        ip = 0 if i == n - 1 else i + 1
        im = n - 1 if i == 0 else i - 1
        for j in range(n):
            jp = 0 if j == n - 1 else j + 1
            jm = n - 1 if j == 0 else j - 1
            for k in range(n):
                kp = 0 if k == n - 1 else k + 1
                km = n - 1 if k == 0 else k - 1
                lap = (
                    phi2[ip, j, k]
                    + phi2[im, j, k]
                    + phi2[i, jp, k]
                    + phi2[i, jm, k]
                    + phi2[i, j, kp]
                    + phi2[i, j, km]
                    - 6.0 * phi2[i, j, k]
                ) * inv_dx2
                k2 = (1.0 / 3.0) * lap - (4.0 / t_half) * pi2[i, j, k]
                dpi2[i, j, k] = k2
                phi3[i, j, k] = phi[i, j, k] + half_dt * pi2[i, j, k]
                pi3[i, j, k] = pi[i, j, k] + half_dt * k2

    for i in prange(n):
        ip = 0 if i == n - 1 else i + 1
        im = n - 1 if i == 0 else i - 1
        for j in range(n):
            jp = 0 if j == n - 1 else j + 1
            jm = n - 1 if j == 0 else j - 1
            for k in range(n):
                kp = 0 if k == n - 1 else k + 1
                km = n - 1 if k == 0 else k - 1
                lap = (
                    phi3[ip, j, k]
                    + phi3[im, j, k]
                    + phi3[i, jp, k]
                    + phi3[i, jm, k]
                    + phi3[i, j, kp]
                    + phi3[i, j, km]
                    - 6.0 * phi3[i, j, k]
                ) * inv_dx2
                k3 = (1.0 / 3.0) * lap - (4.0 / t_half) * pi3[i, j, k]
                dpi3[i, j, k] = k3
                phi4[i, j, k] = phi[i, j, k] + dt * pi3[i, j, k]
                pi4[i, j, k] = pi[i, j, k] + dt * k3

    t_full = t + dt
    dt6 = dt / 6.0
    for i in prange(n):
        ip = 0 if i == n - 1 else i + 1
        im = n - 1 if i == 0 else i - 1
        for j in range(n):
            jp = 0 if j == n - 1 else j + 1
            jm = n - 1 if j == 0 else j - 1
            for k in range(n):
                kp = 0 if k == n - 1 else k + 1
                km = n - 1 if k == 0 else k - 1
                lap = (
                    phi4[ip, j, k]
                    + phi4[im, j, k]
                    + phi4[i, jp, k]
                    + phi4[i, jm, k]
                    + phi4[i, j, kp]
                    + phi4[i, j, km]
                    - 6.0 * phi4[i, j, k]
                ) * inv_dx2
                k4 = (1.0 / 3.0) * lap - (4.0 / t_full) * pi4[i, j, k]
                phi_next[i, j, k] = phi[i, j, k] + dt6 * (
                    pi[i, j, k] + 2.0 * pi2[i, j, k] + 2.0 * pi3[i, j, k] + pi4[i, j, k]
                )
                pi_next[i, j, k] = pi[i, j, k] + dt6 * (
                    dpi1[i, j, k] + 2.0 * dpi2[i, j, k] + 2.0 * dpi3[i, j, k] + k4
                )

    return phi_next, pi_next


@njit(parallel=True, fastmath=True, cache=True)
def _rk4_scalar_step_fd2_into(
    phi: Array,
    pi: Array,
    t: float,
    dt: float,
    dx: float,
    phi2: Array,
    pi2: Array,
    dpi1: Array,
    phi3: Array,
    pi3: Array,
    dpi2: Array,
    phi4: Array,
    pi4: Array,
    dpi3: Array,
    phi_next: Array,
    pi_next: Array,
) -> tuple[Array, Array]:
    n = phi.shape[0]
    inv_dx2 = 1.0 / (dx * dx)
    half_dt = 0.5 * dt

    for i in prange(n):
        ip = 0 if i == n - 1 else i + 1
        im = n - 1 if i == 0 else i - 1
        for j in range(n):
            jp = 0 if j == n - 1 else j + 1
            jm = n - 1 if j == 0 else j - 1
            for k in range(n):
                kp = 0 if k == n - 1 else k + 1
                km = n - 1 if k == 0 else k - 1
                lap = (
                    phi[ip, j, k]
                    + phi[im, j, k]
                    + phi[i, jp, k]
                    + phi[i, jm, k]
                    + phi[i, j, kp]
                    + phi[i, j, km]
                    - 6.0 * phi[i, j, k]
                ) * inv_dx2
                k1 = (1.0 / 3.0) * lap - (4.0 / t) * pi[i, j, k]
                dpi1[i, j, k] = k1
                phi2[i, j, k] = phi[i, j, k] + half_dt * pi[i, j, k]
                pi2[i, j, k] = pi[i, j, k] + half_dt * k1

    t_half = t + half_dt
    for i in prange(n):
        ip = 0 if i == n - 1 else i + 1
        im = n - 1 if i == 0 else i - 1
        for j in range(n):
            jp = 0 if j == n - 1 else j + 1
            jm = n - 1 if j == 0 else j - 1
            for k in range(n):
                kp = 0 if k == n - 1 else k + 1
                km = n - 1 if k == 0 else k - 1
                lap = (
                    phi2[ip, j, k]
                    + phi2[im, j, k]
                    + phi2[i, jp, k]
                    + phi2[i, jm, k]
                    + phi2[i, j, kp]
                    + phi2[i, j, km]
                    - 6.0 * phi2[i, j, k]
                ) * inv_dx2
                k2 = (1.0 / 3.0) * lap - (4.0 / t_half) * pi2[i, j, k]
                dpi2[i, j, k] = k2
                phi3[i, j, k] = phi[i, j, k] + half_dt * pi2[i, j, k]
                pi3[i, j, k] = pi[i, j, k] + half_dt * k2

    for i in prange(n):
        ip = 0 if i == n - 1 else i + 1
        im = n - 1 if i == 0 else i - 1
        for j in range(n):
            jp = 0 if j == n - 1 else j + 1
            jm = n - 1 if j == 0 else j - 1
            for k in range(n):
                kp = 0 if k == n - 1 else k + 1
                km = n - 1 if k == 0 else k - 1
                lap = (
                    phi3[ip, j, k]
                    + phi3[im, j, k]
                    + phi3[i, jp, k]
                    + phi3[i, jm, k]
                    + phi3[i, j, kp]
                    + phi3[i, j, km]
                    - 6.0 * phi3[i, j, k]
                ) * inv_dx2
                k3 = (1.0 / 3.0) * lap - (4.0 / t_half) * pi3[i, j, k]
                dpi3[i, j, k] = k3
                phi4[i, j, k] = phi[i, j, k] + dt * pi3[i, j, k]
                pi4[i, j, k] = pi[i, j, k] + dt * k3

    t_full = t + dt
    dt6 = dt / 6.0
    for i in prange(n):
        ip = 0 if i == n - 1 else i + 1
        im = n - 1 if i == 0 else i - 1
        for j in range(n):
            jp = 0 if j == n - 1 else j + 1
            jm = n - 1 if j == 0 else j - 1
            for k in range(n):
                kp = 0 if k == n - 1 else k + 1
                km = n - 1 if k == 0 else k - 1
                lap = (
                    phi4[ip, j, k]
                    + phi4[im, j, k]
                    + phi4[i, jp, k]
                    + phi4[i, jm, k]
                    + phi4[i, j, kp]
                    + phi4[i, j, km]
                    - 6.0 * phi4[i, j, k]
                ) * inv_dx2
                k4 = (1.0 / 3.0) * lap - (4.0 / t_full) * pi4[i, j, k]
                phi_next[i, j, k] = phi[i, j, k] + dt6 * (
                    pi[i, j, k] + 2.0 * pi2[i, j, k] + 2.0 * pi3[i, j, k] + pi4[i, j, k]
                )
                pi_next[i, j, k] = pi[i, j, k] + dt6 * (
                    dpi1[i, j, k] + 2.0 * dpi2[i, j, k] + 2.0 * dpi3[i, j, k] + k4
                )

    return phi_next, pi_next


@njit(parallel=True, fastmath=True, cache=True)
def _rk4_scalar_step_fd4_fast(
    phi: Array,
    pi: Array,
    t: float,
    dt: float,
    dx: float,
) -> tuple[Array, Array]:
    n = phi.shape[0]
    phi2 = np.empty_like(phi)
    pi2 = np.empty_like(phi)
    dpi1 = np.empty_like(phi)
    phi3 = np.empty_like(phi)
    pi3 = np.empty_like(phi)
    dpi2 = np.empty_like(phi)
    phi4 = np.empty_like(phi)
    pi4 = np.empty_like(phi)
    dpi3 = np.empty_like(phi)
    phi_next = np.empty_like(phi)
    pi_next = np.empty_like(phi)

    half_dt = 0.5 * dt

    for i in prange(n):
        for j in range(n):
            for k in range(n):
                k1 = (1.0 / 3.0) * _lap4(phi, i, j, k, dx) - (4.0 / t) * pi[i, j, k]
                dpi1[i, j, k] = k1
                phi2[i, j, k] = phi[i, j, k] + half_dt * pi[i, j, k]
                pi2[i, j, k] = pi[i, j, k] + half_dt * k1

    t_half = t + half_dt
    for i in prange(n):
        for j in range(n):
            for k in range(n):
                k2 = (1.0 / 3.0) * _lap4(phi2, i, j, k, dx) - (
                    4.0 / t_half
                ) * pi2[i, j, k]
                dpi2[i, j, k] = k2
                phi3[i, j, k] = phi[i, j, k] + half_dt * pi2[i, j, k]
                pi3[i, j, k] = pi[i, j, k] + half_dt * k2

    for i in prange(n):
        for j in range(n):
            for k in range(n):
                k3 = (1.0 / 3.0) * _lap4(phi3, i, j, k, dx) - (
                    4.0 / t_half
                ) * pi3[i, j, k]
                dpi3[i, j, k] = k3
                phi4[i, j, k] = phi[i, j, k] + dt * pi3[i, j, k]
                pi4[i, j, k] = pi[i, j, k] + dt * k3

    t_full = t + dt
    dt6 = dt / 6.0
    for i in prange(n):
        for j in range(n):
            for k in range(n):
                k4 = (1.0 / 3.0) * _lap4(phi4, i, j, k, dx) - (
                    4.0 / t_full
                ) * pi4[i, j, k]
                phi_next[i, j, k] = phi[i, j, k] + dt6 * (
                    pi[i, j, k] + 2.0 * pi2[i, j, k] + 2.0 * pi3[i, j, k] + pi4[i, j, k]
                )
                pi_next[i, j, k] = pi[i, j, k] + dt6 * (
                    dpi1[i, j, k] + 2.0 * dpi2[i, j, k] + 2.0 * dpi3[i, j, k] + k4
                )

    return phi_next, pi_next


@njit(parallel=True, fastmath=True, cache=True)
def _rk4_scalar_step_fd4_into(
    phi: Array,
    pi: Array,
    t: float,
    dt: float,
    dx: float,
    phi2: Array,
    pi2: Array,
    dpi1: Array,
    phi3: Array,
    pi3: Array,
    dpi2: Array,
    phi4: Array,
    pi4: Array,
    dpi3: Array,
    phi_next: Array,
    pi_next: Array,
) -> tuple[Array, Array]:
    n = phi.shape[0]
    half_dt = 0.5 * dt

    for i in prange(n):
        for j in range(n):
            for k in range(n):
                k1 = (1.0 / 3.0) * _lap4(phi, i, j, k, dx) - (4.0 / t) * pi[i, j, k]
                dpi1[i, j, k] = k1
                phi2[i, j, k] = phi[i, j, k] + half_dt * pi[i, j, k]
                pi2[i, j, k] = pi[i, j, k] + half_dt * k1

    t_half = t + half_dt
    for i in prange(n):
        for j in range(n):
            for k in range(n):
                k2 = (1.0 / 3.0) * _lap4(phi2, i, j, k, dx) - (
                    4.0 / t_half
                ) * pi2[i, j, k]
                dpi2[i, j, k] = k2
                phi3[i, j, k] = phi[i, j, k] + half_dt * pi2[i, j, k]
                pi3[i, j, k] = pi[i, j, k] + half_dt * k2

    for i in prange(n):
        for j in range(n):
            for k in range(n):
                k3 = (1.0 / 3.0) * _lap4(phi3, i, j, k, dx) - (
                    4.0 / t_half
                ) * pi3[i, j, k]
                dpi3[i, j, k] = k3
                phi4[i, j, k] = phi[i, j, k] + dt * pi3[i, j, k]
                pi4[i, j, k] = pi[i, j, k] + dt * k3

    t_full = t + dt
    dt6 = dt / 6.0
    for i in prange(n):
        for j in range(n):
            for k in range(n):
                k4 = (1.0 / 3.0) * _lap4(phi4, i, j, k, dx) - (
                    4.0 / t_full
                ) * pi4[i, j, k]
                phi_next[i, j, k] = phi[i, j, k] + dt6 * (
                    pi[i, j, k] + 2.0 * pi2[i, j, k] + 2.0 * pi3[i, j, k] + pi4[i, j, k]
                )
                pi_next[i, j, k] = pi[i, j, k] + dt6 * (
                    dpi1[i, j, k] + 2.0 * dpi2[i, j, k] + 2.0 * dpi3[i, j, k] + k4
                )

    return phi_next, pi_next


@njit(parallel=True, fastmath=True, cache=True)
def _tensor_acceleration_fd(
    phi: Array,
    pi: Array,
    h11: Array,
    h12: Array,
    h13: Array,
    h22: Array,
    h23: Array,
    h33: Array,
    t: float,
    dx: float,
) -> Tensor6:
    n = phi.shape[0]
    ac11 = np.empty_like(phi)
    ac12 = np.empty_like(phi)
    ac13 = np.empty_like(phi)
    ac22 = np.empty_like(phi)
    ac23 = np.empty_like(phi)
    ac33 = np.empty_like(phi)

    inv_2dx = 0.5 / dx
    inv_dx2 = 1.0 / (dx * dx)
    inv_4dx2 = 0.25 / (dx * dx)

    for i in prange(n):
        ip = 0 if i == n - 1 else i + 1
        im = n - 1 if i == 0 else i - 1
        for j in range(n):
            jp = 0 if j == n - 1 else j + 1
            jm = n - 1 if j == 0 else j - 1
            for k in range(n):
                kp = 0 if k == n - 1 else k + 1
                km = n - 1 if k == 0 else k - 1

                phi0 = phi[i, j, k]
                psi0 = t * pi[i, j, k] + phi0

                phix = (phi[ip, j, k] - phi[im, j, k]) * inv_2dx
                phiy = (phi[i, jp, k] - phi[i, jm, k]) * inv_2dx
                phiz = (phi[i, j, kp] - phi[i, j, km]) * inv_2dx

                psix = (
                    t * pi[ip, j, k]
                    + phi[ip, j, k]
                    - t * pi[im, j, k]
                    - phi[im, j, k]
                ) * inv_2dx
                psiy = (
                    t * pi[i, jp, k]
                    + phi[i, jp, k]
                    - t * pi[i, jm, k]
                    - phi[i, jm, k]
                ) * inv_2dx
                psiz = (
                    t * pi[i, j, kp]
                    + phi[i, j, kp]
                    - t * pi[i, j, km]
                    - phi[i, j, km]
                ) * inv_2dx

                phixx = (phi[ip, j, k] - 2.0 * phi0 + phi[im, j, k]) * inv_dx2
                phiyy = (phi[i, jp, k] - 2.0 * phi0 + phi[i, jm, k]) * inv_dx2
                phizz = (phi[i, j, kp] - 2.0 * phi0 + phi[i, j, km]) * inv_dx2
                phixy = (
                    phi[ip, jp, k]
                    - phi[ip, jm, k]
                    - phi[im, jp, k]
                    + phi[im, jm, k]
                ) * inv_4dx2
                phixz = (
                    phi[ip, j, kp]
                    - phi[ip, j, km]
                    - phi[im, j, kp]
                    + phi[im, j, km]
                ) * inv_4dx2
                phiyz = (
                    phi[i, jp, kp]
                    - phi[i, jp, km]
                    - phi[i, jm, kp]
                    + phi[i, jm, km]
                ) * inv_4dx2

                lap11 = (
                    h11[ip, j, k]
                    + h11[im, j, k]
                    + h11[i, jp, k]
                    + h11[i, jm, k]
                    + h11[i, j, kp]
                    + h11[i, j, km]
                    - 6.0 * h11[i, j, k]
                ) * inv_dx2
                lap12 = (
                    h12[ip, j, k]
                    + h12[im, j, k]
                    + h12[i, jp, k]
                    + h12[i, jm, k]
                    + h12[i, j, kp]
                    + h12[i, j, km]
                    - 6.0 * h12[i, j, k]
                ) * inv_dx2
                lap13 = (
                    h13[ip, j, k]
                    + h13[im, j, k]
                    + h13[i, jp, k]
                    + h13[i, jm, k]
                    + h13[i, j, kp]
                    + h13[i, j, km]
                    - 6.0 * h13[i, j, k]
                ) * inv_dx2
                lap22 = (
                    h22[ip, j, k]
                    + h22[im, j, k]
                    + h22[i, jp, k]
                    + h22[i, jm, k]
                    + h22[i, j, kp]
                    + h22[i, j, km]
                    - 6.0 * h22[i, j, k]
                ) * inv_dx2
                lap23 = (
                    h23[ip, j, k]
                    + h23[im, j, k]
                    + h23[i, jp, k]
                    + h23[i, jm, k]
                    + h23[i, j, kp]
                    + h23[i, j, km]
                    - 6.0 * h23[i, j, k]
                ) * inv_dx2
                lap33 = (
                    h33[ip, j, k]
                    + h33[im, j, k]
                    + h33[i, jp, k]
                    + h33[i, jm, k]
                    + h33[i, j, kp]
                    + h33[i, j, km]
                    - 6.0 * h33[i, j, k]
                ) * inv_dx2

                ac11[i, j, k] = lap11 - 4.0 * t * (
                    4.0 * phi0 * phixx + 2.0 * phix * phix - psix * psix
                )
                ac12[i, j, k] = lap12 - 4.0 * t * (
                    4.0 * phi0 * phixy + 2.0 * phix * phiy - psix * psiy
                )
                ac13[i, j, k] = lap13 - 4.0 * t * (
                    4.0 * phi0 * phixz + 2.0 * phix * phiz - psix * psiz
                )
                ac22[i, j, k] = lap22 - 4.0 * t * (
                    4.0 * phi0 * phiyy + 2.0 * phiy * phiy - psiy * psiy
                )
                ac23[i, j, k] = lap23 - 4.0 * t * (
                    4.0 * phi0 * phiyz + 2.0 * phiy * phiz - psiy * psiz
                )
                ac33[i, j, k] = lap33 - 4.0 * t * (
                    4.0 * phi0 * phizz + 2.0 * phiz * phiz - psiz * psiz
                )

    return ac11, ac12, ac13, ac22, ac23, ac33


@njit(parallel=True, fastmath=True, cache=True)
def _tensor_acceleration_fd4(
    phi: Array,
    pi: Array,
    h11: Array,
    h12: Array,
    h13: Array,
    h22: Array,
    h23: Array,
    h33: Array,
    t: float,
    dx: float,
) -> Tensor6:
    n = phi.shape[0]
    ac11 = np.empty_like(phi)
    ac12 = np.empty_like(phi)
    ac13 = np.empty_like(phi)
    ac22 = np.empty_like(phi)
    ac23 = np.empty_like(phi)
    ac33 = np.empty_like(phi)

    for i in prange(n):
        for j in range(n):
            for k in range(n):
                phi0 = phi[i, j, k]

                phix = _d1x4(phi, i, j, k, dx)
                phiy = _d1y4(phi, i, j, k, dx)
                phiz = _d1z4(phi, i, j, k, dx)
                psix = _psi_d1x4(phi, pi, t, i, j, k, dx)
                psiy = _psi_d1y4(phi, pi, t, i, j, k, dx)
                psiz = _psi_d1z4(phi, pi, t, i, j, k, dx)

                phixx = _d2x4(phi, i, j, k, dx)
                phiyy = _d2y4(phi, i, j, k, dx)
                phizz = _d2z4(phi, i, j, k, dx)
                phixy = _dxy4(phi, i, j, k, dx)
                phixz = _dxz4(phi, i, j, k, dx)
                phiyz = _dyz4(phi, i, j, k, dx)

                ac11[i, j, k] = _lap4(h11, i, j, k, dx) - 4.0 * t * (
                    4.0 * phi0 * phixx + 2.0 * phix * phix - psix * psix
                )
                ac12[i, j, k] = _lap4(h12, i, j, k, dx) - 4.0 * t * (
                    4.0 * phi0 * phixy + 2.0 * phix * phiy - psix * psiy
                )
                ac13[i, j, k] = _lap4(h13, i, j, k, dx) - 4.0 * t * (
                    4.0 * phi0 * phixz + 2.0 * phix * phiz - psix * psiz
                )
                ac22[i, j, k] = _lap4(h22, i, j, k, dx) - 4.0 * t * (
                    4.0 * phi0 * phiyy + 2.0 * phiy * phiy - psiy * psiy
                )
                ac23[i, j, k] = _lap4(h23, i, j, k, dx) - 4.0 * t * (
                    4.0 * phi0 * phiyz + 2.0 * phiy * phiz - psiy * psiz
                )
                ac33[i, j, k] = _lap4(h33, i, j, k, dx) - 4.0 * t * (
                    4.0 * phi0 * phizz + 2.0 * phiz * phiz - psiz * psiz
                )

    return ac11, ac12, ac13, ac22, ac23, ac33


@njit(parallel=True, fastmath=True, cache=True)
def _leapfrog_update(
    h11: Array,
    h12: Array,
    h13: Array,
    h22: Array,
    h23: Array,
    h33: Array,
    v11: Array,
    v12: Array,
    v13: Array,
    v22: Array,
    v23: Array,
    v33: Array,
    ac11: Array,
    ac12: Array,
    ac13: Array,
    ac22: Array,
    ac23: Array,
    ac33: Array,
    dt: float,
) -> tuple[Tensor6, Tensor6]:
    n = h11.shape[0]
    hn11 = np.empty_like(h11)
    hn12 = np.empty_like(h11)
    hn13 = np.empty_like(h11)
    hn22 = np.empty_like(h11)
    hn23 = np.empty_like(h11)
    hn33 = np.empty_like(h11)
    vn11 = np.empty_like(h11)
    vn12 = np.empty_like(h11)
    vn13 = np.empty_like(h11)
    vn22 = np.empty_like(h11)
    vn23 = np.empty_like(h11)
    vn33 = np.empty_like(h11)

    for i in prange(n):
        for j in range(n):
            for k in range(n):
                vn11[i, j, k] = v11[i, j, k] + dt * ac11[i, j, k]
                vn12[i, j, k] = v12[i, j, k] + dt * ac12[i, j, k]
                vn13[i, j, k] = v13[i, j, k] + dt * ac13[i, j, k]
                vn22[i, j, k] = v22[i, j, k] + dt * ac22[i, j, k]
                vn23[i, j, k] = v23[i, j, k] + dt * ac23[i, j, k]
                vn33[i, j, k] = v33[i, j, k] + dt * ac33[i, j, k]

                hn11[i, j, k] = h11[i, j, k] + dt * vn11[i, j, k]
                hn12[i, j, k] = h12[i, j, k] + dt * vn12[i, j, k]
                hn13[i, j, k] = h13[i, j, k] + dt * vn13[i, j, k]
                hn22[i, j, k] = h22[i, j, k] + dt * vn22[i, j, k]
                hn23[i, j, k] = h23[i, j, k] + dt * vn23[i, j, k]
                hn33[i, j, k] = h33[i, j, k] + dt * vn33[i, j, k]

    return (
        (hn11, hn12, hn13, hn22, hn23, hn33),
        (vn11, vn12, vn13, vn22, vn23, vn33),
    )


@njit(parallel=True, fastmath=True, cache=True)
def _leapfrog_tensor_step_fd2(
    phi: Array,
    pi: Array,
    h11: Array,
    h12: Array,
    h13: Array,
    h22: Array,
    h23: Array,
    h33: Array,
    v11: Array,
    v12: Array,
    v13: Array,
    v22: Array,
    v23: Array,
    v33: Array,
    t: float,
    dt: float,
    dx: float,
) -> tuple[Tensor6, Tensor6]:
    n = phi.shape[0]
    hn11 = np.empty_like(h11)
    hn12 = np.empty_like(h11)
    hn13 = np.empty_like(h11)
    hn22 = np.empty_like(h11)
    hn23 = np.empty_like(h11)
    hn33 = np.empty_like(h11)
    vn11 = np.empty_like(h11)
    vn12 = np.empty_like(h11)
    vn13 = np.empty_like(h11)
    vn22 = np.empty_like(h11)
    vn23 = np.empty_like(h11)
    vn33 = np.empty_like(h11)

    inv_2dx = 0.5 / dx
    inv_dx2 = 1.0 / (dx * dx)
    inv_4dx2 = 0.25 / (dx * dx)

    for i in prange(n):
        ip = 0 if i == n - 1 else i + 1
        im = n - 1 if i == 0 else i - 1
        for j in range(n):
            jp = 0 if j == n - 1 else j + 1
            jm = n - 1 if j == 0 else j - 1
            for k in range(n):
                kp = 0 if k == n - 1 else k + 1
                km = n - 1 if k == 0 else k - 1

                phi0 = phi[i, j, k]

                phix = (phi[ip, j, k] - phi[im, j, k]) * inv_2dx
                phiy = (phi[i, jp, k] - phi[i, jm, k]) * inv_2dx
                phiz = (phi[i, j, kp] - phi[i, j, km]) * inv_2dx

                psix = (
                    t * pi[ip, j, k]
                    + phi[ip, j, k]
                    - t * pi[im, j, k]
                    - phi[im, j, k]
                ) * inv_2dx
                psiy = (
                    t * pi[i, jp, k]
                    + phi[i, jp, k]
                    - t * pi[i, jm, k]
                    - phi[i, jm, k]
                ) * inv_2dx
                psiz = (
                    t * pi[i, j, kp]
                    + phi[i, j, kp]
                    - t * pi[i, j, km]
                    - phi[i, j, km]
                ) * inv_2dx

                phixx = (phi[ip, j, k] - 2.0 * phi0 + phi[im, j, k]) * inv_dx2
                phiyy = (phi[i, jp, k] - 2.0 * phi0 + phi[i, jm, k]) * inv_dx2
                phizz = (phi[i, j, kp] - 2.0 * phi0 + phi[i, j, km]) * inv_dx2
                phixy = (
                    phi[ip, jp, k]
                    - phi[ip, jm, k]
                    - phi[im, jp, k]
                    + phi[im, jm, k]
                ) * inv_4dx2
                phixz = (
                    phi[ip, j, kp]
                    - phi[ip, j, km]
                    - phi[im, j, kp]
                    + phi[im, j, km]
                ) * inv_4dx2
                phiyz = (
                    phi[i, jp, kp]
                    - phi[i, jp, km]
                    - phi[i, jm, kp]
                    + phi[i, jm, km]
                ) * inv_4dx2

                ac11 = (
                    h11[ip, j, k]
                    + h11[im, j, k]
                    + h11[i, jp, k]
                    + h11[i, jm, k]
                    + h11[i, j, kp]
                    + h11[i, j, km]
                    - 6.0 * h11[i, j, k]
                ) * inv_dx2 - 4.0 * t * (
                    4.0 * phi0 * phixx + 2.0 * phix * phix - psix * psix
                )
                ac12 = (
                    h12[ip, j, k]
                    + h12[im, j, k]
                    + h12[i, jp, k]
                    + h12[i, jm, k]
                    + h12[i, j, kp]
                    + h12[i, j, km]
                    - 6.0 * h12[i, j, k]
                ) * inv_dx2 - 4.0 * t * (
                    4.0 * phi0 * phixy + 2.0 * phix * phiy - psix * psiy
                )
                ac13 = (
                    h13[ip, j, k]
                    + h13[im, j, k]
                    + h13[i, jp, k]
                    + h13[i, jm, k]
                    + h13[i, j, kp]
                    + h13[i, j, km]
                    - 6.0 * h13[i, j, k]
                ) * inv_dx2 - 4.0 * t * (
                    4.0 * phi0 * phixz + 2.0 * phix * phiz - psix * psiz
                )
                ac22 = (
                    h22[ip, j, k]
                    + h22[im, j, k]
                    + h22[i, jp, k]
                    + h22[i, jm, k]
                    + h22[i, j, kp]
                    + h22[i, j, km]
                    - 6.0 * h22[i, j, k]
                ) * inv_dx2 - 4.0 * t * (
                    4.0 * phi0 * phiyy + 2.0 * phiy * phiy - psiy * psiy
                )
                ac23 = (
                    h23[ip, j, k]
                    + h23[im, j, k]
                    + h23[i, jp, k]
                    + h23[i, jm, k]
                    + h23[i, j, kp]
                    + h23[i, j, km]
                    - 6.0 * h23[i, j, k]
                ) * inv_dx2 - 4.0 * t * (
                    4.0 * phi0 * phiyz + 2.0 * phiy * phiz - psiy * psiz
                )
                ac33 = (
                    h33[ip, j, k]
                    + h33[im, j, k]
                    + h33[i, jp, k]
                    + h33[i, jm, k]
                    + h33[i, j, kp]
                    + h33[i, j, km]
                    - 6.0 * h33[i, j, k]
                ) * inv_dx2 - 4.0 * t * (
                    4.0 * phi0 * phizz + 2.0 * phiz * phiz - psiz * psiz
                )

                vn11[i, j, k] = v11[i, j, k] + dt * ac11
                vn12[i, j, k] = v12[i, j, k] + dt * ac12
                vn13[i, j, k] = v13[i, j, k] + dt * ac13
                vn22[i, j, k] = v22[i, j, k] + dt * ac22
                vn23[i, j, k] = v23[i, j, k] + dt * ac23
                vn33[i, j, k] = v33[i, j, k] + dt * ac33

                hn11[i, j, k] = h11[i, j, k] + dt * vn11[i, j, k]
                hn12[i, j, k] = h12[i, j, k] + dt * vn12[i, j, k]
                hn13[i, j, k] = h13[i, j, k] + dt * vn13[i, j, k]
                hn22[i, j, k] = h22[i, j, k] + dt * vn22[i, j, k]
                hn23[i, j, k] = h23[i, j, k] + dt * vn23[i, j, k]
                hn33[i, j, k] = h33[i, j, k] + dt * vn33[i, j, k]

    return (
        (hn11, hn12, hn13, hn22, hn23, hn33),
        (vn11, vn12, vn13, vn22, vn23, vn33),
    )


@njit(parallel=True, fastmath=True, cache=True)
def _leapfrog_tensor_step_fd4(
    phi: Array,
    pi: Array,
    h11: Array,
    h12: Array,
    h13: Array,
    h22: Array,
    h23: Array,
    h33: Array,
    v11: Array,
    v12: Array,
    v13: Array,
    v22: Array,
    v23: Array,
    v33: Array,
    t: float,
    dt: float,
    dx: float,
) -> tuple[Tensor6, Tensor6]:
    n = phi.shape[0]
    hn11 = np.empty_like(h11)
    hn12 = np.empty_like(h11)
    hn13 = np.empty_like(h11)
    hn22 = np.empty_like(h11)
    hn23 = np.empty_like(h11)
    hn33 = np.empty_like(h11)
    vn11 = np.empty_like(h11)
    vn12 = np.empty_like(h11)
    vn13 = np.empty_like(h11)
    vn22 = np.empty_like(h11)
    vn23 = np.empty_like(h11)
    vn33 = np.empty_like(h11)

    for i in prange(n):
        for j in range(n):
            for k in range(n):
                phi0 = phi[i, j, k]

                phix = _d1x4(phi, i, j, k, dx)
                phiy = _d1y4(phi, i, j, k, dx)
                phiz = _d1z4(phi, i, j, k, dx)
                psix = _psi_d1x4(phi, pi, t, i, j, k, dx)
                psiy = _psi_d1y4(phi, pi, t, i, j, k, dx)
                psiz = _psi_d1z4(phi, pi, t, i, j, k, dx)

                phixx = _d2x4(phi, i, j, k, dx)
                phiyy = _d2y4(phi, i, j, k, dx)
                phizz = _d2z4(phi, i, j, k, dx)
                phixy = _dxy4(phi, i, j, k, dx)
                phixz = _dxz4(phi, i, j, k, dx)
                phiyz = _dyz4(phi, i, j, k, dx)

                ac11 = _lap4(h11, i, j, k, dx) - 4.0 * t * (
                    4.0 * phi0 * phixx + 2.0 * phix * phix - psix * psix
                )
                ac12 = _lap4(h12, i, j, k, dx) - 4.0 * t * (
                    4.0 * phi0 * phixy + 2.0 * phix * phiy - psix * psiy
                )
                ac13 = _lap4(h13, i, j, k, dx) - 4.0 * t * (
                    4.0 * phi0 * phixz + 2.0 * phix * phiz - psix * psiz
                )
                ac22 = _lap4(h22, i, j, k, dx) - 4.0 * t * (
                    4.0 * phi0 * phiyy + 2.0 * phiy * phiy - psiy * psiy
                )
                ac23 = _lap4(h23, i, j, k, dx) - 4.0 * t * (
                    4.0 * phi0 * phiyz + 2.0 * phiy * phiz - psiy * psiz
                )
                ac33 = _lap4(h33, i, j, k, dx) - 4.0 * t * (
                    4.0 * phi0 * phizz + 2.0 * phiz * phiz - psiz * psiz
                )

                vn11[i, j, k] = v11[i, j, k] + dt * ac11
                vn12[i, j, k] = v12[i, j, k] + dt * ac12
                vn13[i, j, k] = v13[i, j, k] + dt * ac13
                vn22[i, j, k] = v22[i, j, k] + dt * ac22
                vn23[i, j, k] = v23[i, j, k] + dt * ac23
                vn33[i, j, k] = v33[i, j, k] + dt * ac33

                hn11[i, j, k] = h11[i, j, k] + dt * vn11[i, j, k]
                hn12[i, j, k] = h12[i, j, k] + dt * vn12[i, j, k]
                hn13[i, j, k] = h13[i, j, k] + dt * vn13[i, j, k]
                hn22[i, j, k] = h22[i, j, k] + dt * vn22[i, j, k]
                hn23[i, j, k] = h23[i, j, k] + dt * vn23[i, j, k]
                hn33[i, j, k] = h33[i, j, k] + dt * vn33[i, j, k]

    return (
        (hn11, hn12, hn13, hn22, hn23, hn33),
        (vn11, vn12, vn13, vn22, vn23, vn33),
    )


def rk4_scalar_step_fd(
    phi: Array,
    pi: Array,
    t: float,
    dt: float,
    dx: float,
    order: int = 2,
) -> tuple[Array, Array]:
    if order == 4:
        return _rk4_scalar_step_fd4_fast(phi, pi, t, dt, dx)
    return _rk4_scalar_step_fd2_fast(phi, pi, t, dt, dx)


def rk4_scalar_step_fd_reference(
    phi: Array,
    pi: Array,
    t: float,
    dt: float,
    dx: float,
    order: int = 2,
) -> tuple[Array, Array]:
    """Original allocation-heavy scalar RK4 step, kept for validation."""

    rhs = _scalar_rhs_fd4 if order == 4 else _scalar_rhs_fd
    dphi1, dpi1 = rhs(phi, pi, t, dx)
    dphi2, dpi2 = rhs(
        phi + 0.5 * dt * dphi1,
        pi + 0.5 * dt * dpi1,
        t + 0.5 * dt,
        dx,
    )
    dphi3, dpi3 = rhs(
        phi + 0.5 * dt * dphi2,
        pi + 0.5 * dt * dpi2,
        t + 0.5 * dt,
        dx,
    )
    dphi4, dpi4 = rhs(phi + dt * dphi3, pi + dt * dpi3, t + dt, dx)
    return _rk4_combine(phi, pi, dphi1, dpi1, dphi2, dpi2, dphi3, dpi3, dphi4, dpi4, dt)


def step_state_fd(state: EvolutionState, dt: float, dx: float, order: int = 2) -> EvolutionState:
    phi_next, pi_next = rk4_scalar_step_fd(state.phi, state.pi, state.t, dt, dx, order)
    tensor_step = _leapfrog_tensor_step_fd4 if order == 4 else _leapfrog_tensor_step_fd2
    h_next, v_next = tensor_step(
        state.phi,
        state.pi,
        state.h11,
        state.h12,
        state.h13,
        state.h22,
        state.h23,
        state.h33,
        state.v11,
        state.v12,
        state.v13,
        state.v22,
        state.v23,
        state.v33,
        state.t,
        dt,
        dx,
    )

    return EvolutionState(
        t=state.t + dt,
        phi=phi_next,
        pi=pi_next,
        h11=h_next[0],
        h12=h_next[1],
        h13=h_next[2],
        h22=h_next[3],
        h23=h_next[4],
        h33=h_next[5],
        v11=v_next[0],
        v12=v_next[1],
        v13=v_next[2],
        v22=v_next[3],
        v23=v_next[4],
        v33=v_next[5],
    )


def _scalar_workspace(
    phi: Array,
) -> tuple[Array, Array, Array, Array, Array, Array, Array, Array, Array, Array, Array, Array, Array]:
    return tuple(np.empty_like(phi) for _ in range(13))


def step_state_fd_workspace(
    state: EvolutionState,
    dt: float,
    dx: float,
    order: int,
    workspace: tuple[Array, Array, Array, Array, Array, Array, Array, Array, Array, Array, Array, Array, Array],
    output_slot: int,
) -> EvolutionState:
    phi_out_index = 9 + 2 * output_slot
    rk4_step = _rk4_scalar_step_fd4_into if order == 4 else _rk4_scalar_step_fd2_into
    tensor_step = _leapfrog_tensor_step_fd4 if order == 4 else _leapfrog_tensor_step_fd2
    phi_next, pi_next = rk4_step(
        state.phi,
        state.pi,
        state.t,
        dt,
        dx,
        workspace[0],
        workspace[1],
        workspace[2],
        workspace[3],
        workspace[4],
        workspace[5],
        workspace[6],
        workspace[7],
        workspace[8],
        workspace[phi_out_index],
        workspace[phi_out_index + 1],
    )
    h_next, v_next = tensor_step(
        state.phi,
        state.pi,
        state.h11,
        state.h12,
        state.h13,
        state.h22,
        state.h23,
        state.h33,
        state.v11,
        state.v12,
        state.v13,
        state.v22,
        state.v23,
        state.v33,
        state.t,
        dt,
        dx,
    )

    return EvolutionState(
        t=state.t + dt,
        phi=phi_next,
        pi=pi_next,
        h11=h_next[0],
        h12=h_next[1],
        h13=h_next[2],
        h22=h_next[3],
        h23=h_next[4],
        h33=h_next[5],
        v11=v_next[0],
        v12=v_next[1],
        v13=v_next[2],
        v22=v_next[3],
        v23=v_next[4],
        v33=v_next[5],
    )


def _normalize_fd_order(order: int) -> int:
    order = int(order)
    if order not in (2, 4):
        raise ValueError("fd_order must be 2 or 4")
    return order


def warmup_fd_backend(
    n: int = 8,
    workers: int = -1,
    order: int = 2,
    dtype: np.dtype = np.float64,
) -> None:
    """Compile Numba kernels before timing a real run."""

    order = _normalize_fd_order(order)
    set_num_threads(_normalize_workers(workers))
    config = EvolutionConfig(
        n=n,
        max_steps=1,
        output_every=0,
        save_initial=False,
        dtype=dtype,
    )
    phi = np.zeros((n, n, n), dtype=config.dtype)
    state = EvolutionState.from_phi(phi, config.initial_time, dtype=config.dtype)
    step_state_fd(state, config.dt, config.space_length * np.pi / n, order)
    if order == 2:
        workspace = _scalar_workspace(state.phi)
        step_state_fd_workspace(
            state, config.dt, config.space_length * np.pi / n, 2, workspace, 1
        )
    if order == 4:
        workspace = _scalar_workspace(state.phi)
        step_state_fd_workspace(
            state, config.dt, config.space_length * np.pi / n, 4, workspace, 1
        )


def run_evolution_fd(
    phi_initial: Array,
    config: EvolutionConfig = EvolutionConfig(),
    order: int | None = None,
    warmup: bool = True,
) -> EvolutionState:
    """Run one parameter point with Numba-parallel finite differences."""

    order = _normalize_fd_order(config.fd_order if order is None else order)
    workers = _normalize_workers(config.workers)
    set_num_threads(workers)

    if warmup:
        warmup_fd_backend(min(8, config.n), workers, order, config.dtype)

    dx = config.space_length * np.pi / config.n
    state = EvolutionState.from_phi(phi_initial, config.initial_time, dtype=config.dtype)
    scalar_workspace = _scalar_workspace(state.phi)

    if config.save_initial:
        save_snapshot(config.output_path, state, 0)

    for step_index in range(1, config.max_steps + 1):
        state = step_state_fd_workspace(
            state, config.dt, dx, order, scalar_workspace, step_index % 2
        )
        if config.output_every and step_index % config.output_every == 0:
            save_snapshot(config.output_path, state, step_index)

    return state


def numba_thread_count() -> int:
    return get_num_threads()
