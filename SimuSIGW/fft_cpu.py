"""CPU-parallel FFT implementation of the scalar and tensor evolution."""

from __future__ import annotations

import os
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterable

import numpy as np
from scipy import fft


Array = np.ndarray
Tensor6 = tuple[Array, Array, Array, Array, Array, Array]


@dataclass(frozen=True)
class EvolutionConfig:
    """Parameters for one simulation run.

    ``space_length`` keeps the notebook convention: the physical box length is
    ``space_length * pi``. ``workers=-1`` uses all available CPU workers for
    independent FFT tasks inside one parameter point. ``fft_workers`` is passed
    to each individual SciPy FFT; keep it at 1 unless that backend scales well
    on your machine.
    """

    n: int = 128
    space_length: float = 2.0
    max_steps: int = 1000
    output_every: int = 1000
    output_path: str | Path = "log/test_1"
    workers: int = -1
    fft_workers: int = 1
    fd_order: int = 2
    dtype: np.dtype = np.float64
    save_initial: bool = True

    @property
    def dt(self) -> float:
        return self.space_length / self.n / 5.0

    @property
    def initial_time(self) -> float:
        return self.space_length / self.n


@dataclass
class EvolutionState:
    """Dynamical variables at one time.

    ``hij`` is the evolved variable ``t * h_ij`` from the original notebook.
    ``vij`` is its time derivative.
    """

    t: float
    phi: Array
    pi: Array
    h11: Array
    h12: Array
    h13: Array
    h22: Array
    h23: Array
    h33: Array
    v11: Array
    v12: Array
    v13: Array
    v22: Array
    v23: Array
    v33: Array

    @classmethod
    def from_phi(cls, phi: Array, t: float, dtype: np.dtype = np.float64) -> "EvolutionState":
        phi = np.asarray(phi, dtype=dtype)
        zeros = np.zeros_like(phi)
        return cls(
            t=t,
            phi=phi.copy(),
            pi=zeros.copy(),
            h11=zeros.copy(),
            h12=zeros.copy(),
            h13=zeros.copy(),
            h22=zeros.copy(),
            h23=zeros.copy(),
            h33=zeros.copy(),
            v11=zeros.copy(),
            v12=zeros.copy(),
            v13=zeros.copy(),
            v22=zeros.copy(),
            v23=zeros.copy(),
            v33=zeros.copy(),
        )

    def h_components(self) -> Tensor6:
        return self.h11, self.h12, self.h13, self.h22, self.h23, self.h33

    def v_components(self) -> Tensor6:
        return self.v11, self.v12, self.v13, self.v22, self.v23, self.v33


class SpectralGrid:
    """Fourier pseudospectral derivatives on a cubic periodic grid."""

    def __init__(
        self,
        n: int,
        space_length: float = 2.0,
        workers: int = -1,
        fft_workers: int = 1,
        dtype: np.dtype = np.float64,
    ) -> None:
        self.n = int(n)
        self.space_length = float(space_length)
        self.workers = self._normalize_workers(workers)
        self.fft_workers = self._normalize_workers(fft_workers)
        self.dtype = dtype

        length = self.space_length * np.pi
        self.x = np.arange(self.n, dtype=dtype) * length / self.n

        k1 = (2.0 / self.space_length) * np.fft.fftfreq(self.n) * self.n
        self.kx, self.ky, self.kz = np.meshgrid(k1, k1, k1, indexing="ij")
        self.k2 = self.kx**2 + self.ky**2 + self.kz**2
        k_half = (2.0 / self.space_length) * np.fft.rfftfreq(self.n) * self.n
        kx_r, ky_r, kz_r = np.meshgrid(k1, k1, k_half, indexing="ij")
        self.k2_r = kx_r**2 + ky_r**2 + kz_r**2
        self._executor = (
            ThreadPoolExecutor(max_workers=self.workers) if self.workers > 1 else None
        )

    @staticmethod
    def _normalize_workers(workers: int) -> int:
        if workers == -1:
            return os.cpu_count() or 1
        return max(1, int(workers))

    def _run_tasks(self, tasks: Iterable[Callable[[], Array]]) -> list[Array]:
        tasks = list(tasks)
        if self.workers <= 1 or len(tasks) <= 1:
            return [task() for task in tasks]
        if self._executor is None:
            return [task() for task in tasks]
        return list(self._executor.map(lambda task: task(), tasks))

    def close(self) -> None:
        if self._executor is not None:
            self._executor.shutdown(wait=True)
            self._executor = None

    def fftn(self, values: Array) -> Array:
        return self.fftn_spatial(values)

    def ifftn_real(self, values_hat: Array) -> Array:
        return self.ifftn_real_spatial(values_hat)

    def fftn_spatial(self, values: Array) -> Array:
        """FFT over the last three axes.

        This also supports batched arrays with leading dimensions, e.g.
        ``(6, n, n, n)`` for the six tensor components.
        """

        return fft.fftn(values, axes=(-3, -2, -1), workers=self.fft_workers)

    def ifftn_real_spatial(self, values_hat: Array) -> Array:
        return fft.ifftn(values_hat, axes=(-3, -2, -1), workers=self.fft_workers).real

    def rfftn_spatial(self, values: Array) -> Array:
        return fft.rfftn(values, axes=(-3, -2, -1), workers=self.fft_workers)

    def irfftn_spatial(self, values_hat: Array) -> Array:
        return fft.irfftn(
            values_hat,
            s=(self.n, self.n, self.n),
            axes=(-3, -2, -1),
            workers=self.fft_workers,
        )

    def laplacian_from_hat(self, values_hat: Array) -> Array:
        return self.ifftn_real(-self.k2 * values_hat)

    def laplacian(self, values: Array) -> Array:
        return self.irfftn_spatial(-self.k2_r * self.rfftn_spatial(values))

    def laplacians(self, values: Tensor6) -> Tensor6:
        tasks = (
            lambda value=value: self.irfftn_spatial(-self.k2_r * self.rfftn_spatial(value))
            for value in values
        )
        return tuple(self._run_tasks(tasks))

    def gradient_from_hat(self, values_hat: Array) -> tuple[Array, Array, Array]:
        tasks = (
            lambda: self.ifftn_real_spatial(1j * self.kx * values_hat),
            lambda: self.ifftn_real_spatial(1j * self.ky * values_hat),
            lambda: self.ifftn_real_spatial(1j * self.kz * values_hat),
        )
        return tuple(self._run_tasks(tasks))

    def hessian_from_hat(self, values_hat: Array) -> Tensor6:
        tasks = (
            lambda: self.ifftn_real_spatial(-(self.kx * self.kx) * values_hat),
            lambda: self.ifftn_real_spatial(-(self.kx * self.ky) * values_hat),
            lambda: self.ifftn_real_spatial(-(self.kx * self.kz) * values_hat),
            lambda: self.ifftn_real_spatial(-(self.ky * self.ky) * values_hat),
            lambda: self.ifftn_real_spatial(-(self.ky * self.kz) * values_hat),
            lambda: self.ifftn_real_spatial(-(self.kz * self.kz) * values_hat),
        )
        return tuple(self._run_tasks(tasks))


def scalar_rhs(grid: SpectralGrid, phi: Array, pi: Array, t: float) -> tuple[Array, Array]:
    return pi, (1.0 / 3.0) * grid.laplacian(phi) - (4.0 / t) * pi


def scalar_rhs_hat(
    grid: SpectralGrid,
    phi_hat: Array,
    pi_hat: Array,
    t: float,
) -> tuple[Array, Array]:
    return pi_hat, -(1.0 / 3.0) * grid.k2_r * phi_hat - (4.0 / t) * pi_hat


def rk4_scalar_step(
    grid: SpectralGrid,
    phi: Array,
    pi: Array,
    t: float,
    dt: float,
) -> tuple[Array, Array]:
    phi_hat, pi_hat = grid._run_tasks(
        (
            lambda: grid.rfftn_spatial(phi),
            lambda: grid.rfftn_spatial(pi),
        )
    )
    dphi1, dpi1 = scalar_rhs_hat(grid, phi_hat, pi_hat, t)
    dphi2, dpi2 = scalar_rhs_hat(
        grid,
        phi_hat + 0.5 * dt * dphi1,
        pi_hat + 0.5 * dt * dpi1,
        t + 0.5 * dt,
    )
    dphi3, dpi3 = scalar_rhs_hat(
        grid,
        phi_hat + 0.5 * dt * dphi2,
        pi_hat + 0.5 * dt * dpi2,
        t + 0.5 * dt,
    )
    dphi4, dpi4 = scalar_rhs_hat(
        grid,
        phi_hat + dt * dphi3,
        pi_hat + dt * dpi3,
        t + dt,
    )

    phi_next_hat = phi_hat + (dt / 6.0) * (
        dphi1 + 2.0 * dphi2 + 2.0 * dphi3 + dphi4
    )
    pi_next_hat = pi_hat + (dt / 6.0) * (
        dpi1 + 2.0 * dpi2 + 2.0 * dpi3 + dpi4
    )
    phi_next, pi_next = grid._run_tasks(
        (
            lambda: grid.irfftn_spatial(phi_next_hat),
            lambda: grid.irfftn_spatial(pi_next_hat),
        )
    )
    return phi_next, pi_next


def tensor_acceleration(
    grid: SpectralGrid,
    phi: Array,
    pi: Array,
    h_components: Tensor6,
    t: float,
) -> Tensor6:
    """Acceleration of ``t * h_ij`` for all six independent components."""

    phi_hat, psi_hat = grid._run_tasks(
        (
            lambda: grid.fftn(phi),
            lambda: grid.fftn(t * pi + phi),
        )
    )

    phi_x, phi_y, phi_z = grid.gradient_from_hat(phi_hat)
    psi_x, psi_y, psi_z = grid.gradient_from_hat(psi_hat)
    phi_xx, phi_xy, phi_xz, phi_yy, phi_yz, phi_zz = grid.hessian_from_hat(phi_hat)

    sources = (
        4.0 * phi * phi_xx + 2.0 * phi_x**2 - psi_x**2,
        4.0 * phi * phi_xy + 2.0 * phi_x * phi_y - psi_x * psi_y,
        4.0 * phi * phi_xz + 2.0 * phi_x * phi_z - psi_x * psi_z,
        4.0 * phi * phi_yy + 2.0 * phi_y**2 - psi_y**2,
        4.0 * phi * phi_yz + 2.0 * phi_y * phi_z - psi_y * psi_z,
        4.0 * phi * phi_zz + 2.0 * phi_z**2 - psi_z**2,
    )

    h_laplacians = grid.laplacians(h_components)
    return tuple(
        h_laplacian - 4.0 * t * source
        for h_laplacian, source in zip(h_laplacians, sources, strict=True)
    )


def leapfrog_tensor_step(
    grid: SpectralGrid,
    phi: Array,
    pi: Array,
    h_components: Tensor6,
    v_components: Tensor6,
    t: float,
    dt: float,
) -> tuple[Tensor6, Tensor6]:
    acc = tensor_acceleration(grid, phi, pi, h_components, t)
    v_next = tuple(v + dt * a for v, a in zip(v_components, acc, strict=True))
    h_next = tuple(h + dt * v for h, v in zip(h_components, v_next, strict=True))
    return h_next, v_next


def step_state(grid: SpectralGrid, state: EvolutionState, dt: float) -> EvolutionState:
    phi_next, pi_next = rk4_scalar_step(grid, state.phi, state.pi, state.t, dt)
    h_next, v_next = leapfrog_tensor_step(
        grid,
        state.phi,
        state.pi,
        state.h_components(),
        state.v_components(),
        state.t,
        dt,
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


def save_snapshot(path: str | Path, state: EvolutionState, step_index: int) -> None:
    """Save the same fields as the original notebook, using the state's time."""

    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)

    h_real = tuple(h / state.t for h in state.h_components())
    a_real = tuple((v - h) / state.t for v, h in zip(state.v_components(), h_real, strict=True))

    names_and_values: Iterable[tuple[str, Array]] = (
        ("phi", state.phi),
        ("pi", state.pi),
        ("h11", h_real[0]),
        ("h12", h_real[1]),
        ("h13", h_real[2]),
        ("h22", h_real[3]),
        ("h23", h_real[4]),
        ("h33", h_real[5]),
        ("a11", a_real[0]),
        ("a12", a_real[1]),
        ("a13", a_real[2]),
        ("a22", a_real[3]),
        ("a23", a_real[4]),
        ("a33", a_real[5]),
    )
    for name, values in names_and_values:
        np.save(path / f"{name}_{step_index}.npy", values)


def run_evolution(
    phi_initial: Array,
    config: EvolutionConfig = EvolutionConfig(),
) -> EvolutionState:
    """Run one parameter point with CPU-parallel FFT evolution."""

    grid = SpectralGrid(
        config.n,
        space_length=config.space_length,
        workers=config.workers,
        fft_workers=config.fft_workers,
        dtype=config.dtype,
    )
    state = EvolutionState.from_phi(phi_initial, config.initial_time, dtype=config.dtype)

    if config.save_initial:
        save_snapshot(config.output_path, state, 0)

    try:
        for step_index in range(1, config.max_steps + 1):
            state = step_state(grid, state, config.dt)
            if config.output_every and step_index % config.output_every == 0:
                save_snapshot(config.output_path, state, step_index)
    finally:
        grid.close()

    return state
