"""Optional PyTorch backend matching the original dense-matrix GPU method."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from .fft_cpu import Array, EvolutionConfig, EvolutionState, Tensor6


def _torch():
    try:
        import torch
    except ImportError as exc:
        raise ImportError(
            "The PyTorch backend requires torch. Install the optional GPU "
            "dependency, for example `pip install SimuSIGW[gpu]`, or "
            "install the PyTorch build matching your CUDA version."
        ) from exc
    return torch


def _torch_dtype(torch, dtype: np.dtype):
    dtype = np.dtype(dtype)
    if dtype == np.dtype(np.float32):
        return torch.float32
    if dtype == np.dtype(np.float64):
        return torch.float64
    raise ValueError("PyTorch backend supports only float32 and float64")


def available_torch_device(preferred: str | None = None) -> str:
    """Return ``preferred`` or choose CUDA when PyTorch can see it."""

    torch = _torch()
    if preferred is not None:
        if preferred.startswith("cuda") and not torch.cuda.is_available():
            raise RuntimeError(
                "device='cuda' was requested, but PyTorch cannot see a CUDA GPU."
            )
        return preferred
    return "cuda" if torch.cuda.is_available() else "cpu"


def clear_torch_cache(device: str | None = None) -> None:
    """Release PyTorch's cached GPU memory for this backend."""

    torch = _torch()
    if not torch.cuda.is_available():
        return
    if device is None:
        torch.cuda.empty_cache()
        return
    torch_device = torch.device(device)
    if torch_device.type == "cuda":
        with torch.cuda.device(torch_device):
            torch.cuda.empty_cache()


def fourier_matrices_torch(
    n: int,
    space_length: float = 2.0,
    device: str | None = None,
    dtype: np.dtype = np.float64,
):
    """Dense Fourier differentiation matrices from the original notebook."""

    torch = _torch()
    device = available_torch_device(device)
    torch_dtype = _torch_dtype(torch, dtype)

    n = int(n)
    d1 = torch.zeros((n, n), device=device, dtype=torch_dtype)
    d2 = torch.zeros((n, n), device=device, dtype=torch_dtype)

    idx = torch.arange(n, device=device)
    diff = idx[:, None] - idx[None, :]
    lower = diff > 0
    diff_lower = diff[lower].to(torch_dtype)
    sign = torch.where(diff[lower] % 2 == 0, 1.0, -1.0).to(torch_dtype)
    angle = diff_lower * (torch.tensor(np.pi, device=device, dtype=torch_dtype) / n)

    d1[lower] = sign / (2.0 * torch.tan(angle))
    d2[lower] = -sign / (2.0 * torch.sin(angle) ** 2)

    d1 = d1 - d1.T
    d2 = d2 + d2.T - ((n**2 + 2.0) / 12.0) * torch.eye(
        n, device=device, dtype=torch_dtype
    )
    return (2.0 / space_length) * d1, (2.0 / space_length) ** 2 * d2


def _laplacian(d2, values):
    torch = _torch()
    return (
        torch.einsum("Xx,xyz->Xyz", d2, values)
        + torch.einsum("Yy,xyz->xYz", d2, values)
        + torch.einsum("Zz,xyz->xyZ", d2, values)
    )


def _dx(d1, values):
    torch = _torch()
    return torch.einsum("Xx,xyz->Xyz", d1, values)


def _dy(d1, values):
    torch = _torch()
    return torch.einsum("Yy,xyz->xYz", d1, values)


def _dz(d1, values):
    torch = _torch()
    return torch.einsum("Zz,xyz->xyZ", d1, values)


def _dxx(d2, values):
    torch = _torch()
    return torch.einsum("Xx,xyz->Xyz", d2, values)


def _dyy(d2, values):
    torch = _torch()
    return torch.einsum("Yy,xyz->xYz", d2, values)


def _dzz(d2, values):
    torch = _torch()
    return torch.einsum("Zz,xyz->xyZ", d2, values)


def _scalar_rhs(d2, phi, pi, t: float):
    return pi, (1.0 / 3.0) * _laplacian(d2, phi) - (4.0 / t) * pi


def _rk4_scalar_step(d2, phi, pi, t: float, dt: float):
    dphi1, dpi1 = _scalar_rhs(d2, phi, pi, t)
    dphi2, dpi2 = _scalar_rhs(d2, phi + 0.5 * dt * dphi1, pi + 0.5 * dt * dpi1, t + 0.5 * dt)
    dphi3, dpi3 = _scalar_rhs(d2, phi + 0.5 * dt * dphi2, pi + 0.5 * dt * dpi2, t + 0.5 * dt)
    dphi4, dpi4 = _scalar_rhs(d2, phi + dt * dphi3, pi + dt * dpi3, t + dt)
    return (
        phi + (dt / 6.0) * (dphi1 + 2.0 * dphi2 + 2.0 * dphi3 + dphi4),
        pi + (dt / 6.0) * (dpi1 + 2.0 * dpi2 + 2.0 * dpi3 + dpi4),
    )


def _tensor_acceleration(d1, d2, phi, pi, h_components: Tensor6, t: float):
    psi = t * pi + phi
    phix, phiy, phiz = _dx(d1, phi), _dy(d1, phi), _dz(d1, phi)
    psix, psiy, psiz = _dx(d1, psi), _dy(d1, psi), _dz(d1, psi)

    phixx = _dxx(d2, phi)
    phiyy = _dyy(d2, phi)
    phizz = _dzz(d2, phi)
    phixy = _dy(d1, phix)
    phixz = _dz(d1, phix)
    phiyz = _dz(d1, phiy)

    sources = (
        4.0 * phi * phixx + 2.0 * phix**2 - psix**2,
        4.0 * phi * phixy + 2.0 * phix * phiy - psix * psiy,
        4.0 * phi * phixz + 2.0 * phix * phiz - psix * psiz,
        4.0 * phi * phiyy + 2.0 * phiy**2 - psiy**2,
        4.0 * phi * phiyz + 2.0 * phiy * phiz - psiy * psiz,
        4.0 * phi * phizz + 2.0 * phiz**2 - psiz**2,
    )
    return tuple(
        _laplacian(d2, h) - 4.0 * t * source
        for h, source in zip(h_components, sources, strict=True)
    )


def _save_torch_snapshot(path: str | Path, state: dict, step_index: int) -> None:
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    t = state["t"]
    h_components = state["h"]
    v_components = state["v"]
    h_real = tuple(h / t for h in h_components)
    a_real = tuple((v - h) / t for v, h in zip(v_components, h_real, strict=True))

    names_and_values = (
        ("phi", state["phi"]),
        ("pi", state["pi"]),
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
        np.save(path / f"{name}_{step_index}.npy", values.detach().cpu().numpy())


def _state_to_numpy(state: dict) -> EvolutionState:
    values = {
        key: value.detach().cpu().numpy()
        for key, value in (
            ("phi", state["phi"]),
            ("pi", state["pi"]),
            ("h11", state["h"][0]),
            ("h12", state["h"][1]),
            ("h13", state["h"][2]),
            ("h22", state["h"][3]),
            ("h23", state["h"][4]),
            ("h33", state["h"][5]),
            ("v11", state["v"][0]),
            ("v12", state["v"][1]),
            ("v13", state["v"][2]),
            ("v22", state["v"][3]),
            ("v23", state["v"][4]),
            ("v33", state["v"][5]),
        )
    }
    return EvolutionState(t=state["t"], **values)


def _delete_torch_state(state: dict | None) -> None:
    if not state:
        return
    state.clear()


def _torch_wave_numbers(torch, n: int, space_length: float, device: str, dtype):
    k1 = (2.0 / space_length) * torch.fft.fftfreq(
        int(n), d=1.0 / int(n), device=device
    ).to(dtype)
    kx, ky, kz = torch.meshgrid(k1, k1, k1, indexing="ij")
    k2 = kx * kx + ky * ky + kz * kz
    return kx, ky, kz, k2


def _fftn(torch, values):
    return torch.fft.fftn(values, dim=(-3, -2, -1))


def _ifftn_real(torch, values_hat):
    return torch.fft.ifftn(values_hat, dim=(-3, -2, -1)).real


def _laplacian_fft(torch, k2, values):
    return _ifftn_real(torch, -k2 * _fftn(torch, values))


def _laplacian_from_hat(torch, k2, values_hat):
    return _ifftn_real(torch, -k2 * values_hat)


def _gradient_from_hat(torch, kx, ky, kz, values_hat):
    return (
        _ifftn_real(torch, 1j * kx * values_hat),
        _ifftn_real(torch, 1j * ky * values_hat),
        _ifftn_real(torch, 1j * kz * values_hat),
    )


def _hessian_from_hat(torch, kx, ky, kz, values_hat):
    return (
        _ifftn_real(torch, -(kx * kx) * values_hat),
        _ifftn_real(torch, -(kx * ky) * values_hat),
        _ifftn_real(torch, -(kx * kz) * values_hat),
        _ifftn_real(torch, -(ky * ky) * values_hat),
        _ifftn_real(torch, -(ky * kz) * values_hat),
        _ifftn_real(torch, -(kz * kz) * values_hat),
    )


def _scalar_rhs_fft(torch, k2, phi, pi, t: float):
    return pi, (1.0 / 3.0) * _laplacian_fft(torch, k2, phi) - (4.0 / t) * pi


def _rk4_scalar_step_fft(torch, k2, phi, pi, t: float, dt: float):
    phi_hat = _fftn(torch, phi)
    dphi1 = pi
    dpi1 = (1.0 / 3.0) * _laplacian_from_hat(torch, k2, phi_hat) - (4.0 / t) * pi
    dphi2, dpi2 = _scalar_rhs_fft(
        torch, k2, phi + 0.5 * dt * dphi1, pi + 0.5 * dt * dpi1, t + 0.5 * dt
    )
    dphi3, dpi3 = _scalar_rhs_fft(
        torch, k2, phi + 0.5 * dt * dphi2, pi + 0.5 * dt * dpi2, t + 0.5 * dt
    )
    dphi4, dpi4 = _scalar_rhs_fft(torch, k2, phi + dt * dphi3, pi + dt * dpi3, t + dt)
    return (
        phi + (dt / 6.0) * (dphi1 + 2.0 * dphi2 + 2.0 * dphi3 + dphi4),
        pi + (dt / 6.0) * (dpi1 + 2.0 * dpi2 + 2.0 * dpi3 + dpi4),
        phi_hat,
    )


def _tensor_acceleration_fft(
    torch,
    kx,
    ky,
    kz,
    k2,
    phi,
    pi,
    h_components: Tensor6,
    t: float,
    phi_hat=None,
    psi_workspace=None,
):
    if phi_hat is None:
        phi_hat = _fftn(torch, phi)
    if psi_workspace is None:
        psi = t * pi + phi
    else:
        torch.add(phi, pi, alpha=t, out=psi_workspace)
        psi = psi_workspace
    psi_hat = _fftn(torch, psi)

    phi_x, phi_y, phi_z = _gradient_from_hat(torch, kx, ky, kz, phi_hat)
    psi_x, psi_y, psi_z = _gradient_from_hat(torch, kx, ky, kz, psi_hat)
    phi_xx, phi_xy, phi_xz, phi_yy, phi_yz, phi_zz = _hessian_from_hat(
        torch, kx, ky, kz, phi_hat
    )

    sources = (
        4.0 * phi * phi_xx + 2.0 * phi_x**2 - psi_x**2,
        4.0 * phi * phi_xy + 2.0 * phi_x * phi_y - psi_x * psi_y,
        4.0 * phi * phi_xz + 2.0 * phi_x * phi_z - psi_x * psi_z,
        4.0 * phi * phi_yy + 2.0 * phi_y**2 - psi_y**2,
        4.0 * phi * phi_yz + 2.0 * phi_y * phi_z - psi_y * psi_z,
        4.0 * phi * phi_zz + 2.0 * phi_z**2 - psi_z**2,
    )
    return tuple(
        _laplacian_fft(torch, k2, h) - 4.0 * t * source
        for h, source in zip(h_components, sources, strict=True)
    )


def run_evolution_torch_fft(
    phi_initial: Array,
    config: EvolutionConfig = EvolutionConfig(),
    device: str | None = None,
    dtype: np.dtype | None = None,
    clear_cache: bool = True,
) -> EvolutionState:
    """Run a PyTorch FFT pseudospectral backend.

    This backend uses ``torch.fft`` derivatives instead of dense Fourier
    differentiation matrices, so it is the preferred PyTorch backend for large
    grids.  Use ``device="cuda"`` for GPU execution.
    """

    torch = _torch()
    device = available_torch_device(device)
    compute_dtype = config.dtype if dtype is None else dtype
    torch_dtype = _torch_dtype(torch, compute_dtype)

    kx = ky = kz = k2 = phi = zeros = psi_workspace = None
    state = None
    result = None
    try:
        kx, ky, kz, k2 = _torch_wave_numbers(
            torch, config.n, config.space_length, device, torch_dtype
        )
        phi = torch.as_tensor(phi_initial, device=device, dtype=torch_dtype).clone()
        zeros = torch.zeros_like(phi)
        psi_workspace = torch.empty_like(phi)
        state = {
            "t": config.initial_time,
            "phi": phi,
            "pi": zeros.clone(),
            "h": tuple(zeros.clone() for _ in range(6)),
            "v": tuple(zeros.clone() for _ in range(6)),
        }

        if config.save_initial:
            _save_torch_snapshot(config.output_path, state, 0)

        with torch.no_grad():
            for step_index in range(1, config.max_steps + 1):
                t = state["t"]
                phi_next, pi_next, phi_hat = _rk4_scalar_step_fft(
                    torch, k2, state["phi"], state["pi"], t, config.dt
                )
                acc = _tensor_acceleration_fft(
                    torch,
                    kx,
                    ky,
                    kz,
                    k2,
                    state["phi"],
                    state["pi"],
                    state["h"],
                    t,
                    phi_hat,
                    psi_workspace,
                )
                v_next = tuple(
                    v + config.dt * a for v, a in zip(state["v"], acc, strict=True)
                )
                h_next = tuple(
                    h + config.dt * v for h, v in zip(state["h"], v_next, strict=True)
                )

                state = {
                    "t": t + config.dt,
                    "phi": phi_next,
                    "pi": pi_next,
                    "h": h_next,
                    "v": v_next,
                }
                if config.output_every and step_index % config.output_every == 0:
                    _save_torch_snapshot(config.output_path, state, step_index)

        result = _state_to_numpy(state)
        return result
    finally:
        _delete_torch_state(state)
        del kx, ky, kz, k2, phi, zeros, psi_workspace
        torch_device = torch.device(device)
        if torch_device.type == "cuda" and torch.cuda.is_available():
            torch.cuda.synchronize(torch_device)
        if clear_cache:
            clear_torch_cache(device)


def run_evolution_torch(
    phi_initial: Array,
    config: EvolutionConfig = EvolutionConfig(),
    device: str | None = None,
    dtype: np.dtype | None = None,
    clear_cache: bool = True,
) -> EvolutionState:
    """Run the original dense Fourier-matrix method with PyTorch.

    Use ``device="cuda"`` for GPU execution.  This backend is included for
    compatibility with the original notebook; for CPU-only machines the FD
    backend is usually faster.
    """

    torch = _torch()
    device = available_torch_device(device)
    compute_dtype = config.dtype if dtype is None else dtype
    torch_dtype = _torch_dtype(torch, compute_dtype)

    d1 = d2 = phi = zeros = None
    state = None
    result = None
    try:
        d1, d2 = fourier_matrices_torch(
            config.n, config.space_length, device=device, dtype=compute_dtype
        )
        phi = torch.as_tensor(phi_initial, device=device, dtype=torch_dtype).clone()
        zeros = torch.zeros_like(phi)
        state = {
            "t": config.initial_time,
            "phi": phi,
            "pi": zeros.clone(),
            "h": tuple(zeros.clone() for _ in range(6)),
            "v": tuple(zeros.clone() for _ in range(6)),
        }

        if config.save_initial:
            _save_torch_snapshot(config.output_path, state, 0)

        with torch.no_grad():
            for step_index in range(1, config.max_steps + 1):
                t = state["t"]
                phi_next, pi_next = _rk4_scalar_step(
                    d2, state["phi"], state["pi"], t, config.dt
                )
                acc = _tensor_acceleration(
                    d1, d2, state["phi"], state["pi"], state["h"], t
                )
                v_next = tuple(
                    v + config.dt * a for v, a in zip(state["v"], acc, strict=True)
                )
                h_next = tuple(
                    h + config.dt * v for h, v in zip(state["h"], v_next, strict=True)
                )

                state = {
                    "t": t + config.dt,
                    "phi": phi_next,
                    "pi": pi_next,
                    "h": h_next,
                    "v": v_next,
                }
                if config.output_every and step_index % config.output_every == 0:
                    _save_torch_snapshot(config.output_path, state, step_index)

        result = _state_to_numpy(state)
        return result
    finally:
        _delete_torch_state(state)
        del d1, d2, phi, zeros
        torch_device = torch.device(device)
        if torch_device.type == "cuda" and torch.cuda.is_available():
            torch.cuda.synchronize(torch_device)
        if clear_cache:
            clear_torch_cache(device)
