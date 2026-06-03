# SimuSIGW

Small numerical package for evolving scalar-induced gravitational-wave source
fields on periodic 3-D grids. Version 0.1.0 can only treat adiabatic initial conditions and a radiation-dominated era. The upcoming new version will have the ability to treat isocurvature perturbation and varying equation of state. If you have used or modified the code, please cite arXiv:2508.10812.

## Backends

- `run_evolution`: CPU FFT pseudospectral backend.
- `run_evolution_fd`: CPU Numba finite-difference backend, with
  `fd_order=2` or `fd_order=4`.
- `run_evolution_torch`: optional PyTorch dense Fourier-matrix backend for the
  original GPU-style calculation.
- `run_evolution_torch_fft`: optional PyTorch FFT pseudospectral backend.

## Install

You can directly use pip now

```bash
pip install SimuSIGW
```
Or you can 

```bash
pip install -e .
```

For the optional PyTorch backend, install a PyTorch build that matches your
CUDA setup. If you use the package metadata extra:

```bash
pip install -e ".[gpu]"
```

## Custom Model

Users provide a dimensional power spectrum function `power_spectrum(k)` and,
optionally, a non-Gaussian transform `non_gaussian_transform(zeta_g)`.

```python
import numpy as np

from SimuSIGW import (
    EvolutionConfig,
    gaussian_random_fields,
    run_evolution_fd,
)


def my_power_spectrum(k):
    kstar = 20.0
    width = 0.1
    amplitude = 1e-2
    return (
        amplitude
        * (1 / kstar) ** 3
        / (np.sqrt(2 * np.pi) * width)
        * np.exp(-((k / kstar - 1) ** 2) / (2 * width**2))
        * (2 * np.pi**2)
    )


def my_non_gaussian_transform(zeta_g):
    return zeta_g + 0.5 * (zeta_g**2 - np.mean(zeta_g**2))


config = EvolutionConfig(
    n=128,
    max_steps=1000,
    output_every=1000,
    output_path="log/custom_run",
    workers=-1,
    fd_order=4,
    save_initial=False,
)

field_g, phi_initial = gaussian_random_fields(
    config.n,
    power_spectrum=my_power_spectrum,
    non_gaussian_transform=my_non_gaussian_transform,
    seed=1234,
    workers=config.workers,
    dtype=config.dtype,
)

final_state = run_evolution_fd(phi_initial, config)
```

The default non-Gaussian transform is the logarithmic transform used in the
original notebook. For exploratory finite-difference runs, `dtype=np.float32`
can be faster; use `np.float64` for reference runs.

## GPU Backend

```python
from SimuSIGW import run_evolution_torch

final_state = run_evolution_torch(phi_initial, config, device="cuda")
```

The dense backend follows the original Fourier differentiation-matrix approach.
For larger grids, prefer the FFT backend:

```python
from SimuSIGW import run_evolution_torch_fft

config = EvolutionConfig(...)
final_state = run_evolution_torch_fft(
    phi_initial,
    config,
    device="cuda",
    dtype=np.float32,  # or np.float64
)
```

Both PyTorch backends accept `dtype=np.float32` or `dtype=np.float64`. If
omitted, they use `config.dtype`. Choose `np.float32` for speed and lower GPU
memory use, or `np.float64` for reference runs. They are optional and import
PyTorch only when called.

## Benchmarks

```bash
python benchmark_leapfrog_workers.py
python benchmark_leapfrog_backends.py
python benchmark_torch_backends.py --dtype float32
python benchmark_torch_backends.py --dtype float64 --skip-dense
```
