## Files

- `fields.py`: initial Gaussian and non-Gaussian random fields.
- `fft_cpu.py`: CPU-parallel FFT pseudospectral evolution for one parameter point.
- `fd_cpu.py`: Numba-parallel second- and fourth-order finite-difference evolution.
- `torch_gpu.py`: optional dense-matrix and FFT PyTorch backends.
- `spectrum.py`: vectorized TT projection and GW spectrum calculation.

## Quick Run

```bash
python run_leapfrog_cpu_parallel.py
```

To compare worker counts on the current machine:

```bash
python benchmark_leapfrog_workers.py
```

To compare the FFT and finite-difference evolution backends:

```bash
python benchmark_leapfrog_backends.py
```

The important knob for single-run CPU parallelism is `workers` in
`EvolutionConfig`. It controls a thread pool for independent FFT tasks inside
one parameter point. Use `workers=-1` for all available CPU workers, or set a
fixed number such as `workers=8`.

`fft_workers` is passed to each individual SciPy FFT and defaults to `1`. Keep
it at `1` on machines where changing SciPy's FFT workers does not affect CPU
usage; increase it only if a direct FFT benchmark shows that it helps.

The finite-difference backend is available as `run_evolution_fd`. It supports
periodic second- and fourth-order central differences through
`EvolutionConfig(fd_order=2)` or `EvolutionConfig(fd_order=4)`. It should make
CPU worker counts more visible, but it is not spectrally accurate, so compare
convergence against the FFT backend before using production data.

For exploratory runs, `EvolutionConfig(dtype=np.float32)` often speeds up the
finite-difference backend because the stencil kernels move less memory. Use
`np.float64` for reference runs and compare spectra before trusting single
precision.

Initial conditions are customizable:

```python
field_g, phi = gaussian_random_fields(
    n,
    power_spectrum=my_power_spectrum,
    non_gaussian_transform=my_non_gaussian_transform,
)
```

The optional dense GPU backend is exposed as `run_evolution_torch(phi, config,
device="cuda")`; the FFT PyTorch backend is exposed as
`run_evolution_torch_fft(phi, config, device="cuda")`. PyTorch is imported only
when one of these functions is called.  They clear PyTorch's CUDA cache at the
end by default.  You can also call `clear_torch_cache("cuda")` manually after
custom GPU work.

Both PyTorch backends accept `dtype=np.float32` or `dtype=np.float64`. If
omitted, they use `EvolutionConfig(dtype=...)`. Use `np.float32` for speed and
lower GPU memory use, or `np.float64` for reference runs. The torch benchmark
accepts the same choice:

```bash
python benchmark_torch_backends.py --dtype float32
python benchmark_torch_backends.py --dtype float64 --skip-dense
```

The physical box convention is unchanged from the notebook: the box length is
`space_length * pi`, and the default time step is `space_length / n / 5`.
