# JuliaMag

Micromagnetic simulation in pure Julia, on CPU and (later) GPU.

Inspired by [mumax3](https://github.com/mumax/3) (Vansteenkiste et al., *AIP Advances* **4**, 107133 (2014)),
but written from scratch to take advantage of Julia's multiple dispatch.

## Design

The reduced magnetization `m = M / Msat` is an `Array{T,4}` of shape `(3, Nx, Ny, Nz)`.
The vector component is the fastest-varying index, so `m[:, i, j, k]` is contiguous
and the array maps directly onto a `CuArray`.

Every routine dispatches on the array type. A CPU run passes an `Array`, a GPU run
passes a `CuArray`, and the correct method is selected — no separate code path, no
`if gpu` branches. The element type `T` is propagated too, so `Float32` on the GPU
comes for free.

Index convention is `(x, y, z)`, 1-based, everywhere.

## Status

Under construction. Roadmap:

- [x] **1.** Package skeleton, `Mesh`, `Material`, magnetization states
- [x] **2.** Exchange, uniaxial anisotropy, Zeeman fields
- [ ] **3.** Demagnetization kernel (Newell / brute-force integration)
- [ ] **4.** Demagnetization field via FFT convolution
- [ ] **5.** LLG torque + adaptive RK45 solver
- [ ] **6.** µMAG standard problem 4
- [ ] **7.** OVF I/O and output tables

## Usage

```julia
using JuliaMag

# µMAG standard problem 4 geometry: 500 × 125 × 3 nm
mesh = Mesh((160, 40, 1), (3.125e-9, 3.125e-9, 3e-9))

# Permalloy
mat = Material(Msat = 8.0e5, Aex = 1.3e-11, alpha = 0.02)

m = uniform(mesh, (1, 0, 0))
average(m)   # (1.0, 0.0, 0.0)
```

## Tests

```
julia --project=. -e 'using Pkg; Pkg.test()'
```

## Legacy

`legacy/` holds the earlier exploratory code this package replaces. It is kept for
reference only and is not loaded by the package.

## Author

Rafael L. Novak — rlnovak@gmail.com — UFSC/Blumenau, Brazil
