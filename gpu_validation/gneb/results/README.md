# GNEB: cross-code comparison and resolution study

Program output of the scripts in the parent directory, 2026-11-24, Float64, CPU.
`reject_study.txt` was run with the code of the commit that adds these files; the others
just before the hidden-maximum check and the rejection counter were added. Neither
changes the iterates with `reject=Inf`: every converged case repeated in
`reject_study.txt` has the same number of steps and the same barrier.

| file | script |
|---|---|
| `compare_ellipse_uniform.txt`, `compare_ellipse_wall.txt`, `compare_skyrmion.txt` | `compare_gneb.jl CASE` |
| `compare_lattice4.txt` | `compare_gneb.jl` before the cases were renamed: it prints "CASE skyrmion", but the start state is a lattice of **four** skyrmions (Q = −4), now case `lattice4` |
| `magnumnp_ellipse_uniform.txt` | `ellipse_magnumnp.py`: string method, 400 steps (magnum.np is not in the global Python since 2026-11-24: needs its own venv) |
| `resolution.txt` | `skyrmion_resolution.jl`, at most 20 000 steps (per stage in MicroMagnetic) |
| `reject_study.txt` | `reject_study.jl`, at most 40 000 steps |

`reject_study.txt` has 39 of the 45 planned runs. The processes were stopped from outside
before these six ran: lattice4 N=31 k=1 with reject 10, 3 and 1.5; lattice4 N=21 k=1 with
reject 3 and 1.5; lattice4 N=11 k=1 with reject 1.5.

## Barriers, 21 images, k = 1 T, same end states and initial path

| case | JuliaMag (J) | steps | MicroMagnetic vp, relative | steps | same saddle image, functional difference |
|---|---|---|---|---|---|
| ellipse, +x → +y → −x | 2.4139554504e-18 | 256 | +1.97e-4 | 2068+426 | +1.97e-4 |
| ellipse, wall sweep | 1.7100329947e-18 | 327 | +1.66e-4 | 2323+1397 | +1.66e-4 |
| one skyrmion → uniform | 7.9422822692e-20 | 950 | −5.5e-13 | 5633+2593 | −1e-14 |
| four skyrmions → uniform | 1.2023840745e-19 | 2571 | −9e-9 | 7271+14700 | −3e-14 |

- The ellipse differences are those of the demagnetizing kernel (JuliaMag: the mumax3
  kernel, accuracy 6; MicroMagnetic: Newell): they appear unchanged when the same image is
  evaluated in both codes.
- magnum.np's string method on the uniform path: +2.26e-4 after 400 steps and still
  decreasing; +1.90e-4 for the same image.
- Saddle order, from the eigenvalues of the energy Hessian (`certify_saddles.jl`, dense,
  `certify_*.txt`; stop force 1e-7 T), of the climbing image:

| case | index | lowest curvatures (T) | end states |
|---|---|---|---|
| ellipse, +x → +y → −x | **2** | −0.01593, −0.00537, +0.01093 | minima (lowest +0.0243) |
| ellipse, wall sweep | 1 | −0.00207, +0.00280, +0.02004 | minima |
| one skyrmion → uniform | 1 | −2.905, +0.1103, +0.1103 | minima; start: two zero modes (translations, 1e-13) |
| four skyrmions → uniform | **4** | −2.653, −2.653, −0.0020, −0.00014, +0.0013 | **start is not a minimum**: −0.00162 twice, two zero modes |

- Of the two unstable modes of the uniform path the first is even under inversion, the second odd
  (parity -1.000000): a symmetric band cannot leave it. The path is not a transition path; the wall path is the
  lower one (29 % below). The four-skyrmion lattice is a symmetric stationary state on which
  the minimizer stopped: "the barrier" found from it is not that of a metastable state, and
  its stop-force value is not converged (1.20238e-19 J, 1e-5 T against 1e-7 T: −2.7e-7).
  Both codes agree on it, so the agreement says nothing about its meaning.
- The force tolerances of the two codes are the same quantity (largest per-cell norm of
  the total force over the free images); rows where a code did not converge are not
  comparable: MicroMagnetic starts climbing even if its first stage has not converged, does
  not move the climbing image, and gets the step limit twice.
- lbfgs_atlas: first stage unconverged after 40 000 steps on both film cases.

## Too few images or a soft spring

| band | JuliaMag, `reject=Inf` | MicroMagnetic vp |
|---|---|---|
| four skyrmions, N=11, k=1 | **converges to barrier 0** (24 602 steps): the images slide along translations of the lattice and the transition is one jump | not converged; wrong |
| four skyrmions, N=21, k=0.3 | not converged (0.12 T) | right barrier (11246+12845, earlier run) |
| one skyrmion, N=11, k=1 | not converged (0.28 T at 20 000; 4.7 T at 40 000 steps) | right barrier; first stage unconverged |

- The energy halfway between images 6 and 7 of the first band is 1.71e-19 J above every
  image. `relax!` now warns in that case (`midpointenergies`).
- The optimizer (Barzilai–Borwein) is less robust here than MicroMagnetic's velocity
  projection, and takes 8–12 times fewer steps where both converge (N = 21, 31).

## `reject` (band evaluations to max|F| < 1e-5 T)

| case | Inf | 100 | 10 | 3 | 1.5 |
|---|---|---|---|---|---|
| ellipse uniform | 256 | 273 | 655 | 1213 | 1502 |
| ellipse wall | 327 | 345 | 1095 | 2339 | 2693 |
| skyrmion N=21 k=1 | 950 | 874 | 2264 | 3901 | 4860 |
| skyrmion N=21 k=0.3 | 987 | 1018 | 3029 | 6258 | 7998 |
| four skyrmions N=21 k=1 | 2571 | 3548 | 5029 | not run | not run |
| four skyrmions N=31 k=1 | 1520 | 1548 | not run | not run | not run |
| four skyrmions N=21 k=0.3 | fails | fails | 7104 | 12121 | 15026 |
| skyrmion N=11 k=1 | fails | fails | fails | 16396 | **false barrier (≈0)** |
| four skyrmions N=11 k=1 | false (≈0) | false | false | not converged | not run |

The default stays `Inf`. No finite value is reliable: in the skyrmion N=11 band 10 does not
converge, 3 reaches the right barrier and 1.5 a band that skips the saddle (with the
warning). Choosing 3 would be a fit to two cases, at 4–8 times the cost elsewhere. More
images is the remedy.
