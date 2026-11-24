"""magnum.np side of the ellipse comparison (demo energy_barrier_ellipse: string method).

    python gpu_validation/gneb/ellipse_magnumnp.py DIR [STRING_STEPS]

1. Energies of the JuliaMag images (DIR/juliamag_images.bin) in magnum.np.
2. The demo's string method from its own start (+x, +y, -x uniform, 21 images),
   with more steps than the demo's 100; the barrier is logged at every 10th step.
   The string method has no climbing image: once converged, max(E) over the images is
   a lower estimate of the saddle energy. The end images are free to relax.

Energies are differences from the first image: outside the ellipse m = 0 with A kept
uniform (as in the demo), which adds a constant to the exchange energy.
"""
import os
import sys
import time

os.environ["TORCHDYNAMO_DISABLE"] = "1"      # no C++ compiler for torch.compile here
import numpy as np
import torch

torch.set_default_dtype(torch.float64)
# xitorch's cspline (used by StringSolver) calls torch.solve, removed from torch; without
# this shim StringSolver silently falls back to linear interpolation at every step.
torch.solve = lambda B, A: (torch.linalg.solve(A, B), None)
from magnumnp import DemagField, ExchangeField, Mesh, State, StringSolver  # noqa: E402

out = sys.argv[1]
nsteps = int(sys.argv[2]) if len(sys.argv) > 2 else 400
n, dx = (100, 40, 1), (5e-9, 5e-9, 4e-9)
mesh = Mesh(n, dx, origin=(-n[0] * dx[0] / 2, -n[1] * dx[1] / 2, 0.0))
state = State(mesh)
state.material = {"Ms": 800e3, "A": 1.3e-11, "alpha": 0.2}
x, y, _ = mesh.SpatialCoordinate()
magnetic = (x / (470e-9 / 2)) ** 2 + (y / (170e-9 / 2)) ** 2 <= 1.0
print("dtype", state.dtype, " magnetic cells", int(magnetic.sum()))
terms = [DemagField(), ExchangeField()]


def energies(images):
    e = []
    for m in images:
        state.m = m
        e.append([float(t.E(state)) for t in terms])
    return np.array(e)                        # (N, 2): demag, exchange


raw = np.fromfile(f"{out}/juliamag_images.bin", dtype=np.float64)
N = raw.size // (3 * n[0] * n[1] * n[2])
# Julia column-major (3, Nx, Ny, Nz, N) -> magnum.np (Nx, Ny, Nz, 3) per image
jm = raw.reshape(N, n[2], n[1], n[0], 3).transpose(0, 3, 2, 1, 4)
assert np.array_equal(np.linalg.norm(jm[0], axis=-1) > 0.5, magnetic.numpy()), "different magnetic cells"
EJ = np.loadtxt(f"{out}/juliamag_path.txt")[:, 2]
EM = energies([torch.tensor(a.copy()) for a in jm]).sum(1)
dJ, dM = EJ - EJ[0], EM - EM[0]
i = int(np.argmax(dJ))
print(f"[1] JuliaMag images: barrier JuliaMag {dJ[i]:.10e} J, same image in magnum.np {dM[i]:.10e} J, "
      f"rel {dM[i] / dJ[i] - 1:+.2e}; max diff over images / dE {np.abs(dM - dJ).max() / dJ[i]:.2e}")

images = []
for v in ([1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [-1.0, 0.0, 0.0]):
    m = state.Constant(v)
    m[~magnetic] = 0.0
    images.append(m)
import magnumnp.solvers.string as _st  # noqa: E402

_calls = {"cspline": 0, "linear": 0}
_interp = _st.Interp1D


def _counted(x, y, method):
    _calls[method] += 1
    return _interp(x, y, method=method)


_st.Interp1D = lambda x, y, method="cspline": _counted(x, y, method)
string = StringSolver(terms, num_images=21)
t0 = time.time()
log = []
for k in range(1, nsteps + 1):
    images = string.step(state, images)
    if k <= 2:
        print(f"[2] step {k}: interpolation calls so far {_calls} (linear = fallbacks)", flush=True)
    if k % 10 == 0 or k == nsteps:
        E = energies(images).sum(1)
        log.append((k, E.max() - E[0], E[-1] - E[0]))
        print(f"[2] step {k:4d}  dE = {E.max() - E[0]:.10e} J  ({(E.max() - E[0]) / dJ[i] - 1:+.2e} rel to JuliaMag)  "
              f"E[end]-E[0] = {E[-1] - E[0]:+.1e}  {time.time() - t0:.0f} s", flush=True)
np.savetxt(f"{out}/magnumnp_string_log.txt", log, header="step dE(J) Eend-E0(J)")
np.savetxt(f"{out}/magnumnp_path.txt", np.c_[np.arange(len(E)), E], header="image E(J)")
print("interpolation calls", _calls)
