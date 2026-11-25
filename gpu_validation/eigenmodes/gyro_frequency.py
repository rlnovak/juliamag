"""Frequency of the free gyration against amplitude, extrapolated to zero (the eigenfrequency).

Data: results/gyro/gyro_*mT.txt (gyro_amplitude.jl). z = mx + i my rotates; the staircase disc has
fourfold symmetry. The phase of z and |z| are fitted over a window that starts after the
short transients: phase(t) = phi0 + 2 pi (f0 t + kappa integral |z|^2 dt), ln|z| = c - Gamma t.
"""
import sys
from pathlib import Path
import numpy as np

D = Path(__file__).parent / "results" / "gyro"
F_EIG, G_EIG = 968.2342, 0.079401            # disc_gyrotropic_cpu.txt (MHz, 1/ns)
T0 = float(sys.argv[1]) if len(sys.argv) > 1 else 2.0


def load(b):
    d = np.loadtxt(D / f"gyro_{b}mT.txt")
    return d[:, 0] * 1e9, d[:, 1] + 1j * d[:, 2]


res = []
print(f"window {T0} ns to the end")
for b in (10, 5, 2, 1):
    t, z = load(b)
    s = t >= T0
    t, z = t[s], z[s]
    ph = np.unwrap(np.angle(z))
    a2 = np.abs(z) ** 2
    f_lin = np.polyfit(t, ph, 1)[0] / (2 * np.pi) * 1e3
    g = -np.polyfit(t, np.log(np.abs(z)), 1)[0]
    # phase rotation with elliptic ripple removed by the fit over many periods
    I = np.concatenate([[0], np.cumsum(0.5 * (a2[1:] + a2[:-1]) * np.diff(t))])
    res.append((b, f_lin, g, a2.mean(), np.abs(z[0]), np.abs(z[-1])))
    A = np.c_[np.ones_like(t), t, I]
    c, *_ = np.linalg.lstsq(A, ph, rcond=None)
    print(f"B = {b:2d} mT: |z| {abs(z[0]):.5f} -> {abs(z[-1]):.5f}, f(mean) = {f_lin:.3f} MHz, Gamma = {g:.5f} 1/ns, "
          f"f0 (this run, with kappa) = {c[1] / (2 * np.pi) * 1e3:.3f} MHz")
r = np.array(res)
p = np.polyfit(r[:, 3], r[:, 1], 1)
print("f = f0 + k <|z|^2>: f0 = %.3f MHz (from 4 runs), residuals %s MHz" % (p[1], np.round(r[:, 1] - np.polyval(p, r[:, 3]), 3)))
q = np.polyfit(r[:, 3], r[:, 2], 1)
print("Gamma0 = %.5f 1/ns" % q[1])
print("eigenmode: f = %.3f MHz (f0/eig - 1 = %+.1e), Gamma = %.5f (rel %+.1e)" % (F_EIG, p[1] / F_EIG - 1, G_EIG, q[1] / G_EIG - 1))
