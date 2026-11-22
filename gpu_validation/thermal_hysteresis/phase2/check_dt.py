"""Phase 2, 300 K time-step check: virgin branch (0 -> 150 mT, dwell 1 ns, seed 1)
with Heun dt = 1e-13 s and 5e-14 s, next to the 50 K run with the same dwell
(phase1/conv_dwell1ns.txt) and the T = 0 minimizer branch.

Pass criteria:
- On the saturated branch (B >= 70 mT) the two steps agree with no systematic
  sign. A discretization bias of the stochastic Heun scheme would show there as
  a shift of the thermal reduction of <mx>.
- On the vortex branch the differences stay within the thermal fluctuation.

Usage: python check_dt.py
"""
import os
import sys
import numpy as np

sys.stdout.reconfigure(encoding="utf-8")
HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.join(HERE, "..", "..", "..")
a = np.loadtxt(os.path.join(HERE, "dtcheck_T300_dt1e-13.txt"), comments="#")
b = np.loadtxt(os.path.join(HERE, "dtcheck_T300_dt5e-14.txt"), comments="#")
c = np.loadtxt(os.path.join(HERE, "..", "phase1", "conv_dwell1ns.txt"), comments="#")
r = np.loadtxt(os.path.join(ROOT, "examples", "disc_hysteresis.txt"), comments="#")[:31]
n = min(len(a), len(b))
a, b, c, r = a[:n], b[:n], c[:n], r[:n]
print("B(mT)   T=0    50 K     300 K dt    300 K dt/2   diff    |<my>| dt, dt/2")
for i in range(n):
    print(f"{a[i,0]*1e3:4.0f}  {r[i,1]:+.4f} {c[i,4]:+.4f}  {a[i,4]:+.4f}     {b[i,4]:+.4f}    "
          f"{a[i,4]-b[i,4]:+.4f}   {abs(a[i,5]):.3f} {abs(b[i,5]):.3f}")
hi = a[:, 0] >= 0.0695
lo = a[:, 0] <= 0.0505
d = a[hi, 4] - b[hi, 4]
print(f"rows {n}; saturated (>= 70 mT): mean diff {d.mean():+.1e}, mean |diff| {np.abs(d).mean():.1e},"
      f" positive {np.sum(d > 0)}/{d.size}")
print(f"vortex branch (<= 50 mT): max |diff| {np.abs(a[lo, 4] - b[lo, 4]).max():.4f},"
      f" typical |<my>| {np.abs(a[lo, 5]).mean():.3f}")
red300 = (r[hi, 1] - 0.5 * (a[hi, 4] + b[hi, 4])).mean()
red50 = (r[hi, 1] - c[hi, 4]).mean()
print(f"thermal reduction of saturated <mx> vs T = 0: 300 K {red300:.4f}, 50 K {red50:.4f},"
      f" ratio {red300/red50:.2f} (6 if linear in T)")
