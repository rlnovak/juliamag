"""Phase 1 of the finite-temperature vortex hysteresis: dwell and time-step
convergence on the virgin branch (0 -> 150 mT), T = 50 K, seed 1, against the T = 0
minimizer branch (examples/disc_hysteresis.txt).

Usage: python plot_phase1.py   (writes phase1_convergence.png next to this file)
"""
import os
import sys
import numpy as np
import matplotlib.pyplot as plt

sys.stdout.reconfigure(encoding="utf-8")
HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.join(HERE, "..", "..", "..")

ref = np.loadtxt(os.path.join(ROOT, "examples", "disc_hysteresis.txt"), comments="#")
B0, mx0 = ref[:31, 4], ref[:31, 1]          # columns: t mx my mz Bx By Bz

runs = [("dwell 0.5 ns", "conv_dwell0.5ns.txt", dict(color="tab:cyan", ls=":")),
        ("dwell 1 ns", "conv_dwell1ns.txt", dict(color="tab:blue", ls="--")),
        ("dwell 1 ns, dt/2", "conv_dwell1ns_dt5e-14.txt", dict(color="navy", ls="-.")),
        ("dwell 2 ns", "conv_dwell2ns.txt", dict(color="blue", ls="-"))]
A = {lab: np.loadtxt(os.path.join(HERE, f), comments="#") for lab, f, _ in runs}

print(" B(mT)  T=0     " + "  ".join(f"{lab:>17s}" for lab, _, _ in runs))
for i in range(31):
    print(f"{B0[i]*1e3:5.0f}  {mx0[i]:+.4f}  " + "  ".join(f"{A[lab][i, 4]:+17.4f}" for lab, _, _ in runs))
for lab, _, _ in runs:
    d = A[lab][:, 4] - mx0
    lo = A[lab][:, 0] <= 0.0505
    print(f"{lab:17s} max|<mx> - T=0| for B <= 50 mT: {np.abs(d[lo]).max():.4f};"
          f"  annihilation (first <mx> > 0.95): {A[lab][np.argmax(A[lab][:, 4] > 0.95), 0]*1e3:.0f} mT")

fig, ax = plt.subplots(1, 2, figsize=(10, 4))
for a, (xl, yl) in zip(ax, [((0, 150), (-0.02, 1.01)), ((0, 70), (-0.02, 1.01))]):
    a.plot(B0 * 1e3, mx0, color="red", lw=2, alpha=0.5, marker="o", ms=3, label="T = 0 (minimizer)")
    for lab, _, sty in runs:
        a.plot(A[lab][:, 0] * 1e3, A[lab][:, 4], lw=1, marker=".", ms=3, label=f"50 K, {lab}", **sty)
    a.set_xlim(*xl); a.set_ylim(*yl)
    a.set_xlabel(r"$\mu_0 H_x$ (mT)"); a.set_ylabel(r"$\langle m_x\rangle$ (last quarter of the dwell)")
    a.grid(alpha=0.3)
ax[0].set_title("(a) virgin branch, 500 nm Py disc, Heun dt = 1e-13 s", fontsize=9)
ax[1].set_title("(b) zoom: reversible branch and annihilation", fontsize=9)
ax[1].legend(fontsize=7)
fig.tight_layout()
out = os.path.join(HERE, "phase1_convergence.png")
fig.savefig(out, dpi=200)
print("wrote", out)
