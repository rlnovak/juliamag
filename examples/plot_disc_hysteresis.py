"""Vortex hysteresis loop of a 500 nm x 20 nm Permalloy disc (examples/disc_hysteresis.jl).

Reads examples/disc_hysteresis.txt (t, mx, my, mz, B_extx, B_exty, B_extz; <m> is the
average over the disc, mumax3's table m) and writes examples/disc_hysteresis.png and
paper/figures/vortex_hysteresis.png.

Usage: python plot_disc_hysteresis.py [table] [out.png ...]
"""
import sys, os
import numpy as np
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
TAB = sys.argv[1] if len(sys.argv) > 1 else os.path.join(HERE, "disc_hysteresis.txt")
OUTS = sys.argv[2:] or [os.path.join(HERE, "disc_hysteresis.png"),
                        os.path.join(HERE, "..", "paper", "figures", "vortex_hysteresis.png")]

a = np.loadtxt(TAB, comments="#")
mx, Bx = a[:, 1], a[:, 4] * 1e3
print(f"{len(a)} field steps, <mx> range [{mx.min():.4f}, {mx.max():.4f}]")

plt.rcParams.update({"font.size": 9})
fig, ax = plt.subplots(figsize=(4.2, 3.2))
ax.plot(Bx, mx, "o-", color="blue", lw=1, ms=2)
ax.axhline(0, color="0.6", lw=0.5); ax.axvline(0, color="0.6", lw=0.5)
ax.set_xlabel(r"$\mu_0H_x$ (mT)")
ax.set_ylabel(r"$\langle m_x\rangle$")
ax.set_ylim(-1.05, 1.05)
ax.grid(alpha=0.3)
fig.tight_layout()
for o in OUTS:
    fig.savefig(o, dpi=200)
    print("wrote", os.path.abspath(o))
