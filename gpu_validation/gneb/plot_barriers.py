"""Figure of the paper: energy barriers (gpu_validation/gneb/results -> paper/figures/gneb_barriers.png)."""
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HERE = Path(__file__).parent
R = HERE / "results"
OUT = HERE.parents[1] / "paper" / "figures" / "gneb_barriers.png"
RED = dict(color="red", lw=2, alpha=0.5)
BLUE = dict(color="blue", lw=1)

fig, ax = plt.subplots(1, 3, figsize=(10.5, 3.2), constrained_layout=True)

sw = np.loadtxt(R / "stoner_wohlfarth.txt")
h = np.linspace(0, 1, 200)
KV = sw[0, 2]
ax[0].plot(h, (1 - h) ** 2, label=r"$(1-h)^2$", **RED)
ax[0].plot(sw[:, 0], sw[:, 1] / KV, marker="o", ms=4, mfc="none", ls="none", color="blue", label="JuliaMag")
ax[0].set(xlabel=r"$h = M_s B/2K$", ylabel=r"$\Delta E / KV$", title="(a) Stoner–Wohlfarth", xlim=(0, 1), ylim=(0, 1.05))
ax[0].legend(frameon=False)


def path(case, code):
    d = np.loadtxt(R / f"path_{case}_{code}.txt")
    return d[:, 1] / d[-1, 1], d[:, 2] - d[0, 2]


for case, mk in (("ellipse_uniform", "o"), ("ellipse_wall", "s")):
    s, e = path(case, "micromagnetic")
    ax[1].plot(s, e * 1e18, label="MicroMagnetic.jl" if mk == "o" else None, **RED)
    s, e = path(case, "juliamag")
    ax[1].plot(s, e * 1e18, marker=mk, ms=3, mfc="none", label="JuliaMag" if mk == "o" else None, **BLUE)
ax[1].annotate("uniform path (index 2)", (0.5, 2.5), ha="center", fontsize=8)
ax[1].annotate("wall path (index 1)", (0.5, 1.22), ha="center", fontsize=8)
ax[1].set(xlabel="path coordinate", ylabel=r"$E - E_0$ ($10^{-18}$ J)", title="(b) ellipse", xlim=(0, 1), ylim=(0, 2.75))
ax[1].legend(frameon=False, loc="lower center", fontsize=8)

s, e = path("skyrmion", "micromagnetic")
ax[2].plot(s, e * 1e20, label="MicroMagnetic.jl", **RED)
s, e = path("skyrmion", "juliamag")
ax[2].plot(s, e * 1e20, marker="o", ms=3, mfc="none", label="JuliaMag", **BLUE)
ax[2].set(xlabel="path coordinate", ylabel=r"$E - E_0$ ($10^{-20}$ J)", title="(c) skyrmion collapse (index 1)", xlim=(0, 1))
ax[2].legend(frameon=False, fontsize=8)
fig.savefig(OUT, dpi=200)
print(OUT)
