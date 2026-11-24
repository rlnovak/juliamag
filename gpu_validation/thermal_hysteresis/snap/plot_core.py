"""Figure: vortex-core tracks of the 50 K re-runs (seed 1: core expelled at -25 mT;
seed 7: core kept to -30 mT). Top: <m_x>(t) with the field steps; middle: core x, y
(only where a core exists, |m_z| > 0.7); bottom: track in the disc plane, coloured
by field step."""
import numpy as np
import matplotlib.pyplot as plt
from analyse_core import load, R, T0, DWELL, FIELDS, PMIN

fig, axs = plt.subplots(3, 2, figsize=(10, 11), gridspec_kw=dict(height_ratios=[1, 1, 2]))
cols = plt.cm.viridis(np.linspace(0, 0.95, len(FIELDS)))
for j, seed in enumerate((1, 7)):
    t, mx, x, y, p = load(seed)
    core = np.abs(p) > PMIN
    tn = (t - T0) * 1e9
    ax = axs[0, j]
    ax.plot(tn, mx, "b", lw=1)
    for k, B in enumerate(FIELDS):
        ax.axvspan(2 * k, 2 * k + 2, color=cols[k], alpha=0.12, lw=0)
        ax.text(2 * k + 1, 1.02, f"{B}", ha="center", va="bottom", fontsize=8, transform=ax.get_xaxis_transform())
    ax.set_ylabel(r"$\langle m_x\rangle$")
    ax.set_title(f"T = 50 K, seed {seed}  (B$_x$ in mT above)", pad=16)
    ax.set_ylim(-1.05, 1.05)
    ax = axs[1, j]
    xs = np.where(core, x, np.nan) * 1e9
    ys = np.where(core, y, np.nan) * 1e9
    ax.plot(tn, xs, "b", lw=1, label="core x")
    ax.plot(tn, ys, color="tab:orange", lw=1, label="core y")
    ax.axhline(0, color="k", lw=0.5)
    ax.set_xlabel("t − t(step 62) (ns)")
    ax.set_ylabel("core position (nm)")
    ax.set_ylim(-260, 260)
    ax.legend(fontsize=8, loc="lower left")
    ax = axs[2, j]
    th = np.linspace(0, 2 * np.pi, 400)
    ax.plot(R * 1e9 * np.cos(th), R * 1e9 * np.sin(th), "k", lw=1)
    for k in range(len(FIELDS)):
        w = core & (t >= T0 + k * DWELL) & (t < T0 + (k + 1) * DWELL)
        if w.any():
            ax.plot(x[w] * 1e9, y[w] * 1e9, ".-", color=cols[k], ms=2, lw=0.5, label=f"{FIELDS[k]} mT")
            ax.plot(x[w][-1] * 1e9, y[w][-1] * 1e9, "o", color=cols[k], ms=6, mec="k")
    ax.set_aspect("equal")
    ax.set_xlim(-270, 270); ax.set_ylim(-270, 270)
    ax.set_xlabel("x (nm)"); ax.set_ylabel("y (nm)")
    ax.legend(fontsize=8, loc="upper right", title="core track (o: end of step)", title_fontsize=8)
fig.tight_layout()
fig.savefig("core_tracks.png", dpi=150)
print("wrote core_tracks.png")
