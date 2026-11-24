"""Paper figure: one loop of each vortex class at 50 K, re-run with snapshots.
Colour = in-plane angle of m at the end of the -20 mT step (t0 + 9.9 ns is not
stored; the snapshot at t0 + 9.0 ns, 1 ns into the -20 mT step, is used); white
contours m_z = +-0.5 mark the cores; black/grey dots: core track over the whole
-15 ... -30 mT sequence (only where a core exists), end of each step marked."""
import numpy as np
import matplotlib.pyplot as plt
from analyse_core import load, R, T0, DWELL, FIELDS, PMIN
from plot_snapshots import readovf

fig, axs = plt.subplots(1, 2, figsize=(7.2, 3.7))
X = np.linspace(-247.5, 247.5, 100)
for ax, seed, title in zip(axs, (1, 7), ("(a) seed 1: two vortices, expelled at $-25$ mT",
                                       "(b) seed 7: one vortex, kept")):
    t, m = readovf(f"ovf/seed{seed}_m000090.ovf")
    mm = m.mean(axis=0)
    mx, my, mz = mm[..., 0], mm[..., 1], mm[..., 2]
    inside = (mx**2 + my**2 + mz**2) > 0.25
    ax.imshow(np.where(inside, np.arctan2(my, mx), np.nan), origin="lower", cmap="hsv",
              vmin=-np.pi, vmax=np.pi, extent=(-250, 250, -250, 250), alpha=0.75)
    s = 9
    Xg, Yg = np.meshgrid(X, X)
    ax.quiver(Xg[::s, ::s], Yg[::s, ::s], np.where(inside, mx, 0)[::s, ::s],
              np.where(inside, my, 0)[::s, ::s], scale=20, width=0.005, color="0.25")
    ax.contour(X, X, mz, levels=[-0.5, 0.5], colors="w", linewidths=1.5)
    tt, _, x, y, p = load(seed)
    core = np.abs(p) > PMIN
    ax.plot(x[core] * 1e9, y[core] * 1e9, ".", color="k", ms=1.2)
    for k in range(3, len(FIELDS)):
        w = core & (tt >= T0 + k * DWELL) & (tt < T0 + (k + 1) * DWELL)
        if w.any():
            ax.plot(x[w][-1] * 1e9, y[w][-1] * 1e9, "o", mfc="w", mec="k", ms=5)
            ax.annotate(f"{FIELDS[k]}", (x[w][-1] * 1e9, y[w][-1] * 1e9), xytext=(5, 3),
                        textcoords="offset points", fontsize=7)
    th = np.linspace(0, 2 * np.pi, 300)
    ax.plot(250 * np.cos(th), 250 * np.sin(th), "k", lw=0.8)
    ax.set_xlim(-262, 262); ax.set_ylim(-262, 262); ax.set_aspect("equal")
    ax.set_xlabel("x (nm)")
    ax.set_title(title, fontsize=9)
axs[0].set_ylabel("y (nm)")
fig.tight_layout()
out = "../../../paper/figures/vortex_snapshots_thermal.png"
fig.savefig(out, dpi=200)
print("wrote", out)
