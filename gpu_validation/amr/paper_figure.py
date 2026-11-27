# Figure of the paper for the AMR conduction: the planar-Hall potential of the tilted wall (JuliaMag
# against the FEM) and the vortex between two leads for three AMR ratios.
#
#   python gpu_validation/amr/paper_figure.py TEXDIR VORTEXDIR OUT.png
#
# TEXDIR holds jm_200x40.txt (juliamag_texture.jl) and fem_200x40_q8.txt (fem_reference.py), the
# potential at the cell centres, ny rows of nx values; VORTEXDIR holds table_drho{0.0,0.02,0.04}.txt
# (vortex_amr.jl).
import sys, os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

texdir, vdir, out = sys.argv[1], sys.argv[2], sys.argv[3]
L, W = 400.0, 80.0                                   # nm
phj = np.loadtxt(os.path.join(texdir, "jm_200x40.txt"), comments="#")
phf = np.loadtxt(os.path.join(texdir, "fem_200x40_q8.txt"), comments="#")
ny, nx = phj.shape
x = (np.arange(nx) + 0.5) * L / nx - L / 2
y = (np.arange(ny) + 0.5) * W / ny - W / 2
# the planar-Hall part: the potential minus its mean over each cross-section, in mV
hall = lambda p: (p - p.mean(axis=0, keepdims=True)) * 1e3
hj, hf = hall(phj), hall(phf)

eps = (0.0, 0.02, 0.04)
tab = {e: np.loadtxt(os.path.join(vdir, f"table_drho{e}.txt"), comments="#") for e in eps}
# columns: t mx my mz vortexx vortexy vortexz vortexw R
t = np.linspace(0, min(tab[e][-1, 0] for e in eps), 601)
col = lambda e, k: np.interp(t, tab[e][:, 0], tab[e][:, k])
cx = {e: col(e, 4) for e in eps}; R = {e: col(e, 8) for e in eps}

plt.rcParams.update({"font.size": 11})
fig = plt.figure(figsize=(11, 7.2))
gs = fig.add_gridspec(2, 2, height_ratios=[1, 1.35])
ax = [fig.add_subplot(gs[0, :]), fig.add_subplot(gs[1, 0]), fig.add_subplot(gs[1, 1])]

v = np.abs(hj).max()
im = ax[0].imshow(hj, origin="lower", extent=(-L / 2, L / 2, -W / 2, W / 2), cmap="RdBu_r",
                  vmin=-v, vmax=v, aspect="equal")
lev = np.linspace(-0.8 * v, 0.8 * v, 9)
ax[0].contour(x, y, hf, levels=lev, colors="k", linewidths=0.7)
yy = np.linspace(-W / 2, W / 2, 2)
ax[0].plot(0.3 * yy, yy, ":", color="0.35", lw=1.3)             # the wall centre, x = 0.3 y
ax[0].set_xlim(-L / 2, L / 2); ax[0].set_ylim(-W / 2, W / 2)
ax[0].set_xlabel("x (nm)"); ax[0].set_ylabel("y (nm)")
ax[0].set_title(r"(a) $\phi-\langle\phi\rangle_y$, tilted wall")
cb = fig.colorbar(im, ax=ax[0], fraction=0.02, pad=0.015, aspect=8); cb.set_label("mV")

cs = {0.02: "tab:blue", 0.04: "tab:red"}
for e in (0.02, 0.04):
    ax[1].plot(t * 1e9, (R[e] - R[e][0]) * 1e3, color=cs[e], lw=1, label=rf"$\Delta\rho/\rho_\perp={e}$")
ax[1].set_xlabel("t (ns)"); ax[1].set_ylabel(r"$R(t)-R(0)$ (m$\Omega$)")
ax[1].set_title("(b) resistance of the disc"); ax[1].legend(fontsize=9, loc="lower right")

ax[2].plot(t * 1e9, 2 * (cx[0.02] - cx[0.0]) * 1e9, color="#ff7f7f", lw=2.5, label=r"$2\,[x_{0.02}-x_0]$")
ax[2].plot(t * 1e9, (cx[0.04] - cx[0.0]) * 1e9, color="b", lw=1, label=r"$x_{0.04}-x_0$")
ax[2].set_xlabel("t (ns)"); ax[2].set_ylabel("core shift by AMR (nm)")
ax[2].set_ylim(-0.55, 0.68)
ax[2].set_title("(c) linearity in $\\Delta\\rho$"); ax[2].legend(fontsize=9, loc="upper right", ncol=2)

fig.tight_layout(); fig.savefig(out, dpi=200)
print("max |Hall part|", v, "mV; JuliaMag - FEM, largest", np.abs(hj - hf).max(), "mV")
