# The vortex disc of vortex_amr.jl for three AMR ratios: the core, the resistance, and the linearity of
# the effect of AMR on the dynamics.
#
#   python gpu_validation/amr/compare_vortex.py DIR OUT.png > results/vortex.txt
#
# DIR holds table_drho{0.0,0.02,0.04}.txt. With Δρ/ρ⊥ = ε the conduction changes j by O(ε), so the
# change of the core trajectory and of R must be linear in ε at small ε: d(0.04) ≈ 2 d(0.02), with a
# remainder of relative size O(ε).
import sys, os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

d, out = sys.argv[1], sys.argv[2]
eps = (0.0, 0.02, 0.04)
tab = {e: np.loadtxt(os.path.join(d, f"table_drho{e}.txt"), comments="#") for e in eps}
# columns: t mx my mz vortexx vortexy vortexz vortexw R
t = np.linspace(0, min(tab[e][-1, 0] for e in eps), 601)
col = lambda e, k: np.interp(t, tab[e][:, 0], tab[e][:, k])
x = {e: col(e, 4) for e in eps}; y = {e: col(e, 5) for e in eps}; R = {e: col(e, 8) for e in eps}

print("vortex disc 128 nm x 10 nm, 2 mA through two leads, Zhang-Li (P 0.4, xi 0.05), 3 ns")
for e in eps:
    r = np.hypot(x[e], y[e])
    print(f"drho/rho_perp = {e:.2f}: R(0) = {R[e][0]:.6f} Ohm, R mean = {R[e].mean():.6f} Ohm, "
          f"R peak-to-peak = {np.ptp(R[e][t > 0.5e-9]):.2e} Ohm, core radius max = {r.max() * 1e9:.2f} nm")
# R(0): uniform-in-time part. Linear in ε: R(0.04) − R(0) = 2 (R(0.02) − R(0)) + O(ε²)
dR1 = R[0.02][0] - R[0.0][0]; dR2 = R[0.04][0] - R[0.0][0]
print(f"R(0): change 0.02 -> {dR1:.6f} Ohm, 0.04 -> {dR2:.6f} Ohm, ratio {dR2 / dR1:.4f} (linear: 2)")
dx1 = np.hypot(x[0.02] - x[0.0], y[0.02] - y[0.0]); dx2 = np.hypot(x[0.04] - x[0.0], y[0.04] - y[0.0])
rem = np.hypot((x[0.04] - x[0.0]) - 2 * (x[0.02] - x[0.0]), (y[0.04] - y[0.0]) - 2 * (y[0.02] - y[0.0]))
print(f"core displacement by AMR: max {dx1.max() * 1e9:.3f} nm (0.02), {dx2.max() * 1e9:.3f} nm (0.04); "
      f"max |d(0.04) - 2 d(0.02)| = {rem.max() * 1e9:.3f} nm ({rem.max() / dx2.max():.2f} of d(0.04))")

fig, ax = plt.subplots(1, 3, figsize=(13, 3.8))
cs = {0.0: "k", 0.02: "tab:blue", 0.04: "tab:red"}
for e in eps:
    ax[0].plot(x[e] * 1e9, y[e] * 1e9, color=cs[e], lw=1, label=f"Δρ/ρ⊥ = {e}")
    ax[1].plot(t * 1e9, R[e] - R[e][0], color=cs[e], lw=1)
ax[0].set_xlabel("core x (nm)"); ax[0].set_ylabel("core y (nm)"); ax[0].set_aspect("equal"); ax[0].legend(fontsize=8)
ax[1].set_xlabel("t (ns)"); ax[1].set_ylabel("R(t) − R(0) (Ω)")
ax[2].plot(t * 1e9, 2 * (x[0.02] - x[0.0]) * 1e9, color="tab:red", lw=2, alpha=0.5, label="2 [x(0.02) − x(0)]")
ax[2].plot(t * 1e9, (x[0.04] - x[0.0]) * 1e9, color="tab:blue", lw=1, label="x(0.04) − x(0)")
ax[2].set_xlabel("t (ns)"); ax[2].set_ylabel("core shift by AMR (nm)"); ax[2].legend(fontsize=8)
fig.tight_layout(); fig.savefig(out, dpi=130)
