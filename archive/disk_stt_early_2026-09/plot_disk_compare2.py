# Two comparison figures, JuliaMag vs mumax3 (single 128 nm disk, Slonczewski 40 mA):
#   (1) vortex-core trajectory in the disk plane (core_x vs core_y);
#   (2) mx, my, mz vs time, all three series on one axis.
# In both: JuliaMag = lw 1 opaque; mumax3 = lw 2, alpha 0.5.

import numpy as np
import matplotlib.pyplot as plt

JUL = r"C:\Users\rlnov\Downloads\resultados projeto ufsc\single_disk\JuliaMag\disk_d1=128nm_t=16nm_40mA_table.txt"
MUM = r"C:\Users\rlnov\Downloads\resultados projeto ufsc\single_disk\sims1\disk_d1=128nm_t=16nm_40mA.out\table.txt"
OUTDIR = r"C:\Users\rlnov\Projetos\mumag\examples"

J = np.loadtxt(JUL, comments="#")
M = np.loadtxt(MUM, comments="#")

# JuliaMag cols: t=0, mx=1, my=2, mz=3, ..., core_x=10, core_y=11
tJ = J[:, 0] * 1e9
mxJ, myJ, mzJ = J[:, 1], J[:, 2], J[:, 3]
cxJ, cyJ = J[:, 10] * 1e9, J[:, 11] * 1e9
# mumax3 cols: t=0, mx=1, my=2, mz=3, ..., ext_coreposx=15, ext_coreposy=16
tM = M[:, 0] * 1e9
mxM, myM, mzM = M[:, 1], M[:, 2], M[:, 3]
cxM, cyM = M[:, 15] * 1e9, M[:, 16] * 1e9

JKW = dict(lw=1)                    # JuliaMag: thin, opaque
MKW = dict(lw=2, alpha=0.5)         # mumax3: thick, translucent

# --- Figure 1: vortex-core trajectory (x vs y) -----------------------------
fig1, ax = plt.subplots(figsize=(7, 7))
ax.plot(cxM, cyM, color="red",  label="mumax3 (ref)", **MKW)
ax.plot(cxJ, cyJ, color="blue", label="JuliaMag",     **JKW)
ax.set_xlabel("core x (nm)"); ax.set_ylabel("core y (nm)")
ax.set_title("Vortex core trajectory — 128 nm disk, 40 mA", fontsize=11)
ax.set_aspect("equal", adjustable="datalim")
ax.grid(True, alpha=0.3); ax.legend(loc="upper right", fontsize=9)
fig1.tight_layout()
p1 = OUTDIR + r"\disk_d1=128nm_t=16nm_40mA_core_traj.png"
fig1.savefig(p1, dpi=150); print("Wrote", p1)

# --- Figure 2: mx, my, mz vs time, one axis --------------------------------
fig2, ax = plt.subplots(figsize=(10, 5))
for ref, jul, name, cJ, cM in [
    (mxM, mxJ, "mx", "tab:blue",   "tab:red"),
    (myM, myJ, "my", "tab:green",  "tab:orange"),
    (mzM, mzJ, "mz", "tab:purple", "tab:brown"),
]:
    ax.plot(tM, ref, color=cM, label=f"{name} mumax3", **MKW)
    ax.plot(tJ, jul, color=cJ, label=f"{name} JuliaMag", **JKW)
ax.set_xlabel("t (ns)"); ax.set_ylabel("⟨m⟩")
ax.set_title("⟨m⟩ components vs time — 128 nm disk, 40 mA (JuliaMag vs mumax3)", fontsize=11)
ax.grid(True, alpha=0.3); ax.legend(loc="upper right", fontsize=8, ncol=3)
fig2.tight_layout()
p2 = OUTDIR + r"\disk_d1=128nm_t=16nm_40mA_m_components.png"
fig2.savefig(p2, dpi=150); print("Wrote", p2)
