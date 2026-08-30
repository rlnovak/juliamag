# Vortex-core path vs time: JuliaMag (single 128 nm disk, Slonczewski STT, 40 mA)
# against the mumax3 reference. Two stacked panels: core_x vs t and core_y vs t.
# mumax3 = thick red translucent (lw=2, alpha=0.4); JuliaMag = thin opaque blue.

import numpy as np
import matplotlib.pyplot as plt

JUL = r"C:\Users\rlnov\Downloads\resultados projeto ufsc\single_disk\JuliaMag\disk_d1=128nm_t=16nm_40mA_table.txt"
MUM = r"C:\Users\rlnov\Downloads\resultados projeto ufsc\single_disk\sims1\disk_d1=128nm_t=16nm_40mA.out\table.txt"
OUT = r"C:\Users\rlnov\Projetos\mumag\examples\disk_d1=128nm_t=16nm_40mA_core_compare.png"

J = np.loadtxt(JUL, comments="#")
M = np.loadtxt(MUM, comments="#")

# JuliaMag columns: t=0, ..., core_x=10, core_y=11
tJ = J[:, 0] * 1e9
cxJ, cyJ = J[:, 10] * 1e9, J[:, 11] * 1e9
# mumax3 columns: t=0, ext_coreposx=15, ext_coreposy=16
tM = M[:, 0] * 1e9
cxM, cyM = M[:, 15] * 1e9, M[:, 16] * 1e9

fig, axes = plt.subplots(2, 1, figsize=(9, 7), sharex=True)
panels = [("core x (nm)", cxM, cxJ), ("core y (nm)", cyM, cyJ)]
for ax, (ylabel, ref, jul) in zip(axes, panels):
    ax.plot(tM, ref, lw=2, alpha=0.4, color="red", label="mumax3 (ref)")
    ax.plot(tJ, jul, lw=1, color="blue", label="JuliaMag")
    ax.set_ylabel(ylabel)
    ax.grid(True, alpha=0.3)
    ax.legend(loc="upper right", fontsize=8)

axes[-1].set_xlabel("t (ns)")
axes[0].set_title("Vortex core path — 128 nm disk, 40 mA (JuliaMag vs mumax3)", fontsize=10)
fig.tight_layout()
fig.savefig(OUT, dpi=150)
print("Wrote", OUT)
