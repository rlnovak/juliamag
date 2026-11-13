# Benchmark cross-check JuliaMag vs mumax3: mx, my, mz AND the four energy terms
# on one figure, so any divergence shows up per-column. Usage:
#   python plot_benchmark_compare.py <name> <juliamag_table> <mumax_table> <out.png>
# Both tables: t mx my mz E_demag E_exch E_Zeeman E_total (JuliaMag), and the
# mumax table has the same first 8 columns (extra core-pos cols are ignored).

import sys
import numpy as np
import matplotlib.pyplot as plt

name, JT, MT, OUT = sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4]
J = np.loadtxt(JT, comments="#")
M = np.loadtxt(MT, comments="#")

cols = ["mx", "my", "mz", "E_demag", "E_exch", "E_Zeeman", "E_total"]
tJ, tM = J[:, 0] * 1e9, M[:, 0] * 1e9

fig, axes = plt.subplots(4, 2, figsize=(13, 11))
axes = axes.ravel()
for k, nm in enumerate(cols):
    ax = axes[k]
    c = k + 1
    ax.plot(tM, M[:, c], lw=2, alpha=0.5, color="red", label="mumax3")
    ax.plot(tJ, J[:, c], lw=1, color="blue", label="JuliaMag")
    # RMS diff on the mumax grid
    ji = np.interp(tM, tJ, J[:, c])
    d = ji - M[:, c]
    rms = np.sqrt(np.mean(d**2))
    scale = np.sqrt(np.mean(M[:, c]**2))
    pct = 100 * rms / scale if scale else 0.0
    ax.set_title(f"{nm}   (rms/scale = {pct:.3f}%)", fontsize=9)
    ax.set_xlabel("t (ns)")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=7, loc="best")

# hide the 8th empty panel
axes[7].axis("off")
fig.suptitle(f"{name}: JuliaMag vs mumax3 — components + energies", fontsize=12)
fig.tight_layout(rect=[0, 0, 1, 0.98])
fig.savefig(OUT, dpi=140)
print("Wrote", OUT)
