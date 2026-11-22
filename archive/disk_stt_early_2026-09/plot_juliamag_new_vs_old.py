# JuliaMag current table vs an earlier JuliaMag run (--old), single 128 nm disk.
# Purpose: check whether the current table reflects the exchange edge-fill fix
# (commit bad1f2c). It does NOT — the gyration frequency is 974.9 MHz in both,
# so both are pre-fix runs; the small amplitude difference is run-to-run noise.
# mumax3 (909.9 MHz) is drawn as the reference the fix should move JuliaMag toward.

import numpy as np
import matplotlib.pyplot as plt

D = r"C:\Users\rlnov\Downloads\resultados projeto ufsc\single_disk\JuliaMag"
MUM = r"C:\Users\rlnov\Downloads\resultados projeto ufsc\single_disk\sims1\disk_d1=128nm_t=16nm_40mA.out\table.txt"
OUT = r"C:\Users\rlnov\Projetos\mumag\examples\disk_40mA_juliamag_new_vs_old.png"

N = np.loadtxt(D + r"\disk_d1=128nm_t=16nm_40mA_table.txt", comments="#")
O = np.loadtxt(D + r"\disk_d1=128nm_t=16nm_40mA_table--old.txt", comments="#")
M = np.loadtxt(MUM, comments="#")

tN, tO, tM = N[:, 0] * 1e9, O[:, 0] * 1e9, M[:, 0] * 1e9

fig, axes = plt.subplots(3, 1, figsize=(11, 8), sharex=True)
for ax, (cN, name) in zip(axes, [(1, "mx"), (2, "my"), (3, "mz")]):
    cM = cN  # mumax mx/my/mz are cols 1/2/3
    ax.plot(tM, M[:, cM], color="red",   lw=2, alpha=0.4, label="mumax3 (ref)")
    ax.plot(tO, O[:, cN], color="green", lw=1.5, alpha=0.7, label="JuliaMag (old run)")
    ax.plot(tN, N[:, cN], color="blue",  lw=1, label="JuliaMag (current)")
    ax.set_ylabel(f"⟨{name}⟩")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="upper right", fontsize=8)

axes[-1].set_xlabel("t (ns)")
axes[0].set_title("JuliaMag current vs old vs mumax3 — 128 nm disk, 40 mA "
                  "(current & old both 974.9 MHz → both pre-fix)", fontsize=10)
fig.tight_layout()
fig.savefig(OUT, dpi=150)
print("Wrote", OUT)
