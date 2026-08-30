import numpy as np, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt, os

jm = r"C:/Users/rlnov/Downloads/resultados projeto ufsc/single_disk/JuliaMag/disk_d1=128nm_t=16nm_40mA_table.txt"
mx = r"C:/Users/rlnov/Downloads/resultados projeto ufsc/single_disk/sims1/disk_d1=128nm_t=16nm_40mA.out/table.txt"
outdir = r"C:/Users/rlnov/Projetos/mumag/examples"
os.makedirs(outdir, exist_ok=True)

J = np.loadtxt(jm, comments="#"); M = np.loadtxt(mx, comments="#")
tJ = J[:,0]*1e9; tM = M[:,0]*1e9
comps = [("mx",1),("my",2),("mz",3)]

fig, axes = plt.subplots(3,1, figsize=(8,9), sharex=True)
for ax,(name,c) in zip(axes,comps):
    ax.plot(tM, M[:,c], lw=3, alpha=0.4, color="red",  label="mumax3 (ref)")
    ax.plot(tJ, J[:,c], lw=1,            color="blue", label="JuliaMag")
    ax.set_ylabel(f"⟨{name}⟩"); ax.grid(alpha=0.3); ax.legend(loc="best", fontsize=8)
axes[-1].set_xlabel("t (ns)")
axes[0].set_title("Single 128 nm × 16 nm disk, 40 mA — JuliaMag vs mumax3")
plt.tight_layout()
out = os.path.join(outdir, "disk_d1=128nm_t=16nm_40mA_compare.png")
plt.savefig(out, dpi=130); print("wrote", out)
