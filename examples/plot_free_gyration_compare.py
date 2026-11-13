"""Free (no current) vortex gyration in a 128 nm x 16 nm Py disk: JuliaMag vs mumax3.
Vortex relaxed under Bx = 10 mT, field switched off at t = 0, plain LLG (alpha = 0.01).

Usage: python plot_free_gyration_compare.py [tables dir] [out.png]
  default tables dir: ../gpu_validation/disk_stt/tables
  expects mumax_free_es0.txt (mumax3: t mx my mz corex corey corez)
  and free_es0_dt5.txt (JuliaMag: t mx my mz corex corey)
"""
import sys, os
import numpy as np
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
VAL = os.path.join(HERE, "..", "gpu_validation", "disk_stt")
sys.path.insert(0, VAL)
from gyrofreq import freq_phase  # noqa: E402

D = sys.argv[1] if len(sys.argv) > 1 else os.path.join(VAL, "tables")
OUT = sys.argv[2] if len(sys.argv) > 2 else os.path.join(HERE, "disk_free_gyration_compare.png")
mu = np.loadtxt(os.path.join(D, "mumax_free_es0.txt"), comments="#")
jm = np.loadtxt(os.path.join(D, "free_es0_dt5.txt"), comments="#")


def decay_rate(a):
    """Exponential decay rate of the in-plane <m> amplitude (fit of log |<m>_xy|)."""
    t = a[:, 0]; r = np.hypot(a[:, 1], a[:, 2]); w = t > 0.5e-9
    return -np.polyfit(t[w], np.log(r[w]), 1)[0]


def freq(a):
    return 0.5 * (freq_phase(a[:, 0], a[:, 1], 0.5e-9, 7e-9)
                  + freq_phase(a[:, 0], a[:, 2], 0.5e-9, 7e-9)) / 1e6


gmu, gjm = decay_rate(mu), decay_rate(jm)
fmu, fjm = freq(mu), freq(jm)
print(f"frequency   mumax3 {fmu:.2f} MHz   JuliaMag {fjm:.2f} MHz")
print(f"decay rate  mumax3 {gmu*1e-9:.4f} /ns   JuliaMag {gjm*1e-9:.4f} /ns   "
      f"ratio {gjm/gmu:.4f}")

fig, ax = plt.subplots(1, 3, figsize=(14, 4))
for a in ax[:2]:
    for c, lab in [(1, "x"), (2, "y")]:
        a.plot(mu[:, 0] * 1e9, mu[:, c], color="red", lw=2, alpha=0.5,
               label=f"mumax3 ⟨m{lab}⟩" if a is ax[0] else None)
        a.plot(jm[:, 0] * 1e9, jm[:, c], color="blue", lw=1, ls="-" if c == 1 else "--",
               label=f"JuliaMag ⟨m{lab}⟩" if a is ax[0] else None)
    a.set_xlabel("t (ns)"); a.grid(alpha=0.3)
ax[0].set_xlim(0, 7); ax[0].set_ylabel("⟨m⟩"); ax[0].legend(fontsize=8, ncol=2)
ax[0].set_title(f"mumax3 {fmu:.1f} MHz  ·  JuliaMag {fjm:.1f} MHz", fontsize=10)
ax[1].set_xlim(6, 7); ax[1].set_title("zoom 6–7 ns", fontsize=10)

for a, col, lab, g in [(mu, "red", "mumax3", gmu), (jm, "blue", "JuliaMag", gjm)]:
    ax[2].semilogy(a[:, 0] * 1e9, np.hypot(a[:, 1], a[:, 2]), color=col,
                   lw=2 if col == "red" else 1, alpha=0.5 if col == "red" else 1,
                   label=f"{lab}  Γ = {g*1e-9:.4f} ns⁻¹")
ax[2].set_xlabel("t (ns)"); ax[2].set_ylabel("|⟨m⟩_xy|"); ax[2].set_xlim(0, 7)
ax[2].legend(fontsize=8); ax[2].grid(alpha=0.3, which="both")
ax[2].set_title("in-plane amplitude decay", fontsize=10)
fig.suptitle("Free vortex gyration, 128 nm × 16 nm Py disk, EdgeSmooth=0, α=0.01", fontsize=11)
fig.tight_layout()
fig.savefig(OUT, dpi=150)
print("wrote", OUT)
