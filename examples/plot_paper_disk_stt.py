"""Paper figure: spin-torque vortex oscillator in a 128 nm x 16 nm Py disc,
JuliaMag vs mumax3 (EdgeSmooth 0). Writes paper/figures/disk_stt_validation.png.

(a) <m_x>(t), Slonczewski torque only (the physical model), 0-7 ns.
(b) 5-7 ns, Slonczewski only (solid) vs Slonczewski + Zhang-Li (dashed, mumax3's
    default when a current is set): the Zhang-Li term shifts the frequency, and
    both codes reproduce the shift.
(c) free gyration (no current): in-plane amplitude |<m>_xy| on a log scale.

Usage: python plot_paper_disk_stt.py [tables dir] [out.png]
"""
import sys, os
import numpy as np
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
VAL = os.path.join(HERE, "..", "gpu_validation", "disk_stt")
sys.path.insert(0, VAL)
from gyrofreq import freq_phase  # noqa: E402

D = sys.argv[1] if len(sys.argv) > 1 else os.path.join(VAL, "tables")
OUT = sys.argv[2] if len(sys.argv) > 2 else os.path.join(HERE, "..", "paper", "figures",
                                                         "disk_stt_validation.png")


def tab(fn):
    a = np.loadtxt(os.path.join(D, fn), comments="#")
    return a[a[:, 0] <= 7e-9]


def freq(a, t1=1e-9):
    return 0.5 * (freq_phase(a[:, 0], a[:, 1], t1, 7e-9)
                  + freq_phase(a[:, 0], a[:, 2], t1, 7e-9)) / 1e6


def decay(a):
    t = a[:, 0]; r = np.hypot(a[:, 1], a[:, 2]); w = t > 0.5e-9
    return -np.polyfit(t[w], np.log(r[w]), 1)[0] * 1e-9


mu_sl, jm_sl = tab("mumax_stt_es0_noZL.txt"), tab("jm_stt_es0_dt5.txt")
mu_zl, jm_zl = tab("mumax_ref_es0.txt"), tab("sttzl_es0_dt5.txt")
mu_fr, jm_fr = tab("mumax_free_es0.txt"), tab("free_es0_dt5.txt")
MU = dict(color="red", lw=2, alpha=0.5)
JM = dict(color="blue", lw=1)
ns = 1e9

plt.rcParams.update({"font.size": 9})
fig, ax = plt.subplots(1, 3, figsize=(12, 3.6))

a = ax[0]
a.plot(mu_sl[:, 0] * ns, mu_sl[:, 1], **MU, label=f"mumax$^3$ ({freq(mu_sl):.1f} MHz)")
a.plot(jm_sl[:, 0] * ns, jm_sl[:, 1], **JM, label=f"JuliaMag ({freq(jm_sl):.1f} MHz)")
a.set_xlim(0, 7); a.set_xlabel("t (ns)"); a.set_ylabel(r"$\langle m_x\rangle$")
a.set_title("(a) Slonczewski torque", fontsize=9); a.legend(fontsize=8, loc="lower right")

a = ax[1]
a.plot(mu_sl[:, 0] * ns, mu_sl[:, 1], **MU, label="mumax$^3$, Slonczewski")
a.plot(jm_sl[:, 0] * ns, jm_sl[:, 1], **JM, label="JuliaMag, Slonczewski")
a.plot(mu_zl[:, 0] * ns, mu_zl[:, 1], **MU, ls="--",
       label=f"mumax$^3$, + Zhang–Li ({freq(mu_zl):.1f} MHz)")
a.plot(jm_zl[:, 0] * ns, jm_zl[:, 1], **JM, ls="--",
       label=f"JuliaMag, + Zhang–Li ({freq(jm_zl):.1f} MHz)")
a.set_xlim(5, 7); a.set_ylim(-0.14, 0.06); a.set_xlabel("t (ns)")
a.set_title("(b) with and without the Zhang–Li term", fontsize=9)
a.legend(fontsize=7, loc="lower left")

a = ax[2]
a.semilogy(mu_fr[:, 0] * ns, np.hypot(mu_fr[:, 1], mu_fr[:, 2]), **MU,
           label=f"mumax$^3$ ({freq(mu_fr, 0.5e-9):.1f} MHz, $\\Gamma$={decay(mu_fr):.4f} ns$^{{-1}}$)")
a.semilogy(jm_fr[:, 0] * ns, np.hypot(jm_fr[:, 1], jm_fr[:, 2]), **JM,
           label=f"JuliaMag ({freq(jm_fr, 0.5e-9):.1f} MHz, $\\Gamma$={decay(jm_fr):.4f} ns$^{{-1}}$)")
a.set_xlim(0, 7); a.set_xlabel("t (ns)"); a.set_ylabel(r"$|\langle m\rangle_{xy}|$")
a.set_title("(c) free gyration, no current", fontsize=9); a.legend(fontsize=7)

for a in ax:
    a.grid(alpha=0.3, which="both")
fig.tight_layout()
fig.savefig(OUT, dpi=200)
print("wrote", os.path.abspath(OUT))
