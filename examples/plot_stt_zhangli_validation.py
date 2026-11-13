"""Single 128 nm x 16 nm Py disk, 40 mA: JuliaMag vs mumax3 with and without the
Zhang-Li torque (mumax3 applies Zhang-Li + Slonczewski by default when J is set).

Usage: python plot_stt_zhangli_validation.py [tables dir] [out.png]
  default tables dir: ../gpu_validation/disk_stt/tables
"""
import sys, os
import numpy as np
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
VAL = os.path.join(HERE, "..", "gpu_validation", "disk_stt")
sys.path.insert(0, VAL)
from gyrofreq import freq_phase  # noqa: E402

D = sys.argv[1] if len(sys.argv) > 1 else os.path.join(VAL, "tables")
OUT = sys.argv[2] if len(sys.argv) > 2 else os.path.join(HERE, "disk_40mA_zhangli_validation.png")


def tab(fn):
    a = np.loadtxt(os.path.join(D, fn), comments="#")
    return a[a[:, 0] <= 7e-9]


def freq(a):
    """Gyration frequency [MHz]: mean of the <mx>, <my> Hilbert-phase estimates, 1-7 ns."""
    return 0.5 * (freq_phase(a[:, 0], a[:, 1], 1e-9, 7e-9)
                  + freq_phase(a[:, 0], a[:, 2], 1e-9, 7e-9)) / 1e6


cases = [
    ("Slonczewski + Zhang-Li (mumax3 default)", tab("mumax_ref_es0.txt"), tab("sttzl_es0_dt5.txt")),
    ("Slonczewski only", tab("mumax_stt_es0_noZL.txt"), tab("jm_stt_es0_dt5.txt")),
]

fig, ax = plt.subplots(2, 2, figsize=(11, 6.5), sharey="row")
for c, (title, mu, jm) in enumerate(cases):
    fmu, fjm = freq(mu), freq(jm)
    print(f"{title:40s} mumax3 {fmu:.2f} MHz   JuliaMag {fjm:.2f} MHz")
    for r, (t1, t2) in enumerate([(0, 7), (6, 7)]):
        a = ax[r, c]
        a.plot(mu[:, 0] * 1e9, mu[:, 1], color="red", lw=2, alpha=0.5,
               label=f"mumax3  ({fmu:.1f} MHz)")
        a.plot(jm[:, 0] * 1e9, jm[:, 1], color="blue", lw=1,
               label=f"JuliaMag  ({fjm:.1f} MHz)")
        a.set_xlim(t1, t2)
        a.set_xlabel("t (ns)")
        if c == 0:
            a.set_ylabel(r"$\langle m_x \rangle$")
        if r == 0:
            a.set_title(title)
            a.legend(loc="upper left", fontsize=9)
        a.grid(alpha=0.3)
fig.suptitle("128 nm × 16 nm Py disk, 40 mA, EdgeSmooth=0 — vortex gyration", fontsize=11)
fig.tight_layout()
fig.savefig(OUT, dpi=150)
print("wrote", OUT)
