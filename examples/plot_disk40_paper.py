"""Paper figure for the 40 ns disc spin-torque oscillator series (EdgeSmooth 4,
Slonczewski torque only): JuliaMag (RK4, 50 fs) vs mumax3.

(a) <m_x>(t) over 40 ns.
(b) Gyration frequency in sliding 2 ns windows, from the phase of the in-plane <m>
    vector about the orbit centre (one-period moving average).
(c) maxTorque: JuliaMag, mumax3 with its default RK45 tolerance (MaxErr 1e-5) and
    mumax3 with MaxErr 1e-7. The default-tolerance plateau is integration error.

Usage: python plot_disk40_paper.py [TABLES_DIR] [out.png]
"""
import sys, os
import numpy as np
import matplotlib.pyplot as plt

sys.stdout.reconfigure(encoding="utf-8")
HERE = os.path.dirname(os.path.abspath(__file__))
TD = sys.argv[1] if len(sys.argv) > 1 else os.path.join(HERE, "..", "gpu_validation", "disk_stt", "tables")
OUT = sys.argv[2] if len(sys.argv) > 2 else os.path.join(HERE, "..", "paper", "figures", "disk_40ns_es4.png")
L = lambda f: np.loadtxt(os.path.join(TD, f), comments="#")
jm = L("jm_es4_40ns.txt")
mu = L("mumax_stt_es4_noZL_40ns.txt")
m7 = L("mumax_stt_es4_noZL_40ns_maxerr1e-7.txt")
T2 = min(jm[-1, 0], mu[-1, 0])
DT = 5e-12
tu = np.arange(0, T2, DT)


def fft_period(t, x, t1=1e-9):
    s = tu >= t1
    y = np.interp(tu[s], t, x); y = y - y.mean()
    n = 1 << 20; f = np.fft.rfftfreq(n, DT); P = np.abs(np.fft.rfft(y * np.hanning(len(y)), n))
    return 1 / f[1 + np.argmax(P[1:])]


P = fft_period(mu[:, 0], mu[:, 1])
nw = int(round(P / DT)) | 1
ker = np.ones(nw) / nw
h = nw // 2


def phase(d):
    x = np.interp(tu, d[:, 0], d[:, 1]); y = np.interp(tu, d[:, 0], d[:, 2])
    xc = np.convolve(x, ker, mode="same"); yc = np.convolve(y, ker, mode="same")
    return tu[h:-h], np.unwrap(np.arctan2((y - yc)[h:-h], (x - xc)[h:-h]))


tp, pj = phase(jm)
_, pm = phase(mu)
win = 2e-9
cent, fj, fm = [], [], []
for a in np.arange(1e-9, tp[-1] - win + 1e-15, 1e-9):
    s = (tp >= a) & (tp < a + win)
    cent.append((a + win / 2) * 1e9)
    fj.append(abs(np.polyfit(tp[s], pj[s], 1)[0]) / (2e6 * np.pi))
    fm.append(abs(np.polyfit(tp[s], pm[s], 1)[0]) / (2e6 * np.pi))
print(f"max |f_JM - f_mumax3| over the windows: {np.max(np.abs(np.array(fj) - np.array(fm))):.4f} MHz")

MU = dict(color="red", lw=2, alpha=0.5)
JMs = dict(color="blue", lw=1)
ns = 1e9
plt.rcParams.update({"font.size": 9})
fig, ax = plt.subplots(1, 3, figsize=(12, 3.5))
a = ax[0]
a.plot(mu[:, 0] * ns, mu[:, 1], **MU, label="mumax$^3$")
a.plot(jm[:, 0] * ns, jm[:, 1], **JMs, label="JuliaMag")
a.set_xlim(0, T2 * ns); a.set_xlabel("t (ns)"); a.set_ylabel(r"$\langle m_x\rangle$")
a.set_title(r"(a) $\langle m_x\rangle$", fontsize=9); a.legend(fontsize=8)
a = ax[1]
a.plot(cent, fm, "o-", **MU, ms=4, label="mumax$^3$")
a.plot(cent, fj, "s-", **JMs, ms=2.5, label="JuliaMag")
a.set_xlabel("window centre (ns)"); a.set_ylabel("f (MHz)")
a.set_title("(b) gyration frequency, 2 ns windows", fontsize=9); a.legend(fontsize=8)
a = ax[2]
a.semilogy(mu[:, 0] * ns, mu[:, 4] * 1e3, **MU, label=r"mumax$^3$, MaxErr $10^{-5}$ (default)")
a.semilogy(m7[:, 0] * ns, m7[:, 4] * 1e3, color="darkorange", lw=2, alpha=0.6,
           label=r"mumax$^3$, MaxErr $10^{-7}$")
a.semilogy(jm[:, 0] * ns, jm[:, 4] * 1e3, **JMs, label="JuliaMag")
a.set_xlim(0, T2 * ns); a.set_xlabel("t (ns)"); a.set_ylabel("maxTorque (mT)")
a.set_title("(c) maximum torque", fontsize=9); a.legend(fontsize=7.5)
for a in ax:
    a.grid(alpha=0.3)
fig.tight_layout()
fig.savefig(OUT, dpi=200)
print("wrote", os.path.abspath(OUT))
