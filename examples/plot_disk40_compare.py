"""40 ns production series, 128 nm x 16 nm Py disc, 40 mA, Slonczewski torque only,
EdgeSmooth 4: JuliaMag (GPU, RK4 dt 5e-14 s) vs mumax3 (adaptive RK45).

Inputs (tables written by gpu_validation/disk_stt/run_queue_disk40.sh):
  JuliaMag: t mx my mz maxTorque E_demag E_exch E_Zeeman E_total Jz core_x core_y
  mumax3  : t mx my mz maxTorque E_demag E_exch E_Zeeman E_total Jx Jy Jz core_x core_y core_z
Prints the frequencies (Hilbert phase on <mx>, <my>; FFT cross-check), the maximum
|Δ<m>| on the common time span (no extrapolation), and the energy / maxTorque
deviations; writes a 4-panel figure.

Usage: python plot_disk40_compare.py JM_TABLE MUMAX_TABLE [out.png]
"""
import sys, os
import numpy as np
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "gpu_validation", "disk_stt"))
from gyrofreq import freq_phase, freq_fft  # noqa: E402

sys.stdout.reconfigure(encoding="utf-8")   # Δ, ⟨⟩ on a Windows cp1252 console

jm = np.loadtxt(sys.argv[1], comments="#")
mu = np.loadtxt(sys.argv[2], comments="#")
OUT = sys.argv[3] if len(sys.argv) > 3 else os.path.join(HERE, "disk_40ns_es4_compare.png")

# column maps -> common names
J = dict(t=jm[:, 0], mx=jm[:, 1], my=jm[:, 2], mz=jm[:, 3], tq=jm[:, 4],
         Ed=jm[:, 5], Ee=jm[:, 6], Et=jm[:, 8], cx=jm[:, 10], cy=jm[:, 11])
M = dict(t=mu[:, 0], mx=mu[:, 1], my=mu[:, 2], mz=mu[:, 3], tq=mu[:, 4],
         Ed=mu[:, 5], Ee=mu[:, 6], Et=mu[:, 8], cx=mu[:, 12], cy=mu[:, 13])
T1 = 1e-9
T2 = min(J["t"][-1], M["t"][-1])


def freqs(d, t1=T1, t2=T2):
    ff = freq_fft(d["t"], d["mx"], t1, t2)
    fx = freq_phase(d["t"], d["mx"], t1, t2, period=1 / ff)
    fy = freq_phase(d["t"], d["my"], t1, t2, period=1 / ff)
    return fx, fy, ff


print(f"common span 0 – {T2*1e9:.3f} ns; frequency window {T1*1e9:.0f} – {T2*1e9:.1f} ns")
for name, d in (("mumax3", M), ("JuliaMag", J)):
    fx, fy, ff = freqs(d)
    print(f"{name:9s} f_x = {fx/1e6:8.2f}  f_y = {fy/1e6:8.2f}  mean = {(fx+fy)/2e6:8.2f}"
          f"  FFT = {ff/1e6:7.1f} MHz")

# deviations on the common span: interpolate mumax3 (non-uniform t) onto the JuliaMag
# grid restricted to t <= T2, so nothing is extrapolated.
s = J["t"] <= T2
for k in ("mx", "my", "mz"):
    d = J[k][s] - np.interp(J["t"][s], M["t"], M[k])
    print(f"max |Δ<{k}>| = {np.max(np.abs(d)):.2e}   rms = {np.sqrt(np.mean(d**2)):.2e}")
for k, lab in (("Et", "E_total"), ("Ed", "E_demag"), ("Ee", "E_exch"), ("tq", "maxTorque")):
    ref = np.interp(J["t"][s], M["t"], M[k])
    rel = (J[k][s] - ref) / np.abs(ref)
    print(f"{lab:9s} t=0: JM {J[k][0]:.6e}  mumax3 {M[k][0]:.6e}   "
          f"max rel dev over the run {np.max(np.abs(rel)):.2e}")
# core position (vortexcore vs ext_corepos, same algorithm since 01a62ff)
for k in ("cx", "cy"):
    d = J[k][s] - np.interp(J["t"][s], M["t"], M[k])
    print(f"max |Δ{k}| = {np.max(np.abs(d))*1e9:.3f} nm   rms = {np.sqrt(np.mean(d**2))*1e9:.3f} nm")
rj = np.hypot(J["cx"], J["cy"]); rm = np.hypot(M["cx"], M["cy"])
tail = J["t"] >= T2 - 2e-9
print(f"orbit radius, last 2 ns: JM {rj[tail].mean()*1e9:.3f} nm  "
      f"mumax3 {rm[(M['t'] >= T2 - 2e-9) & (M['t'] <= T2)].mean()*1e9:.3f} nm")

# Sliding frequency (2 ns windows): does the gyration settle the same way?
# Computed once over the whole series, not per window (a per-window Hilbert fit has
# only ~1 usable period left after its edge cuts): resample on a uniform grid, remove
# the orbit centre with a moving average over exactly one period (1/f_FFT), take the
# phase of the in-plane <m> vector about that centre, f = slope of the phase / 2π.
P = 1 / freq_fft(M["t"], M["mx"], T1, T2)
tu = np.arange(0, T2, 5e-12)
nw = int(round(P / 5e-12)) | 1
ker = np.ones(nw) / nw
h = nw // 2


def gyrophase(d):
    x = np.interp(tu, d["t"], d["mx"]); y = np.interp(tu, d["t"], d["my"])
    xc = np.convolve(x, ker, mode="same"); yc = np.convolve(y, ker, mode="same")
    return tu[h:-h], np.unwrap(np.arctan2((y - yc)[h:-h], (x - xc)[h:-h]))


tp, phj = gyrophase(J)
_, phm = gyrophase(M)
win = 2e-9
cent, fjm, fmu = [], [], []
for a in np.arange(1e-9, T2 - win - P + 1e-15, 1e-9):
    s2 = (tp >= a) & (tp < a + win)
    cent.append((a + win / 2) * 1e9)
    fjm.append(abs(np.polyfit(tp[s2], phj[s2], 1)[0]) / (2e6 * np.pi))
    fmu.append(abs(np.polyfit(tp[s2], phm[s2], 1)[0]) / (2e6 * np.pi))
fjm, fmu = np.array(fjm), np.array(fmu)
print(f"sliding 2 ns windows: f from {min(fjm.min(), fmu.min()):.2f} to "
      f"{max(fjm.max(), fmu.max()):.2f} MHz; max |f_JM - f_mumax3| = {np.abs(fjm - fmu).max():.3f} MHz")
s3 = tp >= T1
fglob = [abs(np.polyfit(tp[s3], ph[s3], 1)[0]) / (2e6 * np.pi) for ph in (phj, phm)]
print(f"orbit-phase frequency {T1*1e9:.0f}–{tp[-1]*1e9:.1f} ns: JM {fglob[0]:.3f}  mumax3 {fglob[1]:.3f} MHz")

MU = dict(color="red", lw=2, alpha=0.5)
JMs = dict(color="blue", lw=1)
ns = 1e9
plt.rcParams.update({"font.size": 9})
fig, ax = plt.subplots(2, 2, figsize=(11, 6.4))
a = ax[0, 0]
a.plot(M["t"] * ns, M["mx"], **MU, label="mumax$^3$")
a.plot(J["t"] * ns, J["mx"], **JMs, label="JuliaMag")
a.set_xlim(0, T2 * ns); a.set_xlabel("t (ns)"); a.set_ylabel(r"$\langle m_x\rangle$")
a.set_title("(a) Slonczewski torque, EdgeSmooth 4", fontsize=9); a.legend(fontsize=8)
a = ax[0, 1]
a.plot(M["t"] * ns, M["mx"], **MU, label="mumax$^3$")
a.plot(J["t"] * ns, J["mx"], **JMs, label="JuliaMag")
a.set_xlim(T2 * ns - 2, T2 * ns); a.set_xlabel("t (ns)"); a.set_ylabel(r"$\langle m_x\rangle$")
a.set_title("(b) last 2 ns", fontsize=9); a.legend(fontsize=8)
a = ax[1, 0]
a.plot(cent, fmu, "o-", **MU, ms=4, label="mumax$^3$")
a.plot(cent, fjm, "s-", **JMs, ms=3, label="JuliaMag")
a.set_xlabel("window centre (ns)"); a.set_ylabel("f (MHz), 2 ns windows")
a.set_title("(c) gyration frequency vs time", fontsize=9); a.legend(fontsize=8)
a = ax[1, 1]
a.plot(M["t"] * ns, M["Et"], **MU, label="mumax$^3$")
a.plot(J["t"] * ns, J["Et"], **JMs, label="JuliaMag")
a.set_xlim(0, T2 * ns); a.set_xlabel("t (ns)"); a.set_ylabel(r"$E_\mathrm{total}$ (J)")
a.set_title("(d) total energy", fontsize=9); a.legend(fontsize=8)
for a in ax.flat:
    a.grid(alpha=0.3)
fig.tight_layout()
fig.savefig(OUT, dpi=200)
print("wrote", os.path.abspath(OUT))
