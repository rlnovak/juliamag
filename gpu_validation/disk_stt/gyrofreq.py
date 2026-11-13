# Gyration frequency of a vortex-core signal, robust on short windows.
# Method: remove the slow drift (centred moving average over ~1 period), take the
# analytic signal (Hilbert), unwrap its phase and fit a line -> f = slope / 2π.
# A zero-padded FFT peak is reported as a cross-check.
import numpy as np
from scipy.signal import hilbert

def _detrend(t, x, period):
    dt = t[1] - t[0]; w = max(3, int(round(period / dt)) | 1)
    k = np.ones(w) / w
    xm = np.convolve(x, k, mode="same")
    h = w // 2
    return t[h:-h], (x - xm)[h:-h]

def freq_phase(t, x, t1, t2, period=1.08e-9):
    s = (t >= t1) & (t <= t2)
    tu = np.linspace(t[s][0], t[s][-1], s.sum())     # uniform grid for the moving average
    tt, y = _detrend(tu, np.interp(tu, t[s], x[s]), period)
    ph = np.unwrap(np.angle(hilbert(y)))
    e = len(tt) // 10                              # drop Hilbert edge effects
    p = np.polyfit(tt[e:-e], ph[e:-e], 1)
    return abs(p[0]) / (2 * np.pi)

def freq_fft(t, x, t1, t2):
    # mumax3 tables have non-uniform t (adaptive steps): resample on a uniform grid
    s = (t >= t1) & (t <= t2); tu = np.linspace(t[s][0], t[s][-1], s.sum())
    y = np.interp(tu, t[s], x[s]); y = y - y.mean()
    n = 1 << 20; f = np.fft.rfftfreq(n, tu[1] - tu[0]); P = np.abs(np.fft.rfft(y * np.hanning(len(y)), n))
    return f[1 + np.argmax(P[1:])]

def load(fn, cx, cy):
    A = np.loadtxt(fn, comments="#")
    return A[:, 0], A[:, cx], A[:, cy]

def report(label, t, x, y, t1, t2):
    fx, fy = freq_phase(t, x, t1, t2), freq_phase(t, y, t1, t2)
    ff = freq_fft(t, x, t1, t2)
    r = np.hypot(x - np.mean(x[-200:]), y - np.mean(y[-200:])) * 1e9
    print(f"{label:34s} f(phase) x={fx/1e6:7.2f} y={fy/1e6:7.2f}  fft={ff/1e6:7.1f} MHz   r_max={r.max():5.2f} nm")
    return 0.5 * (fx + fy)
