"""Vortex-core tracks of the two 50 K re-runs with snapshots (steps 62-68, B = 0 ... -30 mT).

The core is the cell of max |m_z| (mumax3 ext_corepos) with sub-cell interpolation;
it is a vortex core only when that |m_z| is large (> 0.7: the thermal C-state has
max |m_z| ~ 0.3). Prints, per field step, the fraction of time a core exists and its
mean position over the last 0.5 ns of the 2 ns dwell, and the entry/exit points.
"""
import numpy as np

R = 250e-9                    # disc radius
T0 = 1.22e-7                  # start of step 62 (B = 0)
DWELL = 2e-9
FIELDS = [0, -5, -10, -15, -20, -25, -30]          # mT, steps 62..68
PMIN = 0.7


def load(seed):
    d = np.loadtxt(f"core_seed{seed}.txt", comments="#")
    t, mx, x, y, p = d[:, 0], d[:, 1], d[:, 4], d[:, 5], d[:, 7]
    keep = t >= T0 - 1e-15
    return t[keep], mx[keep], x[keep], y[keep], p[keep]


def report(seed):
    t, mx, x, y, p = load(seed)
    core = np.abs(p) > PMIN
    print(f"\nseed {seed}: {len(t)} rows, t = {t[0]:.4e} ... {t[-1]:.4e} s")
    for k, B in enumerate(FIELDS):
        w = (t >= T0 + k * DWELL) & (t < T0 + (k + 1) * DWELL)
        late = w & (t >= T0 + (k + 1) * DWELL - 0.5e-9)
        frac = core[w].mean() if w.any() else np.nan
        s = f"  B = {B:+4d} mT  <mx> end {mx[w][-1]:+.3f}  core {100*frac:5.1f} %"
        cl = late & core
        if cl.sum() > 0:
            r = np.hypot(x[cl], y[cl])
            s += (f"  late core (x, y) = ({1e9*x[cl].mean():+6.1f}, {1e9*y[cl].mean():+6.1f}) nm"
                  f"  r = {1e9*r.mean():5.1f} nm (r/R {r.mean()/R:.2f})  p = {np.sign(p[cl]).mean():+.0f}")
        print(s)
    # entry and exit: first and last row of the longest run of core rows
    idx = np.flatnonzero(core)
    if idx.size:
        runs = np.split(idx, np.flatnonzero(np.diff(idx) > 1) + 1)
        run = max(runs, key=len)
        a, b = run[0], run[-1]
        Ba = FIELDS[min(int((t[a] - T0) // DWELL), 6)]
        Bb = FIELDS[min(int((t[b] - T0) // DWELL), 6)]
        print(f"  core appears t = {t[a]:.4e} s (B = {Ba} mT) at ({1e9*x[a]:+.1f}, {1e9*y[a]:+.1f}) nm,"
              f" r/R = {np.hypot(x[a], y[a])/R:.2f}")
        print(f"  last core    t = {t[b]:.4e} s (B = {Bb} mT) at ({1e9*x[b]:+.1f}, {1e9*y[b]:+.1f}) nm,"
              f" r/R = {np.hypot(x[b], y[b])/R:.2f}"
              + ("  [end of run]" if b == len(t) - 1 else ""))
    return t, mx, x, y, p, core


if __name__ == "__main__":
    for s in (1, 7):
        report(s)
