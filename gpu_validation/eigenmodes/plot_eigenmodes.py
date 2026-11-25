"""Figure of the paper: linear eigenmodes (results -> paper/figures/eigenmodes.png)."""
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HERE = Path(__file__).parent
R = HERE / "results"
OUT = HERE.parents[1] / "paper" / "figures" / "eigenmodes.png"
RED = dict(color="red", lw=2, alpha=0.5)
BLUE = dict(color="blue", lw=1)


def ovf(path):
    n, rows, data = {}, [], False
    for l in open(path, encoding="utf-8"):
        if l.startswith("#"):
            w = l[1:].split()
            if len(w) > 1 and w[0].rstrip(":") in ("xnodes", "ynodes", "znodes"):
                n[w[0][0]] = int(w[1])
        elif l.strip():
            rows.append([float(x) for x in l.split()])
    return np.array(rows).reshape(n["z"], n["y"], n["x"], 3)


def amplitude(k):
    d = ovf(R / "modes" / f"mode{k:03d}_re.ovf") + 1j * ovf(R / "modes" / f"mode{k:03d}_im.ovf")
    return np.sqrt((np.abs(d) ** 2).sum(-1)).mean(0)


fig = plt.figure(figsize=(10.5, 3.3), constrained_layout=True)
gs = fig.add_gridspec(2, 3, width_ratios=[1.5, 0.55, 1.1])

ax = fig.add_subplot(gs[:, 0])
s = np.loadtxt(R / "oommf_spectrum.txt")
ax.semilogy(s[:, 0], s[:, 1], label="OOMMF ringdown", **RED)
f = [float(l.split("\t")[1]) for l in open(R / "fmr_eigenmodes.txt") if not l.startswith("#")][:15]
for i, x in enumerate(f):
    ax.axvline(x, ls="--", label="JuliaMag eigenmodes" if i == 0 else None, **BLUE)
ax.set(xlim=(7, 20.5), ylim=(1e-5, 20), xlabel="frequency (GHz)", ylabel=r"$\tilde S_y$ (arb. units)", title="(a) FMR standard problem")
ax.legend(frameon=True, fontsize=8)

for i, k in enumerate((1, 4)):
    a = fig.add_subplot(gs[i, 1])
    a.imshow(amplitude(k), origin="lower", cmap="magma", extent=(0, 120, 0, 120))
    a.set_title(f"{f[k-1]:.2f} GHz" if i else f"(b) {f[k-1]:.2f} GHz", fontsize=9)
    a.set(xticks=[], yticks=[])

ax = fig.add_subplot(gs[:, 2])
amp, fr = [], []
for b in (10, 5, 2, 1):
    d = np.loadtxt(R / "gyro" / f"gyro_{b}mT.txt")
    t, z = d[:, 0] * 1e9, d[:, 1] + 1j * d[:, 2]
    k = t >= 2
    amp.append((np.abs(z[k]) ** 2).mean())
    fr.append(np.polyfit(t[k], np.unwrap(np.angle(z[k])), 1)[0] / 2 / np.pi * 1e3)
amp, fr = np.array(amp), np.array(fr)
p = np.polyfit(amp, fr, 1)
x = np.linspace(0, amp.max() * 1.05, 10)
ax.plot(x * 1e3, np.polyval(p, x), label="LLG, linear fit", **RED)
ax.plot(amp * 1e3, fr, "o", color="red", alpha=0.5, label="LLG, free gyration")
fe, fd = 968.2342, 968.2342 * (1 - 1.678e-4)
ax.plot(0, fd, marker="s", ms=7, mfc="none", ls="none", color="blue", label="eigenmode, shifted by damping")
ax.set(xlabel=r"mean $|\langle m_x\rangle + i\langle m_y\rangle|^2$ ($10^{-3}$)", ylabel="frequency (MHz)", title="(c) vortex gyration", xlim=(-0.1, None))
ax.legend(frameon=False, fontsize=8)
fig.savefig(OUT, dpi=200)
print(OUT, p[1], fd)
