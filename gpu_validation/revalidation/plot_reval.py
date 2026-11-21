"""Figures of the re-validation (2026-11) from the new tables, with matplotlib.

Writes the paper figures (paper/figures/) for standard problems 2 and 4 and the
skyrmion study, and the example figures (examples/, gpu_validation/std4/) for the
same problems plus the magnum.np stage-2 problems and the thermal nanodot. The disc
figures have their own scripts (examples/plot_paper_disk_stt.py,
plot_disk40_paper.py, plot_disc_hysteresis.py, …).

Style: references (OOMMF, mumax3) red, lw 2, alpha 0.5; JuliaMag blue, lw 1.

    python gpu_validation/revalidation/plot_reval.py
"""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
EX, FIG = ROOT / "examples", ROOT / "paper" / "figures"
REF = dict(color="red", lw=2, alpha=0.5)
JM = dict(color="blue", lw=1)
plt.rcParams.update({"font.size": 9, "axes.grid": True, "grid.alpha": 0.3})


def save(fig, *paths):
    for p in paths:
        fig.savefig(p, dpi=200, bbox_inches="tight")
        print("wrote", p.relative_to(ROOT))
    plt.close(fig)


def oommf_std2():
    rows = [[float(x) for x in l.split()[:6]] for l in (EX / "stdprob2_oommf.txt").read_text(encoding="utf-8").splitlines()
            if l.split() and l.split()[0].replace(".", "", 1).isdigit()]
    return np.array(sorted(rows))


def std2():
    j = np.loadtxt(EX / "stdproblem2.txt", comments="#"); o = oommf_std2()
    fig, ax = plt.subplots(figsize=(4.6, 3.2))
    ax.plot(o[:, 0], o[:, 4], **REF, label=r"$\langle m_x\rangle$ OOMMF")
    ax.plot(o[:, 0], o[:, 5], **REF, ls="--", label=r"$\langle m_y\rangle$ OOMMF")
    ax.plot(j[:, 0], j[:, 1], "o", ms=4, mfc="none", mec="blue", label=r"$\langle m_x\rangle$ JuliaMag")
    ax.plot(j[:, 0], j[:, 2], "s", ms=4, mfc="none", mec="blue", label=r"$\langle m_y\rangle$ JuliaMag")
    ax.set_xlabel(r"$d/\ell_{\mathrm{ex}}$"); ax.set_ylabel(r"remanent $\langle m\rangle$")
    ax.set_ylim(-0.05, 1.05); ax.legend(fontsize=7, loc="center right")
    save(fig, FIG / "stdproblem2_compare.png", EX / "stdproblem2_compare.png")

    h = np.loadtxt(EX / "stdproblem2_hysteresis.txt", comments="#")
    fig, ax = plt.subplots(figsize=(4.6, 3.2))
    ax.plot(o[:, 0], o[:, 2], **REF, label=r"$H_c/M_s$ OOMMF")
    ax.plot(o[:, 0], o[:, 3], **REF, ls="--", label=r"$H_s/M_s$ OOMMF")
    ax.plot(h[:, 0], h[:, 1], "o", ms=4, mfc="none", mec="blue", label=r"$H_c/M_s$ JuliaMag")
    ax.plot(h[:, 0], h[:, 2], "s", ms=4, mfc="none", mec="blue", label=r"$H_s/M_s$ JuliaMag")
    ax.set_xlabel(r"$d/\ell_{\mathrm{ex}}$"); ax.set_ylabel(r"$H/M_s$"); ax.legend(fontsize=7)
    save(fig, FIG / "stdproblem2_hysteresis.png", EX / "stdproblem2_hysteresis.png")


def _cols(path, cols):
    out = []
    for l in Path(path).read_text(encoding="utf-8").splitlines():
        v = l.split()
        if not v or l.lstrip().startswith("#"):
            continue
        try:
            out.append([float(v[c]) for c in cols])
        except (ValueError, IndexError):
            pass
    return np.array(out)


def std4():
    j = np.loadtxt(EX / "stdproblem4.txt", comments="#")
    mm = _cols(EX / "stdproblem4_mumax3.txt", (0, 1, 2, 3))
    oo = _cols(EX / "stdprob4a_oommf.odt", (18, 14, 15, 16)); oo = oo[oo[:, 0] <= 1e-9 + 1e-15]
    fig, ax = plt.subplots(figsize=(4.6, 3.2))
    for c, nm in ((1, "x"), (2, "y"), (3, "z")):
        ax.plot(mm[:, 0] * 1e9, mm[:, c], **REF, label="mumax3" if c == 1 else None)
        ax.plot(oo[::4, 0] * 1e9, oo[::4, c], "^", ms=3, mfc="none", mec="darkorange", mew=0.7,
                label="OOMMF" if c == 1 else None)
        ax.plot(j[:, 0] * 1e9, j[:, c], **JM, label="JuliaMag" if c == 1 else None)
        ax.annotate(rf"$\langle m_{nm}\rangle$", (j[-1, 0] * 1e9, j[-1, c]), xytext=(3, 0),
                    textcoords="offset points", va="center", fontsize=8)
    ax.set_xlabel("t (ns)"); ax.set_ylabel(r"$\langle m\rangle$"); ax.set_xlim(0, 1.08)
    ax.legend(fontsize=7, loc="lower left")
    save(fig, FIG / "stdproblem4_compare.png", EX / "stdproblem4_compare.png")

    fig, ax = plt.subplots(figsize=(4.6, 3.0))
    for c, col, nm in ((1, "tab:red", "x"), (2, "tab:green", "y"), (3, "tab:blue", "z")):
        ax.plot(j[:, 0] * 1e9, j[:, c], color=col, lw=1.2, label=rf"$\langle m_{nm}\rangle$")
    ax.axhline(0, color="gray", lw=0.6, ls="--")
    ax.set_xlabel("t (ns)"); ax.set_ylabel(r"$\langle m\rangle$"); ax.legend(fontsize=7)
    ax.set_title("µMAG standard problem 4 (field 1), JuliaMag", fontsize=9)
    save(fig, EX / "stdproblem4.png")

    g = ROOT / "gpu_validation" / "std4" / "stdproblem4_gpu.txt"
    if g.exists():
        gg = np.loadtxt(g, comments="#"); n = min(len(gg), len(j))
        fig, (a1, a2) = plt.subplots(2, 1, figsize=(4.6, 4.2), sharex=True)
        for c, nm in ((1, "x"), (2, "y"), (3, "z")):
            a1.plot(j[:n, 0] * 1e9, j[:n, c], color="red", lw=2, alpha=0.5, label="CPU" if c == 1 else None)
            a1.plot(gg[:n, 0] * 1e9, gg[:n, c], **JM, label="GPU" if c == 1 else None)
            a2.plot(j[:n, 0] * 1e9, gg[:n, c] - j[:n, c], lw=1, label=rf"$m_{nm}$")
        a1.set_ylabel(r"$\langle m\rangle$"); a1.legend(fontsize=7)
        a2.set_ylabel("GPU − CPU"); a2.set_xlabel("t (ns)"); a2.legend(fontsize=7)
        save(fig, ROOT / "gpu_validation" / "std4" / "stdproblem4_cpu_vs_gpu.png")


def _ovf_mz(path):
    hdr = {}
    for l in Path(path).read_text(encoding="utf-8").splitlines()[:40]:
        if ":" in l and l.startswith("#"):
            k, v = l[1:].split(":", 1); hdr[k.strip()] = v.strip()
    nx, ny = int(hdr["xnodes"]), int(hdr["ynodes"])
    d = np.loadtxt(path, comments="#")
    return d[:, 2].reshape(-1, ny, nx)[0], float(hdr["xstepsize"]), float(hdr["ystepsize"])


def skyrmion():
    d = np.loadtxt(EX / "skyrmion_track.txt", comments="#")
    L = 150 * 2e-9
    t, x = d[:, 0], d[:, 4].copy()
    for i in range(1, len(x)):
        dx = x[i] - x[i - 1]
        if dx > L / 2:
            x[i:] -= L
        elif dx < -L / 2:
            x[i:] += L
    tt, xx = t - t[0], x - x[0]
    v = np.sum(tt * xx) / np.sum(tt**2)
    fig = plt.figure(figsize=(7.2, 2.5), layout="constrained")
    gs = fig.add_gridspec(1, 3, width_ratios=[1.0, 1.25, 1.25])
    a = fig.add_subplot(gs[0])
    a.plot(t * 1e9, x * 1e9, **JM)
    a.set_xlabel("t (ns)"); a.set_ylabel("x, unwrapped (nm)")
    a.set_title(f"(a)  $v_x$ = {v:.0f} m/s", fontsize=9)
    for k, (fn, lab) in enumerate((("skyrmion_00.ovf", "(b)  t = 0"), ("skyrmion_03.ovf", "(c)  t = 3 ns"))):
        mz, cx, cy = _ovf_mz(EX / fn)
        ny, nx = mz.shape
        ax = fig.add_subplot(gs[k + 1])
        im = ax.imshow(mz, origin="lower", cmap="RdBu_r", vmin=-1, vmax=1,
                       extent=[-nx * cx / 2e-9, nx * cx / 2e-9, -ny * cy / 2e-9, ny * cy / 2e-9])
        ax.set_title(lab, fontsize=9); ax.set_xlabel("x (nm)"); ax.grid(False)
        ax.set_ylabel("y (nm)" if k == 0 else "")
    fig.colorbar(im, ax=fig.axes[1:], shrink=0.7, pad=0.02, label="$m_z$")
    save(fig, FIG / "skyrmion_drive.png", EX / "skyrmion_drive.png")


def stage2():
    lv = np.loadtxt(EX / "stage2_langevin.txt", comments="#")
    xi = np.linspace(0.3, 22, 300)
    fig, ax = plt.subplots(figsize=(4.4, 3.0))
    ax.plot(xi, 1 / np.tanh(xi) - 1 / xi, **REF, label=r"Langevin $L(\xi)$")
    ax.plot(lv[:, 0], lv[:, 2], "o", ms=5, mfc="none", mec="blue", label="JuliaMag, 4096 macrospins")
    ax.set_xlabel(r"$\xi=\mu_0 M_s V H/k_BT$"); ax.set_ylabel(r"$\langle m_x\rangle$"); ax.legend(fontsize=7)
    save(fig, EX / "stage2_langevin.png")
    dp = np.loadtxt(EX / "stage2_depinning.txt", comments="#")
    fig, ax = plt.subplots(figsize=(4.4, 3.0))
    ax.plot(dp[:, 0], dp[:, 1], "o-", ms=2, **JM)
    ax.set_xlabel(r"$\mu_0 H_y$ (T)"); ax.set_ylabel(r"$\langle m_y\rangle$")
    ax.set_title("Domain-wall depinning at a soft/hard interface", fontsize=9)
    save(fig, EX / "stage2_depinning.png")
    sp = np.loadtxt(EX / "stage2_dmi_spiral.txt", comments="#")
    fig, ax = plt.subplots(figsize=(4.4, 3.0))
    for c, nm in ((1, "x"), (2, "y"), (3, "z")):
        ax.plot(sp[:, 0], sp[:, c], lw=1, label=f"$m_{nm}$")
    ax.set_xlabel("x (nm)"); ax.set_ylabel("m"); ax.legend(fontsize=7)
    ax.set_title("Bulk-DMI helix, " + (EX / "stage2_dmi_spiral.txt").read_text(encoding="utf-8").splitlines()[0].split("measured")[1].strip(), fontsize=8)
    save(fig, EX / "stage2_dmi_spiral.png")


def thermal():
    d = np.loadtxt(EX / "thermal_demagnetization.txt", comments="#")
    x = np.linspace(0, 1, 200001)
    T = np.linspace(20, 520, 200); KuV = d[1, 3] * d[1, 0]          # σ·T is constant
    th = [np.trapezoid(x * np.exp(KuV / t * (x**2 - 1)), x) / np.trapezoid(np.exp(KuV / t * (x**2 - 1)), x) for t in T]
    fig, ax = plt.subplots(figsize=(4.4, 3.0))
    ax.plot(T, th, **REF, label="Boltzmann, one easy-axis well")
    ax.errorbar(d[:, 0], d[:, 1], yerr=d[:, 2] / np.sqrt(11), fmt="o", ms=4, mfc="none", color="blue",
                lw=1, label="JuliaMag (time average)")
    ax.set_xlabel("T (K)"); ax.set_ylabel(r"$\langle m_z\rangle$"); ax.legend(fontsize=7)
    save(fig, EX / "thermal_demagnetization.png")


def dischyst():
    """Vortex loop: the deterministic run (examples/disc_hysteresis.txt) and two runs
    with a seeded 1e-6 random perturbation before every relaxation."""
    det = np.loadtxt(EX / "disc_hysteresis.txt", comments="#")
    D = ROOT / "gpu_validation" / "revalidation" / "disc_hysteresis"
    fig, ax = plt.subplots(figsize=(4.4, 3.3))
    for k, (fn, col) in enumerate((("loop_noise1e-6_seed1.txt", "darkorange"),
                                   ("loop_noise1e-6_seed2.txt", "green"))):
        p = D / fn
        if p.exists():
            a = np.loadtxt(p, comments="#")
            ax.plot(a[:, 4] * 1e3, a[:, 1], color=col, lw=0.9, ls="--", alpha=0.8,
                    label=rf"perturbed $10^{{-6}}$, seed {k+1}")
    ax.plot(det[:, 4] * 1e3, det[:, 1], "o-", color="blue", lw=1, ms=2, label="deterministic")
    ax.axhline(0, color="0.6", lw=0.5); ax.axvline(0, color="0.6", lw=0.5)
    ax.set_xlabel(r"$\mu_0H_x$ (mT)"); ax.set_ylabel(r"$\langle m_x\rangle$")
    ax.set_ylim(-1.05, 1.05); ax.legend(fontsize=7, loc="lower right")
    save(fig, FIG / "vortex_hysteresis.png", EX / "disc_hysteresis.png")


if __name__ == "__main__":
    import sys
    for name in (sys.argv[1:] or ["std2", "std4", "skyrmion", "stage2", "thermal", "dischyst"]):
        try:
            globals()[name]()
        except FileNotFoundError as e:
            print(f"[{name}] skipped: {e}")
