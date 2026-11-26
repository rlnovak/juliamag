# mumax3 against JuliaMag on the four problems of run_fields.sh: tables and a figure.
#   python gpu_validation/fields/compare.py RESULTS FIGURE
# RESULTS has mumax3/CASE.out/table.txt and juliamag/CASE/table.txt. JuliaMag's rows (adaptive steps)
# are interpolated linearly at mumax3's times. Differences relative to the largest |value| of the column.
import sys, os, re
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

CASES = [("pulse", "Gaussian pulse in space and time (vortex)", "mx"),
         ("regions", "field per region, functions of time", "my"),
         ("zhangli_half", "Zhang-Li current in one half", "my"),
         ("nanocontact", "Slonczewski current in a nanocontact", "mx")]


def table(f):
    head = open(f, encoding="utf-8").readline().lstrip("# ").rstrip("\n").split("\t")
    names = [re.sub(r"\s*\(.*\)$", "", h) for h in head]
    return names, np.loadtxt(f, ndmin=2)


def key(n):                                   # one name for the same column in the two codes
    n = n.lower().replace("e_zeeman", "e_zeeman").replace("m.region1", "m_region1")
    return {"ext_coreposx": "vortexx", "ext_coreposy": "vortexy", "ext_coreposz": "vortexw"}.get(n, n)


def main(res, fig):
    fs, axs = plt.subplots(1, len(CASES), figsize=(4.2 * len(CASES), 3.3))
    for (case, title, col), ax in zip(CASES, axs):
        hm, M = table(os.path.join(res, "mumax3", case + ".out", "table.txt"))
        hj, J = table(os.path.join(res, "juliamag", case, "table.txt"))
        kj = {key(n): i for i, n in enumerate(hj)}
        print(f"{case}: rows {len(M)} / {len(J)}, end {M[-1, 0]:.4e} / {J[-1, 0]:.4e} s")
        for c, n in enumerate(hm):
            if c == 0 or key(n) not in kj:
                c == 0 or print(f"   {n}: no such column in JuliaMag")
                continue
            ok = M[:, 0] <= J[-1, 0]
            d = np.interp(M[ok, 0], J[:, 0], J[:, kj[key(n)]]) - M[ok, c]
            s = np.abs(M[:, c]).max()
            if s > 0:
                print(f"   {n:14s} rms {np.sqrt(np.mean(d**2)) / s:.1e}  largest {np.abs(d).max() / s:.1e}  (largest |value| {s:.4g})")
        c = hm.index(col)
        ax.plot(M[:, 0] * 1e9, M[:, c], color="red", lw=2, alpha=0.5, label="mumax3")
        ax.plot(J[:, 0] * 1e9, J[:, kj[col]], color="blue", lw=1, label="JuliaMag")
        ax.set_title(title, fontsize=9); ax.set_xlabel("t (ns)"); ax.set_ylabel("<" + col + ">")
    axs[0].legend(fontsize=8)
    fs.tight_layout(); fs.savefig(fig, dpi=130)


main(*sys.argv[1:3])
