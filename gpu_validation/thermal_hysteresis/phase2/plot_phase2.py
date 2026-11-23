"""Phase 2 of the finite-temperature vortex hysteresis: 8 loops at 50 K and 8 at 300 K
(dwell 2 ns, Heun dt 1e-13 s, alpha 0.5), against the T = 0 minimizer loops
(deterministic, 1e-6-perturbed seeds 1-2, field tilted by 0.1 and 1 degree).

For each loop and each branch, it finds:
- B_sw: the switching field, i.e. the field step with the largest jump of <mx>;
- the state reached: "vortex" if <mx> then stays in the vortex range for at least
  two field steps before saturating, otherwise "reversed" (direct reversal);
- B_ann: the vortex annihilation field, the first field after the switch at
  which s<mx> < -0.95, with s the sign of the branch's starting magnetization.

On the virgin branch: the annihilation field (first <mx> > 0.95).
The thermal loops use <mx> averaged over the last quarter of each dwell.

Usage: python plot_phase2.py   (prints the tables; writes phase2_loops.png here and
       paper/figures/vortex_hysteresis_thermal.png)
"""
import os
import sys
import glob
import numpy as np
import matplotlib.pyplot as plt

sys.stdout.reconfigure(encoding="utf-8")
HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", "..", ".."))
REV = os.path.join(ROOT, "gpu_validation", "revalidation", "disc_hysteresis")


def load_T0(path):
    a = np.loadtxt(path, comments="#")
    return a[:, 4], a[:, 1]                       # B_x, <mx>


def load_thermal(path):
    a = np.loadtxt(path, comments="#")
    return a[:, 0], a[:, 4]                       # B_x, <mx> (last-quarter mean)


def branches(B):
    """Index ranges of the virgin, descending and ascending branches."""
    imax = int(np.argmax(B[:40]))                 # end of the virgin branch (+150 mT)
    imin = imax + int(np.argmin(B[imax:]))        # -150 mT
    return slice(0, imax + 1), slice(imax, imin + 1), slice(imin, len(B))


def events(B, mx):
    v, d, u = branches(B)
    out = {"virgin_ann": B[v][np.argmax(mx[v] > 0.95)] * 1e3}
    for name, sl, s in (("desc", d, +1), ("asc", u, -1)):
        b, m = B[sl], s * mx[sl]                  # s·mx goes from +1 to -1 on each branch
        jump = m[:-1] - m[1:]
        k = int(np.argmax(jump)) + 1              # first step after the largest jump
        bsw = b[k] * 1e3
        sat = np.nonzero(m[k:] < -0.95)[0]
        kann = k + sat[0] if sat.size else len(m) - 1
        state = "vortex" if kann - k >= 2 else "reversed"
        out[name] = (bsw, state, b[kann] * 1e3 if state == "vortex" else np.nan)
    return out


runs0 = [("T=0 deterministic", os.path.join(ROOT, "examples", "disc_hysteresis.txt")),
         ("T=0 noise 1e-6, seed 1", os.path.join(REV, "loop_noise1e-6_seed1.txt")),
         ("T=0 noise 1e-6, seed 2", os.path.join(REV, "loop_noise1e-6_seed2.txt")),
         ("T=0 tilt 0.1°", os.path.join(REV, "loop_tilt0.1deg.txt")),
         ("T=0 tilt 1°", os.path.join(REV, "loop_tilt1deg.txt"))]
data = {}
for lab, f in runs0:
    data[lab] = load_T0(f)
thermal = {}
for T in (50, 300):
    fs = sorted(glob.glob(os.path.join(HERE, f"loop_T{T}_seed*.txt")),
                key=lambda s: int(s.rsplit("seed", 1)[1][:-4]))
    thermal[T] = [(f"{T} K seed {f.rsplit('seed', 1)[1][:-4]}", load_thermal(f)) for f in fs]

print(f"{'run':26s} {'virgin ann':>10s} | {'desc B_sw':>9s} {'state':>8s} {'B_ann':>6s} | "
      f"{'asc B_sw':>8s} {'state':>8s} {'B_ann':>6s}   [mT]")
EV = {}
for lab, (B, mx) in list(data.items()) + [x for T in (50, 300) for x in thermal[T]]:
    e = events(B, mx); EV[lab] = e
    print(f"{lab:26s} {e['virgin_ann']:10.0f} | {e['desc'][0]:9.0f} {e['desc'][1]:>8s} "
          f"{e['desc'][2]:6.0f} | {e['asc'][0]:8.0f} {e['asc'][1]:>8s} {e['asc'][2]:6.0f}")

print()
for T in (50, 300):
    es = [EV[lab] for lab, _ in thermal[T]]
    va = np.array([e["virgin_ann"] for e in es])
    print(f"{T} K, N = {len(es)}: virgin annihilation {va.mean():.1f} ± {va.std(ddof=1):.1f} mT")
    for br in ("desc", "asc"):
        sw = np.array([e[br][0] for e in es])
        nv = sum(e[br][1] == "vortex" for e in es)
        ann = np.array([e[br][2] for e in es if e[br][1] == "vortex"])
        sgn = -1 if br == "desc" else 1
        a = (f"; vortex annihilation {sgn*np.mean(sgn*ann):+.1f} ± {ann.std(ddof=1):.1f} mT"
             if ann.size > 1 else "")
        print(f"   {br}: switching {sw.mean():+.1f} ± {sw.std(ddof=1):.1f} mT "
              f"(range {sw.min():+.0f} … {sw.max():+.0f}); vortex {nv}/{len(es)}, "
              f"direct reversal {len(es) - nv}/{len(es)}{a}")

# --- figure ------------------------------------------------------------------
B0, mx0 = data["T=0 deterministic"]
fig, ax = plt.subplots(1, 3, figsize=(15, 4.4))
for a, T in zip(ax[:2], (50, 300)):
    a.plot(B0 * 1e3, mx0, color="red", lw=2, alpha=0.5, label="T = 0 (deterministic)")
    for i, (lab, (B, mx)) in enumerate(thermal[T]):
        a.plot(B * 1e3, mx, color="blue", lw=0.8, alpha=0.55,
               label=f"{T} K, 8 realizations" if i == 0 else None)
    a.set_xlim(-150, 150); a.set_ylim(-1.05, 1.05)
    a.set_xlabel(r"$\mu_0 H_x$ (mT)"); a.set_ylabel(r"$\langle m_x\rangle$")
    a.set_title(f"({'a' if T == 50 else 'b'}) T = {T} K, dwell 2 ns per 5 mT step", fontsize=9)
    a.legend(fontsize=7, loc="lower right"); a.grid(alpha=0.3)
a = ax[2]
rows = [lab for lab, _ in runs0] + [lab for T in (50, 300) for lab, _ in thermal[T]]
for y, lab in enumerate(rows):
    e = EV[lab]
    col = "red" if lab.startswith("T=0") else "blue"
    for br, mk in (("desc", "v"), ("asc", "^")):
        bsw, st, ann = e[br]
        a.plot(bsw, y, mk, color=col, mfc=col if st == "vortex" else "white", ms=6)
        if st == "vortex":
            a.plot([bsw, ann], [y, y], color=col, lw=0.8, alpha=0.6)
    a.plot(e["virgin_ann"], y, "s", color=col, ms=4, alpha=0.7)
a.set_yticks(range(len(rows))); a.set_yticklabels(rows, fontsize=6)
a.invert_yaxis(); a.axvline(0, color="k", lw=0.5)
a.set_xlabel(r"$\mu_0 H_x$ (mT)")
a.set_title("(c) switching (▼ desc., ▲ asc.; filled: vortex, open: direct reversal),\n"
            "line: vortex range, ■ virgin annihilation", fontsize=8)
a.grid(alpha=0.3, axis="x")
fig.tight_layout()
for out in (os.path.join(HERE, "phase2_loops.png"),
            os.path.join(ROOT, "paper", "figures", "vortex_hysteresis_thermal.png")):
    fig.savefig(out, dpi=200)
    print("wrote", out)
