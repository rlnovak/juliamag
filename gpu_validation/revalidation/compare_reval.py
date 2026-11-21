"""Re-validation 2026-11: compare each re-run with its previous result and its references.

For every problem, prints (1) the new result against the external reference (OOMMF,
mumax3, or theory) and (2) the new result against the previous JuliaMag result,
read from git at the commit before the re-validation (BASE). Run from the repo root:

    python gpu_validation/revalidation/compare_reval.py [section ...]

Sections: std2 std2hyst std4 bench thermal stage2 dischyst skyrmion disk (default: all
whose new tables exist).
"""
import io
import subprocess
import sys
from pathlib import Path

import numpy as np

sys.stdout.reconfigure(encoding="utf-8")
ROOT = Path(__file__).resolve().parents[2]
BASE = "36a619c"            # last commit before the re-validation
EX = ROOT / "examples"
BEN = ROOT / "gpu_validation" / "benchmarks"
DISK = ROOT / "gpu_validation" / "disk_stt" / "tables"


def old(path, commit=BASE):
    """Text of `path` (repo-relative) at `commit`."""
    return subprocess.run(["git", "show", f"{commit}:{path}"], cwd=ROOT, check=True,
                          capture_output=True).stdout.decode("utf-8")


def load(src):
    if isinstance(src, Path):
        return np.loadtxt(src, comments="#")
    return np.loadtxt(io.StringIO(src), comments="#")


def head(s):
    print("\n" + "=" * 78 + "\n" + s + "\n" + "=" * 78)


def maxdiff(a, b):
    return np.max(np.abs(np.asarray(a) - np.asarray(b)))


def oommf_std2():
    rows = []
    for line in (EX / "stdprob2_oommf.txt").read_text(encoding="utf-8").splitlines():
        v = line.split()
        if v and v[0].replace(".", "", 1).isdigit():
            rows.append([float(x) for x in v[:6]])
    return np.array(rows)          # d/lex, cell, Hc, Hs, mx, my


# --- Standard problem 2 ----------------------------------------------------------
def std2():
    head("Standard problem 2 — remanence (new vs OOMMF, new vs old)")
    new = load(EX / "stdproblem2.txt"); prev = load(old("examples/stdproblem2.txt"))
    o = oommf_std2()
    print(" d/lex   mx_new    my_new    | mx_OOMMF  my_OOMMF | Δmx_ref   Δmy_ref  | Δ vs old")
    for r in new:
        k = np.where(np.isclose(o[:, 0], r[0]))[0]
        ref = o[k[0]] if len(k) else None
        p = prev[np.isclose(prev[:, 0], r[0])][0]
        refs = f"{ref[4]:.4f}    {ref[5]:.4f}  | {r[1]-ref[4]:+.4f}   {r[2]-ref[5]:+.4f}" if ref is not None else " " * 38
        print(f" {r[0]:5.1f}   {r[1]:.6f}  {r[2]:.6f}  | {refs} | {maxdiff(r[1:], p[1:]):.1e}")


def std2hyst():
    head("Standard problem 2 — coercive / switching field (new vs OOMMF, new vs old)")
    new = load(EX / "stdproblem2_hysteresis.txt"); prev = load(old("examples/stdproblem2_hysteresis.txt"))
    o = oommf_std2()
    print(" d/lex   Hc/Ms     Hs/Ms    | Hc_OOMMF  Hs_OOMMF | ΔHc      ΔHs     | Δ vs old")
    for r in new:
        ref = o[np.isclose(o[:, 0], r[0])][0]
        p = prev[np.isclose(prev[:, 0], r[0])][0]
        print(f" {r[0]:5.1f}   {r[1]:.5f}  {r[2]:.5f}  | {ref[2]:.5f}   {ref[3]:.5f}  | "
              f"{r[1]-ref[2]:+.5f} {r[2]-ref[3]:+.5f} | {maxdiff(r[1:], p[1:]):.1e}")
    print(" (field step dH/Ms = 0.001: Hs is resolved to ±0.0005; Hc is interpolated)")


# --- Standard problem 4 ----------------------------------------------------------
def _read_cols(path, cols):
    rows = []
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        s = line.strip()
        if not s or s.startswith("#"):
            continue
        v = s.split()
        try:
            rows.append([float(v[c]) for c in cols])
        except (ValueError, IndexError):
            continue
    return np.array(rows)


def std4():
    head("Standard problem 4 — ⟨m⟩(t), 0–1 ns (new vs mumax3 / OOMMF, new vs old)")
    j = load(EX / "stdproblem4.txt")
    mm = _read_cols(EX / "stdproblem4_mumax3.txt", (0, 1, 2, 3))
    oo = _read_cols(EX / "stdprob4a_oommf.odt", (18, 14, 15, 16))
    print(f" relaxed S-state ⟨m⟩ = ({j[0,1]:.4f}, {j[0,2]:.4f}, {j[0,3]:.4f})")
    for name, ref in (("mumax3", mm), ("OOMMF", oo)):
        sel = ref[:, 0] <= j[-1, 0] + 1e-15
        r = ref[sel]
        line = []
        for c, nm in ((1, "mx"), (2, "my"), (3, "mz")):
            d = np.interp(r[:, 0], j[:, 0], j[:, c]) - r[:, c]
            line.append(f"{nm}: max {np.max(np.abs(d)):.4f} rms {np.sqrt(np.mean(d**2)):.4f}")
        print(f" vs {name:7s} " + " | ".join(line))
    p = load(old("examples/stdproblem4.txt"))
    n = min(len(p), len(j))
    print(f" vs old JuliaMag table (same 2.5 ps grid): max |Δ⟨m⟩| = {maxdiff(j[:n,1:4], p[:n,1:4]):.2e}")
    g = ROOT / "gpu_validation" / "std4" / "stdproblem4_gpu.txt"
    if g.exists():
        gg = load(g); n = min(len(gg), len(j))
        print(f" GPU vs CPU (new tables):                 max |Δ⟨m⟩| = {maxdiff(gg[:n,1:4], j[:n,1:4]):.2e}")


def bench():
    head("Std 4 / Std 5 column-by-column vs mumax3 (rms diff / rms scale), new and old")
    cols = ["mx", "my", "mz", "E_demag", "E_exch", "E_Zeeman", "E_total"]
    for nm in ("std4", "std5"):
        M = load(BEN / f"{nm}_mumax_table.txt")
        for label, J in (("new", load(BEN / f"{nm}_juliamag_table.txt")),
                         ("old", load(old(f"gpu_validation/benchmarks/{nm}_juliamag_table.txt")))):
            out = []
            for k, c in enumerate(cols, start=1):
                d = np.interp(M[:, 0], J[:, 0], J[:, k]) - M[:, k]
                sc = np.sqrt(np.mean(M[:, k] ** 2))
                out.append(f"{c} {100*np.sqrt(np.mean(d**2))/sc if sc else 0:.3f}%")
            print(f" {nm} {label}: " + "  ".join(out))
        Jn = load(BEN / f"{nm}_juliamag_table.txt"); Jo = load(old(f"gpu_validation/benchmarks/{nm}_juliamag_table.txt"))
        n = min(len(Jn), len(Jo))
        rel = np.max(np.abs(Jn[:n, 4:8] - Jo[:n, 4:8]) / np.max(np.abs(Jo[:n, 4:8]), axis=0))
        print(f" {nm} new vs old: max |Δ⟨m⟩| = {maxdiff(Jn[:n,1:4], Jo[:n,1:4]):.2e}, "
              f"max rel. Δ energies = {rel:.2e}")
        print(f" {nm} ⟨m⟩(1 ns): new ({Jn[-1,1]:.5f}, {Jn[-1,2]:.5f}, {Jn[-1,3]:.5f})  "
              f"mumax3 ({M[-1,1]:.5f}, {M[-1,2]:.5f}, {M[-1,3]:.5f})")


# --- Thermal demagnetization: Boltzmann statistics in one easy-axis well ---------
def thermal():
    head("Thermal nanodot — time-averaged ⟨mz⟩ vs Boltzmann statistics in the upper well")
    d = load(EX / "thermal_demagnetization.txt")
    x = np.linspace(0, 1, 200001)
    print(" T[K]   σ=KuV/kBT   ⟨mz⟩_sim ± std/√N_ind   ⟨mz⟩_theory   Δ/σ_err")
    # correlation time of mz near the minimum: 1/(α γ' B_K), B_K = 2Ku/Ms = 1 T
    tau = 1 / (0.5 * 1.7595e11 / 1.25 * 1.0)
    n_ind = 2000 * 40 * 2e-15 / tau
    for T, mz, sd, sig in d:
        if T == 0:
            continue
        w = np.exp(sig * (x**2 - 1))
        th = np.trapezoid(x * w, x) / np.trapezoid(w, x)
        err = sd / np.sqrt(n_ind)
        print(f" {T:5.0f}  {sig:8.2f}    {mz:.4f} ± {err:.4f}           {th:.4f}        {(mz-th)/err:+.1f}")
    print(f" (N_ind ≈ {n_ind:.0f} independent samples per temperature: 160 ps of sampling, τ ≈ {tau*1e12:.0f} ps;"
          " above ~200 K the two-well hopping makes the single-well average a lower bound only)")


# --- magnum.np stage 2 -------------------------------------------------------------
def stage2():
    head("magnum.np stage-2 problems (new vs theory, new vs old)")
    lv = load(EX / "stage2_langevin.txt"); lo = load(old("examples/stage2_langevin.txt"))
    print(" Langevin, ensemble of 4096 macrospins: ξ, ⟨mx⟩_new, ⟨mx⟩_old, L(ξ)")
    for r, q in zip(lv, lo):
        print(f"   ξ={r[0]:5.1f}  {r[2]:.4f}  {q[2]:.4f}  L={r[3]:.4f}  (new−L = {r[2]-r[3]:+.4f})")
    dp = load(EX / "stage2_depinning.txt"); do = load(old("examples/stage2_depinning.txt"))
    jn = dp[np.argmax(np.diff(dp[:, 1])), 0]; jo = do[np.argmax(np.diff(do[:, 1])), 0]
    # Analytical depinning field of a wall at a soft/hard interface (magnum.np demo,
    # after Paul, J. Appl. Phys. 53, 1649 (1982)): Hp = 2K1/(μ0 M1) (1 − εk εA) / (1 + sqrt(εm εA))²,
    # with εk = K2/K1, εA = A2/A1, εm = M2/M1 (1 = hard, 2 = soft); here as μ0 Hp [T].
    mu0 = 4e-7 * np.pi
    M1, K1, A1 = 1.0 / mu0, 1e6, 1.0e-11
    M2, K2, A2 = 0.25 / mu0, 1e5, 0.25e-11
    ek, eA, em = K2 / K1, A2 / A1, M2 / M1
    Hp = 2 * K1 / M1 * (1 - ek * eA) / (1 + np.sqrt(em * eA)) ** 2
    print(f" Depinning: field at the largest ⟨my⟩ jump: new {jn:.4f} T, old {jo:.4f} T, "
          f"analytical μ0Hp = {Hp:.4f} T (field step 0.005 T); max |Δ⟨my⟩| new vs old = "
          f"{maxdiff(dp[:,1:], do[:,1:]):.1e}")
    sp = (EX / "stage2_dmi_spiral.txt").read_text(encoding="utf-8").splitlines()[0]
    spo = old("examples/stage2_dmi_spiral.txt").splitlines()[0]
    print(f" Bulk-DMI helix: new  {sp[2:]}\n                 old  {spo[2:]}")
    s1, s0 = load(EX / "stage2_dmi_spiral.txt"), load(old("examples/stage2_dmi_spiral.txt"))
    print(f"   max |Δm(x)| new vs old = {maxdiff(s1[:,1:], s0[:,1:]):.1e}")


# --- Vortex hysteresis in a disc ---------------------------------------------------
def dischyst():
    head("Vortex hysteresis in a 500 nm × 20 nm disc (new vs old)")
    n = load(EX / "disc_hysteresis.txt"); p = load(old("examples/disc_hysteresis.txt"))
    print(f" {len(n)} field steps; max |Δ⟨m⟩| new vs old = {maxdiff(n[:,1:4], p[:,1:4]):.2e}")
    B, mx = n[:, 4], n[:, 1]
    up = np.arange(len(B)) <= np.argmax(B)
    jumps = np.argsort(np.abs(np.diff(mx)))[::-1][:2]
    for k in sorted(jumps):
        print(f"   jump in ⟨mx⟩ between {B[k]*1e3:+.0f} and {B[k+1]*1e3:+.0f} mT: "
              f"{mx[k]:.3f} → {mx[k+1]:.3f}")
    i150 = np.argmax(B)
    i0 = np.where((np.abs(B) < 1e-12) & (np.arange(len(B)) > i150))[0][0]
    print(f"   ⟨mx⟩ at +150 mT = {mx[i150]:.3f}; remanence on the return sweep = {mx[i0]:.3f}")


# --- Skyrmion on a periodic wire ---------------------------------------------------
def skyrmion():
    head("Current-driven skyrmion on a periodic wire (new vs old)")
    L = 150 * 2e-9

    def vel(d):
        t, x = d[:, 0], d[:, 4].copy()
        for i in range(1, len(x)):
            dx = x[i] - x[i - 1]
            if dx > L / 2:
                x[i:] -= L
            elif dx < -L / 2:
                x[i:] += L
        tt, xx = t - t[0], x - x[0]
        y = d[:, 5]
        return np.sum(tt * xx) / np.sum(tt**2), y[-1] - y[0], d[:, 7]
    n = load(EX / "skyrmion_track.txt"); p = load(old("examples/skyrmion_track.txt"))
    vn, dyn, qn = vel(n); vo, dyo, qo = vel(p)
    m = min(len(n), len(p))
    print(f" v_x: new {vn:.1f} m/s, old {vo:.1f} m/s; Δy over 3 ns: new {dyn*1e9:.1f} nm, old {dyo*1e9:.1f} nm")
    print(f" Q: new {qn.min():.3f}…{qn.max():.3f}; max |Δ position| new vs old = "
          f"{maxdiff(n[:m,4:6], p[:m,4:6])*1e9:.3f} nm (wrap-aware not needed below L/2)")


# --- Disc STT (GPU tables) ---------------------------------------------------------
def disk():
    head("Current-driven vortex in a disc (GPU tables)")
    sys.path.insert(0, str(ROOT / "gpu_validation" / "disk_stt"))
    import frequencies  # noqa: F401  (prints the like-for-like frequency table)


SECTIONS = dict(std2=std2, std2hyst=std2hyst, std4=std4, bench=bench, thermal=thermal,
                stage2=stage2, dischyst=dischyst, skyrmion=skyrmion, disk=disk)

if __name__ == "__main__":
    for s in (sys.argv[1:] or list(SECTIONS)):
        try:
            SECTIONS[s]()
        except FileNotFoundError as e:
            print(f"\n[{s}] skipped: {e}")
