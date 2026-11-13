"""Why the maxTorque columns of the 40 ns disc series part after ~13 ns.

(a) maxTorque over the run, JuliaMag (RK4, dt 5e-14 s) vs mumax3 (default adaptive
    RK45, MaxErr 1e-5). The two agree to 0.5 % until ~12.5 ns; then mumax3 settles
    on a plateau while JuliaMag keeps decreasing.
(b) Restart from the mumax3 state at 17 ns (stt_es4_noZL_snap.out/m000001.ovf):
    mumax3 with its default solver keeps the plateau; mumax3 with MaxErr 1e-7 or with
    fixed-step RK4 (dt 5e-14 s) drops within ~2 ps to the JuliaMag value, and so does
    JuliaMag restarted from the same state. The plateau is the grid-scale error that
    the adaptive RK45 keeps in the stiff exchange modes at its tolerance, not physics.

Usage: python plot_disk40_maxtorque.py [TABLES_DIR] [out.png]
"""
import sys, os
import numpy as np
import matplotlib.pyplot as plt

sys.stdout.reconfigure(encoding="utf-8")
HERE = os.path.dirname(os.path.abspath(__file__))
TD = sys.argv[1] if len(sys.argv) > 1 else os.path.join(HERE, "..", "gpu_validation", "disk_stt", "tables")
OUT = sys.argv[2] if len(sys.argv) > 2 else os.path.join(HERE, "disk_40ns_es4_maxtorque.png")
L = lambda f: np.loadtxt(os.path.join(TD, f), comments="#")

jm = L("jm_es4_40ns.txt")                      # t mx my mz maxTorque ...
mu = L("mumax_stt_es4_noZL_40ns.txt")          # t mx my mz maxTorque ...
rs = {v: L(f"mumax_restart17_{v}.txt") for v in ("default", "maxerr1e-7", "rk4fixed")}
rj = L("jm_restart17.txt")                     # t core body cell mx my mz
rj_max = np.maximum(rj[:, 1], rj[:, 2])

for v, d in rs.items():
    s = d[:, 0] >= 50e-12
    print(f"mumax3 {v:11s} 50–200 ps: maxTorque {d[s, 4].mean():.4e} T")
s = rj[:, 0] >= 50e-12
print(f"JuliaMag           50–200 ps: maxTorque {rj_max[s].mean():.4e} T")

MU = dict(color="red", lw=2, alpha=0.5)
JMs = dict(color="blue", lw=1)
plt.rcParams.update({"font.size": 9})
fig, ax = plt.subplots(1, 2, figsize=(11, 3.8))
a = ax[0]
a.plot(mu[:, 0] * 1e9, mu[:, 4] * 1e3, **MU, label="mumax$^3$ (RK45, MaxErr $10^{-5}$)")
a.plot(jm[:, 0] * 1e9, jm[:, 4] * 1e3, **JMs, label="JuliaMag (RK4, 50 fs)")
a.set_xlabel("t (ns)"); a.set_ylabel("maxTorque (mT)"); a.set_ylim(0, None)
a.set_title("(a) 40 ns series", fontsize=9); a.legend(fontsize=8)
a = ax[1]
ps = 1e12
a.plot(rs["default"][:, 0] * ps, rs["default"][:, 4] * 1e3, **MU, label="mumax$^3$ RK45, MaxErr $10^{-5}$")
a.plot(rs["maxerr1e-7"][:, 0] * ps, rs["maxerr1e-7"][:, 4] * 1e3, color="darkorange", lw=1.5,
       label="mumax$^3$ RK45, MaxErr $10^{-7}$")
a.plot(rs["rk4fixed"][:, 0] * ps, rs["rk4fixed"][:, 4] * 1e3, color="green", lw=1, ls="--",
       label="mumax$^3$ RK4, 50 fs")
a.plot(rj[:, 0] * ps, rj_max * 1e3, **JMs, label="JuliaMag RK4, 50 fs")
a.set_xlabel("time after restart at 17 ns (ps)"); a.set_ylabel("maxTorque (mT)")
a.set_title("(b) restart from the mumax$^3$ state at 17 ns", fontsize=9); a.legend(fontsize=8)
for a in ax:
    a.grid(alpha=0.3)
fig.tight_layout()
fig.savefig(OUT, dpi=200)
print("wrote", os.path.abspath(OUT))
