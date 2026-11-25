"""Write the mumax3 scripts of the DMI boundary check (one per case) to the directory given.

Each saves m (random), geom, regions and B_exch (exchange + DMI) of the initial state, and
E_exch in the table. compare_mumax3.jl evaluates the same states in JuliaMag.
"""
import sys
from pathlib import Path

OUT = Path(sys.argv[1])
OUT.mkdir(parents=True, exist_ok=True)

BASE = "Msat = 8e5\nAex = 1.3e-11\nalpha = 0.02\n"
IND, BULK = "Dind = 2.5e-3\n", "Dbulk = 1.5e-3\n"
# name: (Nx, Ny, Nz, PBC, OpenBC, material and geometry lines)
CASES = {
    "ind_open":        (12, 10, 3, (0, 0, 0), False, IND),
    "ind_openbc":      (12, 10, 3, (0, 0, 0), True, IND),
    "ind_pbcx":        (12, 10, 3, (1, 0, 0), False, IND),
    "ind_pbcxy":       (12, 10, 3, (1, 1, 0), False, IND),
    "bulk_open":       (12, 10, 3, (0, 0, 0), False, BULK),
    "bulk_openbc":     (12, 10, 3, (0, 0, 0), True, BULK),
    "bulk_nz1":        (12, 10, 1, (0, 0, 0), False, BULK),
    "bulk_nz1_openbc": (12, 10, 1, (0, 0, 0), True, BULK),
    "bulk_pbcyz":      (12, 10, 3, (0, 1, 1), False, BULK),
    "ind_disc":        (16, 16, 2, (0, 0, 0), False, IND + "setgeom(circle(50e-9))\n"),
    "ind_disc_openbc": (16, 16, 2, (0, 0, 0), True, IND + "setgeom(circle(50e-9))\n"),
    "bulk_sphere":     (16, 12, 14, (0, 0, 0), False, BULK + "setgeom(ellipsoid(44e-9, 44e-9, 26e-9))\n"),
    "ind_smooth":      (16, 16, 2, (0, 0, 0), False, "EdgeSmooth = 4\n" + IND + "setgeom(circle(50e-9))\n"),
    "bulk_smooth":     (16, 16, 2, (0, 0, 0), False, "EdgeSmooth = 4\n" + BULK + "setgeom(circle(50e-9))\n"),
    # two regions: A and Msat differ, D does not
    "ind_regA":        (12, 10, 3, (0, 0, 0), False, IND + "defregion(1, xrange(0, inf))\nMsat.setregion(1, 1.2e6)\nAex.setregion(1, 2e-11)\n"),
    # D differs between regions: mumax3 uses the mean of the bond, JuliaMag the cell's own D
    "ind_regD":        (12, 10, 3, (0, 0, 0), False, IND + "defregion(1, xrange(0, inf))\nDind.setregion(1, 1e-3)\n"),
}

for name, (nx, ny, nz, pbc, obc, body) in CASES.items():
    s = f"SetGridsize({nx}, {ny}, {nz})\nSetCellsize(3e-9, 4e-9, 2e-9)\n"
    if any(pbc):
        s += f"SetPBC({pbc[0]}, {pbc[1]}, {pbc[2]})\n"
    s += BASE + f"OpenBC = {'true' if obc else 'false'}\n" + body
    s += "EnableDemag = false\nm = RandomMagSeed(7)\n"
    s += 'saveas(m, "m")\nsaveas(geom, "geom")\nsaveas(regions, "regions")\nsaveas(B_exch, "B_exch")\n'
    s += "tableadd(E_exch)\ntablesave()\n"
    (OUT / f"{name}.mx3").write_text(s, newline="\n")
print(len(CASES), "scripts in", OUT)
