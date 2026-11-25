"""Write the mumax3 scripts of the DMI and anisotropy checks (one per case) to the directory given.

Each saves m (random), geom, regions, B_exch (exchange + DMI) and B_anis of the initial
state, with E_exch and E_anis in the table. NAME.params has the parameters of every region,
one per line; compare_mumax3.jl reads it and evaluates the same states in JuliaMag.
"""
import sys
from pathlib import Path

OUT = Path(sys.argv[1])
OUT.mkdir(parents=True, exist_ok=True)

BASE = {"Msat": 8e5, "Aex": 1.3e-11, "alpha": 0.02}
IND, BULK = {"Dind": 2.5e-3}, {"Dbulk": 1.5e-3}
ANI = {"Ku": 5e5, "anisU": (0.3, 0.2, 1.0)}
BOX, FILM, DISC = (12, 10, 3), (12, 10, 1), (16, 16, 2)
O = (0, 0, 0)
CIRCLE = "setgeom(circle(50e-9))\n"
XPOS = "xrange(0, inf)"
# name: (size, PBC, OpenBC, geometry lines, region 0, {region: (shape, parameters)})
CASES = {
    "ind_open":        (BOX, O, False, "", IND, {}),
    "ind_openbc":      (BOX, O, True, "", IND, {}),
    "ind_pbcx":        (BOX, (1, 0, 0), False, "", IND, {}),
    "ind_pbcxy":       (BOX, (1, 1, 0), False, "", IND, {}),
    "bulk_open":       (BOX, O, False, "", BULK, {}),
    "bulk_openbc":     (BOX, O, True, "", BULK, {}),
    "bulk_nz1":        (FILM, O, False, "", BULK, {}),
    "bulk_nz1_openbc": (FILM, O, True, "", BULK, {}),
    "bulk_pbcyz":      (BOX, (0, 1, 1), False, "", BULK, {}),
    "ind_disc":        (DISC, O, False, CIRCLE, IND, {}),
    "ind_disc_openbc": (DISC, O, True, CIRCLE, IND, {}),
    "bulk_sphere":     ((16, 12, 14), O, False, "setgeom(ellipsoid(44e-9, 44e-9, 26e-9))\n", BULK, {}),
    "ind_smooth":      (DISC, O, False, "EdgeSmooth = 4\n" + CIRCLE, IND, {}),
    "bulk_smooth":     (DISC, O, False, "EdgeSmooth = 4\n" + CIRCLE, BULK, {}),
    # two regions: A and Msat differ, D does not
    "ind_regA":        (BOX, O, False, "", IND, {1: (XPOS, {"Msat": 1.2e6, "Aex": 2e-11})}),
    # D differs between regions: same sign, opposite sign, equal and opposite, zero
    "ind_regD":        (BOX, O, False, "", IND, {1: (XPOS, {"Dind": 1e-3})}),
    "ind_regDsign":    (BOX, O, False, "", IND, {1: (XPOS, {"Dind": -1e-3})}),
    "ind_regDopp":     (BOX, (1, 0, 0), False, "", IND, {1: (XPOS, {"Dind": -2.5e-3})}),
    "ind_regDy":       (BOX, O, False, "", IND, {1: ("yrange(0, inf)", {"Dind": -1e-3, "Aex": 2e-11})}),
    "ind_regD0":       (BOX, O, False, "", IND, {1: (XPOS, {"Dind": 0.0})}),
    "ind_regD_openbc": (BOX, (0, 1, 0), True, "", IND, {1: (XPOS, {"Dind": -1e-3, "Msat": 6e5})}),
    "bulk_regD":       (BOX, O, False, "", BULK, {1: (XPOS, {"Dbulk": 0.6e-3}),
                                                  2: ("zrange(0, inf)", {"Dbulk": -0.9e-3, "Aex": 2e-11, "Msat": 1.2e6})}),
    "bulk_regD_pbc":   (BOX, (1, 1, 1), False, "", BULK, {1: ("yrange(0, inf)", {"Dbulk": -0.6e-3, "Msat": 6e5})}),
    "bulk_regD_openbc": (BOX, O, True, "", BULK, {1: ("yrange(0, inf)", {"Dbulk": -0.6e-3, "Msat": 6e5})}),
    "ind_regDsmooth":  (DISC, O, False, "EdgeSmooth = 4\n" + CIRCLE, IND, {1: (XPOS, {"Dind": -1e-3, "Msat": 1.2e6, "Aex": 2e-11})}),
    # uniaxial anisotropy, no DMI
    "ani_open":        (BOX, O, False, "", ANI, {}),
    "ani_smooth":      (DISC, O, False, "EdgeSmooth = 4\n" + CIRCLE, ANI, {}),
    "ani_regsmooth":   (DISC, O, False, "EdgeSmooth = 4\n" + CIRCLE, ANI,
                        {1: (XPOS, {"Ku": -2e5, "Msat": 1.2e6, "anisU": (0, 1, 0)})}),
}
MX3 = {"Ku": "Ku1"}


def val(v):
    return "vector({}, {}, {})".format(*v) if isinstance(v, tuple) else repr(v)


for name, (size, pbc, obc, geom, reg0, regions) in CASES.items():
    s = f"SetGridsize{size}\nSetCellsize(3e-9, 4e-9, 2e-9)\n"
    if any(pbc):
        s += f"SetPBC{pbc}\n"
    s += "".join(f"{MX3.get(k, k)} = {val(v)}\n" for k, v in {**BASE, **reg0}.items())
    s += f"OpenBC = {'true' if obc else 'false'}\n" + geom
    for r, (shape, par) in regions.items():
        s += f"defregion({r}, {shape})\n"
        s += "".join(f"{MX3.get(k, k)}.setregion({r}, {val(v)})\n" for k, v in par.items())
    s += "EnableDemag = false\nm = RandomMagSeed(7)\n"
    s += "".join(f'saveas({q}, "{q}")\n' for q in ("m", "geom", "regions", "B_exch", "B_anis"))
    s += "tableadd(E_exch)\ntableadd(E_anis)\ntablesave()\n"
    (OUT / f"{name}.mx3").write_text(s, newline="\n")
    p = f"pbc {pbc[0]} {pbc[1]} {pbc[2]}\nopenbc {int(obc)}\n"
    for r, par in {0: {**BASE, **reg0}, **{r: {**BASE, **reg0, **q[1]} for r, q in regions.items()}}.items():
        for k, v in par.items():
            p += f"{r} {k} " + (" ".join(map(str, v)) if isinstance(v, tuple) else repr(v)) + "\n"
    (OUT / f"{name}.params").write_text(p, newline="\n")
print(len(CASES), "scripts in", OUT)
