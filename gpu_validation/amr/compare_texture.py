# JuliaMag's AMR conduction against the FEM reference: resistance and potential.
#
#   python gpu_validation/amr/compare_texture.py DIR [DIR ...]  > results/texture.txt
#
# DIR holds the outputs of fem_reference.py (fem_[tag]NXxNY_qQ.txt) and juliamag_texture.jl
# (jm_[tag]NXxNY.txt), tag "" (tilted wall) or "uniform45_". Prints, for each texture: R of JuliaMag on
# each grid and its Richardson extrapolation, R of the FEM on each node grid and its extrapolation, and
# the largest difference of the potentials at the cell centres against the finest FEM solution.
import sys, os, re, glob
import numpy as np


def read(path):
    with open(path) as f:
        head = f.readline()
    R = float(re.search(r"R\s+([-+0-9.eE]+)", head).group(1))
    return R, np.loadtxt(path, comments="#")


def richardson(h, R):
    # last three values: order from their ratio, extrapolated limit
    r = (R[-3] - R[-2]) / (R[-2] - R[-1])
    p = np.log(abs(r)) / np.log(h[-3] / h[-2])
    return p, R[-1] + (R[-1] - R[-2]) / ((h[-2] / h[-1]) ** p - 1)


files = {}
for d in sys.argv[1:]:
    for f in glob.glob(os.path.join(d, "*.txt")):
        files[os.path.basename(f)] = f

for tag, name in (("", "tilted domain wall"), ("uniform45_", "uniform m at 45 degrees")):
    jm = sorted([k for k in files if re.fullmatch(rf"jm_{tag}\d+x\d+\.txt", k)], key=lambda k: int(re.findall(r"(\d+)x", k)[-1]))
    fem = [k for k in files if re.fullmatch(rf"fem_{tag}\d+x\d+_q\d+\.txt", k)]
    if not jm or not fem:
        continue
    print(f"== {name}")
    nx = [int(re.findall(r"(\d+)x", k)[-1]) for k in jm]
    Rj = [read(files[k])[0] for k in jm]
    for n, R in zip(nx, Rj):
        print(f"JuliaMag {n:4d} cells along x: R = {R:.9f} Ohm")
    p, Rinf = richardson(1 / np.array(nx, float), Rj)
    print(f"JuliaMag extrapolated: {Rinf:.9f} Ohm (order {p:.2f})")
    # FEM: the node grids of the finest cell grid that has three refinements, else of each
    groups = {}
    for k in fem:
        g = re.fullmatch(rf"fem_{tag}(\d+)x(\d+)_q(\d+)\.txt", k)
        groups.setdefault((int(g.group(1)), int(g.group(2))), []).append((int(g.group(3)), k))
    Rf_best = None
    for (cx, cy), lst in sorted(groups.items()):
        lst.sort()
        Rf = [read(files[k])[0] for _, k in lst]
        nodes = [q * cx for q, _ in lst]
        for n, R in zip(nodes, Rf):
            print(f"FEM {n:5d} nodes along x: R = {R:.9f} Ohm")
        if len(Rf) >= 3:
            pf, Rfinf = richardson(1 / np.array(nodes, float), Rf)
            print(f"FEM extrapolated: {Rfinf:.9f} Ohm (order {pf:.2f})")
            Rf_best = Rfinf
    if Rf_best is not None:
        print(f"JuliaMag - FEM, extrapolated: {Rinf - Rf_best:.2e} Ohm ({(Rinf - Rf_best) / Rf_best:.1e} relative)")
    # potentials at the cell centres, against the finest FEM on the same cell grid
    for (cx, cy), lst in sorted(groups.items()):
        k = f"jm_{tag}{cx}x{cy}.txt"
        if k not in files:
            continue
        phj = read(files[k])[1]
        phf = read(files[max(lst)[1]])[1]
        print(f"potential, {cx}x{cy} cells: max |JuliaMag - FEM (x{max(lst)[0]})| = {np.max(np.abs(phj - phf)):.2e} V of 1 V")
    print()
