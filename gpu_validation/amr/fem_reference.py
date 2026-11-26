# Independent reference for the AMR conduction of JuliaMag: linear finite elements (P1 on triangles),
# a separate discretization of the same problem.
#
#   python gpu_validation/amr/fem_reference.py OUT [wall|uniform45]
#
# A rectangle L x W (thickness t), contacts on the short sides (V = 1 and 0), insulating long sides.
# Texture: a domain wall at an angle, m = (cos θ, sin θ, 0), θ = 2 atan(exp((x - 0.3 y)/δ)) (θ from 0 to π
# across the wall), so the conductivity tensor σ = (1/ρ⊥)[I − (Δρ/ρ∥) m mᵀ] varies in x and y and its
# off-diagonal part changes sign. Writes OUT/fem_N.txt: the resistance and the potential on the cell
# centres of a 2D grid of N_x x N_y cells (the points where JuliaMag has its unknowns), for several meshes.
# `uniform45`: m = (1, 1, 0)/√2 everywhere instead, the largest planar Hall field, shorted by the
# contacts (a test of the boundary condition on their faces).
import sys, os
import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla

L, W, t = 400e-9, 80e-9, 4e-9
rho_par, rho_perp = 210e-9, 200e-9
delta = 15e-9


TEXTURE = "wall"


def mvec(x, y):
    if TEXTURE == "uniform45":
        return np.sqrt(0.5), np.sqrt(0.5)
    th = 2 * np.arctan(np.exp((x - 0.3 * y) / delta))
    return np.cos(th), np.sin(th)


def sigma(x, y):
    mx, my = mvec(x, y)
    a = (rho_par - rho_perp) / rho_par
    s = 1 / rho_perp
    return np.array([[s * (1 - a * mx * mx), -s * a * mx * my], [-s * a * mx * my, s * (1 - a * my * my)]])


def solve(nx, ny, q=4):
    # nodes of a uniform grid refined q times relative to the cell grid of nx x ny cells
    Nx, Ny = q * nx, q * ny
    xs = np.linspace(-L / 2, L / 2, Nx + 1); ys = np.linspace(-W / 2, W / 2, Ny + 1)
    nid = lambda i, j: j * (Nx + 1) + i
    N = (Nx + 1) * (Ny + 1)
    rows, cols, vals = [], [], []
    hx, hy = L / Nx, W / Ny
    for j in range(Ny):
        for i in range(Nx):
            # two triangles per square, alternating the diagonal (no preferred direction)
            a, b, c, d = nid(i, j), nid(i + 1, j), nid(i + 1, j + 1), nid(i, j + 1)
            tris = [(a, b, c), (a, c, d)] if (i + j) % 2 == 0 else [(a, b, d), (b, c, d)]
            for tr in tris:
                P = np.array([[xs[k % (Nx + 1)], ys[k // (Nx + 1)]] for k in tr])
                B = np.array([[P[1, 0] - P[0, 0], P[2, 0] - P[0, 0]], [P[1, 1] - P[0, 1], P[2, 1] - P[0, 1]]])
                area = 0.5 * abs(np.linalg.det(B))
                G = np.linalg.solve(B.T, np.array([[-1.0, 1.0, 0.0], [-1.0, 0.0, 1.0]]))   # gradients of the 3 hat functions
                cx, cy = P.mean(axis=0)
                K = t * area * (G.T @ sigma(cx, cy) @ G)
                for r in range(3):
                    for s_ in range(3):
                        rows.append(tr[r]); cols.append(tr[s_]); vals.append(K[r, s_])
    K = sp.csr_matrix((vals, (rows, cols)), shape=(N, N))
    left = [nid(0, j) for j in range(Ny + 1)]; right = [nid(Nx, j) for j in range(Ny + 1)]
    fixed = np.zeros(N, bool); fixed[left] = True; fixed[right] = True
    u = np.zeros(N); u[left] = 1.0
    free = ~fixed
    rhs = -K[free][:, fixed] @ u[fixed]
    u[free] = spla.spsolve(K[free][:, free].tocsc(), rhs)
    I = (K @ u)[left].sum()                              # current fed by the contact at V = 1
    # potential at the cell centres of the nx x ny grid, by linear interpolation in the P1 field
    U = u.reshape(Ny + 1, Nx + 1)
    cxs = -L / 2 + (np.arange(nx) + 0.5) * L / nx; cys = -W / 2 + (np.arange(ny) + 0.5) * W / ny
    phi = np.zeros((ny, nx))
    for jj, y in enumerate(cys):
        for ii, x in enumerate(cxs):
            fi = (x + L / 2) / hx; fj = (y + W / 2) / hy
            i0, j0 = int(fi), int(fj); a, b = fi - i0, fj - j0
            # inside the square both triangles interpolate linearly: the bilinear value differs at second order,
            # but the cell centres of the coarse grid fall on nodes when q is even
            phi[jj, ii] = (1 - a) * (1 - b) * U[j0, i0] + a * (1 - b) * U[j0, i0 + 1] + (1 - a) * b * U[j0 + 1, i0] + a * b * U[j0 + 1, i0 + 1]
    return 1 / I, phi


if __name__ == "__main__":
    out = sys.argv[1]; os.makedirs(out, exist_ok=True)
    TEXTURE = sys.argv[2] if len(sys.argv) > 2 else "wall"
    tag = "" if TEXTURE == "wall" else TEXTURE + "_"
    for nx, ny in (((100, 20), (200, 40)) if TEXTURE == "wall" else ((100, 20),)):
        for q in ((2, 4, 8) if TEXTURE == "wall" else (4, 8, 16)):
            R, phi = solve(nx, ny, q)
            print(f"cells {nx}x{ny}, FEM nodes x{q}: R = {R:.9f} Ohm")
            np.savetxt(os.path.join(out, f"fem_{tag}{nx}x{ny}_q{q}.txt"), phi, header=f"R {R:.12e}")
