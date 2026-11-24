"""Magnetization snapshots of the 50 K re-runs (OVF text, 100x100x4, 5 nm cells).

Colour: in-plane angle of m (layer-averaged); white contours: m_z = +-0.5 (cores).
For every core (|m_z| > 0.7 blob) the in-plane winding number on a loop of radius
4 cells around it is printed: +1 vortex, -1 antivortex.
File m000NNN.ovf is at t = t(step 62) + NNN x 0.1 ns.
"""
import numpy as np
import matplotlib.pyplot as plt
from scipy import ndimage

FIELDS = [0, -5, -10, -15, -20, -25, -30]


def readovf(path):
    with open(path, "rb") as f:
        lines = f.read().decode("utf-8", "replace").splitlines()
    t = float([l for l in lines if l.startswith("# Title")][0].split("t=")[1].split()[0])
    i0 = next(i for i, l in enumerate(lines) if l.startswith("# Begin: Data"))
    data = np.array([list(map(float, l.split())) for l in lines[i0 + 1:] if l and not l.startswith("#")])
    m = data.reshape(4, 100, 100, 3)       # z, y, x, comp (OVF order x fastest)
    return t, m


def winding(mx, my, cx, cy, r=4):
    th = np.linspace(0, 2 * np.pi, 64, endpoint=False)
    xs = np.clip(np.round(cx + r * np.cos(th)).astype(int), 0, 99)
    ys = np.clip(np.round(cy + r * np.sin(th)).astype(int), 0, 99)
    a = np.unwrap(np.arctan2(my[ys, xs], mx[ys, xs]))
    return (a[-1] - a[0] + np.angle(np.exp(1j * (a[0] - a[-1])))) / (2 * np.pi)


if __name__ == "__main__":
    fig, axs = plt.subplots(2, 3, figsize=(12, 8.4))
    for r, seed in enumerate(("seed1", "seed7")):
        for c, idx in enumerate((65, 90, 110)):
            t, m = readovf(f"ovf/{seed}_m{idx:06d}.ovf")
            mm = m.mean(axis=0)
            mx, my, mz = mm[..., 0], mm[..., 1], mm[..., 2]
            inside = (mx**2 + my**2 + mz**2) > 0.25
            ang = np.where(inside, np.arctan2(my, mx), np.nan)
            ax = axs[r, c]
            ext = (-250, 250, -250, 250)
            ax.imshow(ang, origin="lower", cmap="hsv", vmin=-np.pi, vmax=np.pi, extent=ext)
            ax.contour(np.linspace(-247.5, 247.5, 100), np.linspace(-247.5, 247.5, 100), mz,
                       levels=[-0.5, 0.5], colors="w", linewidths=1.2)
            s = 8
            X, Y = np.meshgrid(np.linspace(-247.5, 247.5, 100), np.linspace(-247.5, 247.5, 100))
            ax.quiver(X[::s, ::s], Y[::s, ::s], np.where(inside, mx, 0)[::s, ::s], np.where(inside, my, 0)[::s, ::s],
                      scale=18, width=0.004)
            lab, n = ndimage.label(np.abs(mz) > 0.7)
            cores = []
            for k in range(1, n + 1):
                yy, xx = np.nonzero(lab == k)
                cy, cx = yy.mean(), xx.mean()
                w = winding(mx, my, cx, cy)
                p = np.sign(mz[lab == k].mean())
                cores.append((5 * cx - 247.5, 5 * cy - 247.5, w, p))
            k62 = int(idx // 20)
            B = FIELDS[min(k62, 6)]
            ttl = f"{seed}, t0+{idx/10:.1f} ns, B = {B} mT\n" + ", ".join(
                f"({x:+.0f},{y:+.0f}) w={w:+.0f} p={p:+.0f}" for x, y, w, p in cores)
            ax.set_title(ttl, fontsize=8)
            ax.set_xlabel("x (nm)"); ax.set_ylabel("y (nm)")
            print(f"{seed} idx {idx} t={t:.4e}  <mx>={mx[inside].mean():+.3f}  cores: {cores}")
    fig.tight_layout()
    fig.savefig("snapshots.png", dpi=130)
    print("wrote snapshots.png")
