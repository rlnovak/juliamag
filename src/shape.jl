# Geometric shapes for defining sample geometry and material regions.
#
# Ported from mumax3 (engine/shape.go). A Shape is a predicate on position:
# `shape(x, y, z)` is true inside the shape and false outside, with x, y, z in
# metres measured from the CENTRE of the sample (the same convention as Config).
# Shapes are combined with set operations (union, intersection, difference,
# complement) and repositioned with `translate`, `scale`, and `rotz`, so complex
# geometries are built compositionally.

"""
    Shape

A geometric predicate `(x, y, z) -> Bool`, true inside the shape. Coordinates are
in metres from the sample centre. Combine shapes with `∪`, `∩`, `\\`, `!`, and
reposition them with [`translate`](@ref), [`scale`](@ref), [`rotz`](@ref).
"""
const Shape = Function

sqr(x) = x * x

# --- Primitive shapes (mirroring engine/shape.go) --------------------------

"""
    Cuboid(sx, sy, sz) -> Shape

Rectangular box (parallelepiped) with side lengths `sx, sy, sz`, centred at the
origin.
"""
function Cuboid(sx, sy, sz)
    rx, ry, rz = sx/2, sy/2, sz/2
    return (x, y, z) -> (-rx < x < rx) && (-ry < y < ry) && (-rz < z < rz)
end

"""
    Rect(sx, sy) -> Shape

2D rectangle (infinite along z) with side lengths `sx, sy`.
"""
Rect(sx, sy) = (x, y, z) -> (-sx/2 < x < sx/2) && (-sy/2 < y < sy/2)

"Square with side `s` (2D, infinite along z)."
Square(s) = Rect(s, s)

"""
    Cylinder(diam, height) -> Shape

Cylinder of diameter `diam` and `height`, axis along z, centred at the origin.
"""
function Cylinder(diam, height)
    r2 = (diam/2)^2
    h = height/2
    return (x, y, z) -> (-h <= z <= h) && (x*x + y*y <= r2)
end

"Circle of diameter `diam` (2D, infinite along z)."
Circle(diam) = (x, y, z) -> (x*x + y*y <= (diam/2)^2)

"""
    Ellipsoid(dx, dy, dz) -> Shape

Ellipsoid with axis diameters `dx, dy, dz`, centred at the origin.
"""
Ellipsoid(dx, dy, dz) = (x, y, z) -> sqr(x/dx) + sqr(y/dy) + sqr(z/dz) <= 0.25

"2D ellipse with axis diameters `dx, dy` (infinite along z)."
Ellipse(dx, dy) = (x, y, z) -> sqr(x/dx) + sqr(y/dy) <= 0.25

"""
    Cone(diam, height) -> Shape

Cone of base diameter `diam`, base at z=0, tip at z=`height`.
"""
function Cone(diam, height)
    return (x, y, z) -> (height - z) * z >= 0 && sqr(x/diam) + sqr(y/diam) <= 0.25 * sqr(1 - z/height)
end

"""
    Superball(diam, p) -> Shape

Superball of diameter `diam` and shape parameter `p`: interpolates between an
octahedron (p=0.5), a sphere (p=1), and a cube (p→∞). `p ≤ 0` is empty.
"""
function Superball(diam, p)
    p <= 0 && return (x, y, z) -> false
    return (x, y, z) -> abs(2x/diam)^(2p) + abs(2y/diam)^(2p) + abs(2z/diam)^(2p) <= 1
end

# --- Triangles and line segments (mirroring engine/shape.go) ---------------

"""
    Triangle(x0, y0, x1, y1, x2, y2) -> Shape

2D triangle with the three given vertices (metres), infinite along z. A point is
inside when it lies on the same side of all three edges (via the sign of the 2D
cross product; orientation-independent).
"""
function Triangle(x0, y0, x1, y1, x2, y2)
    # Signed area of the triangle (p, a, b).
    edge(px, py, ax, ay, bx, by) = (bx - ax) * (py - ay) - (by - ay) * (px - ax)
    return (x, y, z) -> begin
        d1 = edge(x, y, x0, y0, x1, y1)
        d2 = edge(x, y, x1, y1, x2, y2)
        d3 = edge(x, y, x2, y2, x0, y0)
        neg = (d1 < 0) || (d2 < 0) || (d3 < 0)
        pos = (d1 > 0) || (d2 > 0) || (d3 > 0)
        !(neg && pos)               # inside if not straddling both signs
    end
end

# Squared distance from point p to the segment a—b, clamped to the endpoints.
@inline function _seg_dist2(px, py, pz, ax, ay, az, bx, by, bz)
    dx, dy, dz = bx - ax, by - ay, bz - az
    l2 = dx*dx + dy*dy + dz*dz
    t = l2 == 0 ? 0.0 : ((px - ax)*dx + (py - ay)*dy + (pz - az)*dz) / l2
    t = clamp(t, 0.0, 1.0)
    qx, qy, qz = ax + t*dx, ay + t*dy, az + t*dz
    return sqr(px - qx) + sqr(py - qy) + sqr(pz - qz)
end

"""
    Line(p1, p2, diam) -> Shape

3D line segment (a capsule) from `p1` to `p2` (each a 3-tuple, metres) with
diameter `diam`: the set of points within `diam/2` of the segment (round caps).
"""
function Line(p1, p2, diam)
    r2 = (diam/2)^2
    a = NTuple{3,Float64}(p1); b = NTuple{3,Float64}(p2)
    return (x, y, z) -> _seg_dist2(x, y, z, a[1], a[2], a[3], b[1], b[2], b[3]) <= r2
end

"""
    Line2D(x1, y1, x2, y2, diam) -> Shape

2D line segment (capsule) from `(x1,y1)` to `(x2,y2)` with diameter `diam`,
infinite along z.
"""
function Line2D(x1, y1, x2, y2, diam)
    r2 = (diam/2)^2
    return (x, y, z) -> _seg_dist2(x, y, 0.0, x1, y1, 0.0, x2, y2, 0.0) <= r2
end

# --- Slabs and index-based shapes ------------------------------------------

"Half-space slab `x1 ≤ x < x2` (metres)."
XRange(x1, x2) = (x, y, z) -> x1 <= x < x2
"Half-space slab `y1 ≤ y < y2` (metres)."
YRange(y1, y2) = (x, y, z) -> y1 <= y < y2
"Half-space slab `z1 ≤ z < z2` (metres)."
ZRange(z1, z2) = (x, y, z) -> z1 <= z < z2

"Entire space."
Universe() = (x, y, z) -> true
"Empty shape."
Empty() = (x, y, z) -> false

"""
    Layers(mesh, k1, k2) -> Shape

The z-layers with cell index `k1 ≤ k < k2` (1-based), i.e. a horizontal slab a
few cells thick — the building block of multilayers.
"""
function Layers(mesh::Mesh, k1::Int, k2::Int)
    cz = mesh.cellsize[3]
    Nz = mesh.size[3]
    # Cell k centre is at z = (k - (Nz+1)/2) cz; its lower edge at that minus cz/2.
    zlo = (k1 - 1 - (Nz)/2) * cz
    zhi = (k2 - 1 - (Nz)/2) * cz
    return (x, y, z) -> zlo <= z < zhi
end

"Single z-layer with cell index `k` (1-based)."
Layer(mesh::Mesh, k::Int) = Layers(mesh, k, k + 1)

"""
    Cell(mesh, ix, iy, iz) -> Shape

The single cell at 1-based index `(ix, iy, iz)` — a `Cuboid` the size of one cell
placed at that cell's centre (mumax3's index-based `Cell`).
"""
function Cell(mesh::Mesh, ix::Int, iy::Int, iz::Int)
    Nx, Ny, Nz = mesh.size
    cx, cy, cz = mesh.cellsize
    x0 = (ix - (Nx + 1) / 2) * cx      # cell-centre convention (origin at sample centre)
    y0 = (iy - (Ny + 1) / 2) * cy
    z0 = (iz - (Nz + 1) / 2) * cz
    return (x, y, z) -> (abs(x - x0) < cx/2) && (abs(y - y0) < cy/2) && (abs(z - z0) < cz/2)
end

# --- Transforms (mirroring Config.Transl / Scale / RotZ) -------------------

"""
    translate(shape, dx, dy, dz) -> Shape

Move a shape so its origin is at `(dx, dy, dz)` [m].
"""
translate(s::Shape, dx, dy, dz) = (x, y, z) -> s(x - dx, y - dy, z - dz)

"""
    scale(shape, sx, sy, sz) -> Shape

Scale a shape by factors `sx, sy, sz` about the origin.
"""
scale(s::Shape, sx, sy, sz) = (x, y, z) -> s(x/sx, y/sy, z/sz)

"""
    rotz(shape, θ) -> Shape

Rotate a shape by `θ` radians about the z-axis.
"""
function rotz(s::Shape, θ)
    c, sn = cos(θ), sin(θ)
    return (x, y, z) -> s(x*c + y*sn, -x*sn + y*c, z)
end

"""
    rotx(shape, θ) -> Shape

Rotate a shape by `θ` radians about the x-axis.
"""
function rotx(s::Shape, θ)
    c, sn = cos(θ), sin(θ)
    return (x, y, z) -> s(x, y*c + z*sn, -y*sn + z*c)
end

"""
    roty(shape, θ) -> Shape

Rotate a shape by `θ` radians about the y-axis.
"""
function roty(s::Shape, θ)
    c, sn = cos(θ), sin(θ)
    return (x, y, z) -> s(x*c - z*sn, y, x*sn + z*c)
end

"""
    mirror(shape; x=false, y=false, z=false) -> Shape

Reflect a shape across the chosen coordinate planes (negate the selected axes).
"""
function mirror(s::Shape; x::Bool = false, y::Bool = false, z::Bool = false)
    return (px, py, pz) -> s(x ? -px : px, y ? -py : py, z ? -pz : pz)
end

"""
    repeat_shape(shape, px, py, pz) -> Shape

Tile `shape` with periods `px, py, pz` [m]: along each axis with a nonzero
period the coordinate is wrapped into `[-p/2, p/2)`, giving an infinite periodic
repetition (mumax3's `Repeat`). A zero period leaves that axis unrepeated. Named
`repeat_shape` to avoid shadowing `Base.repeat`.
"""
function repeat_shape(s::Shape, px, py, pz)
    wrap(v, p) = p == 0 ? v : (v - p * round(v / p))
    return (x, y, z) -> s(wrap(x, px), wrap(y, py), wrap(z, pz))
end

# --- Set operations --------------------------------------------------------
# Named functions (union/intersect/difference/complement) plus operator forms.

"Union of shapes: inside either."
shapeunion(a::Shape, b::Shape) = (x, y, z) -> a(x, y, z) || b(x, y, z)
"Intersection of shapes: inside both."
shapeintersect(a::Shape, b::Shape) = (x, y, z) -> a(x, y, z) && b(x, y, z)
"Difference: inside `a` but not `b`."
shapediff(a::Shape, b::Shape) = (x, y, z) -> a(x, y, z) && !b(x, y, z)
"Complement: outside `a`."
shapecomplement(a::Shape) = (x, y, z) -> !a(x, y, z)
"Exclusive or: inside exactly one of `a`, `b`."
shapexor(a::Shape, b::Shape) = (x, y, z) -> a(x, y, z) ⊻ b(x, y, z)
