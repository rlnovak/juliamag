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
