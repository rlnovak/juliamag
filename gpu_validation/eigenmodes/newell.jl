# The demagnetization tensor of Newell, Williams and Dunlop (J. Geophys. Res. 98, 9551
# (1993)), analytic for uniformly magnetized cells, as a JuliaMag DemagKernel (open
# boundaries): the tensor of OOMMF. JuliaMag's own kernel is mumax3's numerical integral.
# 256-bit arithmetic: the second differences cancel strongly for distant cells.
using JuliaMag


function _nf(x, y, z)
    R = sqrt(x^2 + y^2 + z^2)
    s = (2x^2 - y^2 - z^2) * R / 6
    iszero(y) || iszero(x^2 + z^2) || (s += y / 2 * (z^2 - x^2) * asinh(y / sqrt(x^2 + z^2)))
    iszero(z) || iszero(x^2 + y^2) || (s += z / 2 * (y^2 - x^2) * asinh(z / sqrt(x^2 + y^2)))
    iszero(x) || (s -= x * y * z * atan(y * z / (x * R)))
    return s
end
function _ng(x, y, z)
    R = sqrt(x^2 + y^2 + z^2)
    s = -x * y * R / 3
    iszero(z) || iszero(x^2 + y^2) || (s += x * y * z * asinh(z / sqrt(x^2 + y^2)))
    iszero(x) || iszero(y^2 + z^2) || (s += y / 6 * (3z^2 - y^2) * asinh(x / sqrt(y^2 + z^2)))
    iszero(y) || iszero(x^2 + z^2) || (s += x / 6 * (3z^2 - x^2) * asinh(y / sqrt(x^2 + z^2)))
    iszero(z) || (s -= z^3 / 6 * atan(x * y / (z * R)))
    iszero(y) || (s -= z * y^2 / 2 * atan(x * z / (y * R)))
    iszero(x) || (s -= z * x^2 / 2 * atan(y * z / (x * R)))
    return s
end
# −(1,−2,1)³ second differences over 4π V
function _n3(f, I, c)
    s = zero(BigFloat)
    for a in -1:1, b in -1:1, d in -1:1
        w = (a == 0 ? -2 : 1) * (b == 0 ? -2 : 1) * (d == 0 ? -2 : 1)
        s += w * f((I[1] + a) * c[1], (I[2] + b) * c[2], (I[3] + d) * c[3])
    end
    return -s / (4 * big(π) * c[1] * c[2] * c[3])
end

"Newell's tensor for `mesh` (open boundaries) as a DemagKernel{Float64}."
function newellkernel(mesh::Mesh)
    all(==(0), mesh.pbc) || error("open boundaries only")
    P = JuliaMag.padsize(mesh); N = mesh.size
    h = minimum(mesh.cellsize)
    c = big.(mesh.cellsize) ./ big(h)                       # scale free
    K = ntuple(_ -> zeros(P...), 6)
    fs = (I -> _n3(_nf, I, c),
          I -> _n3((x, y, z) -> _nf(y, x, z), I, c),
          I -> _n3((x, y, z) -> _nf(z, y, x), I, c),
          I -> _n3(_ng, I, c),
          I -> _n3((x, y, z) -> _ng(x, z, y), I, c),
          I -> _n3((x, y, z) -> _ng(y, z, x), I, c))
    odd = ((0, 0, 0), (0, 0, 0), (0, 0, 0), (1, 1, 0), (1, 0, 1), (0, 1, 1))
    for q in 1:6, k in 0:N[3]-1, j in 0:N[2]-1, i in 0:N[1]-1
        v = -Float64(fs[q]((i, j, k)))                      # field: B = −N m
        for sx in (1, -1), sy in (1, -1), sz in (1, -1)
            s = (odd[q][1] == 1 ? sx : 1) * (odd[q][2] == 1 ? sy : 1) * (odd[q][3] == 1 ? sz : 1)
            K[q][mod(sx * i, P[1])+1, mod(sy * j, P[2])+1, mod(sz * k, P[3])+1] = s * v
        end
    end
    return JuliaMag.DemagKernel{Float64}(K..., P)
end

"Replace the demag kernel of a Float64 CPU world by Newell's."
function usenewell!(w::World)
    w.demagplan = DemagPlan(newellkernel(w.mesh), w.mesh, JuliaMag._demag_msat(w.material), Float64)
    return w
end
