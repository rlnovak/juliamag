# Reference demag field by direct circular convolution of the padded arrays with
# the (real-space) kernel. Deliberately simple and slow — a check, not the path.
# Defined before the testset so it exists when the testset body runs.
function direct_demag(kernel::DemagKernel{T}, m::AbstractArray{T,4},
                      mesh::Mesh, mat::Material) where {T}
    psize = kernel.padsize
    rx, ry, rz = dataregion(mesh)
    px, py, pz = psize

    pm = ntuple(_ -> zeros(T, psize...), 3)
    for c in 1:3, (kk, k) in enumerate(rz), (jj, j) in enumerate(ry), (ii, i) in enumerate(rx)
        pm[c][ii, jj, kk] = m[c, i, j, k]
    end

    Kc = (kernel.Kxx, kernel.Kyy, kernel.Kzz, kernel.Kxy, kernel.Kxz, kernel.Kyz)
    Bpad = ntuple(_ -> zeros(T, psize...), 3)
    pref = T(μ0 * mat.Msat)

    # Circular convolution: b[r] = Σ_r' K[(r-r') mod P] · m[r'].
    for kk in 1:pz, jj in 1:py, ii in 1:px
        bx = by = bz = zero(T)
        for k2 in 1:pz, j2 in 1:py, i2 in 1:px
            di = mod(ii - i2, px) + 1
            dj = mod(jj - j2, py) + 1
            dk = mod(kk - k2, pz) + 1
            Kxx = Kc[1][di, dj, dk]; Kyy = Kc[2][di, dj, dk]; Kzz = Kc[3][di, dj, dk]
            Kxy = Kc[4][di, dj, dk]; Kxz = Kc[5][di, dj, dk]; Kyz = Kc[6][di, dj, dk]
            Mx = pm[1][i2, j2, k2]; My = pm[2][i2, j2, k2]; Mz = pm[3][i2, j2, k2]
            bx += Kxx*Mx + Kxy*My + Kxz*Mz
            by += Kxy*Mx + Kyy*My + Kyz*Mz
            bz += Kxz*Mx + Kyz*My + Kzz*Mz
        end
        Bpad[1][ii, jj, kk] = pref * bx
        Bpad[2][ii, jj, kk] = pref * by
        Bpad[3][ii, jj, kk] = pref * bz
    end

    B = zeromag(T, mesh)
    for c in 1:3, (kk, k) in enumerate(rz), (jj, j) in enumerate(ry), (ii, i) in enumerate(rx)
        B[c, i, j, k] = Bpad[c][ii, jj, kk]
    end
    return B
end

@testset "Demag field" begin
    Msat = 8.0e5
    mat = Material(Msat = Msat, Aex = 1.3e-11, alpha = 0.02)

    # Build a plan from a mesh + material.
    makeplan(mesh) = DemagPlan(demagkernel(mesh), mesh, mat)

    @testset "uniformly magnetized cube: B = -μ0 Msat N·m, N=1/3" begin
        # Odd dimensions so that a single cell (4,4,4) sits at the exact centre
        # of the cube; there the field is purely antiparallel by symmetry.
        mesh = Mesh((7, 7, 7), (5e-9, 5e-9, 5e-9))
        plan = makeplan(mesh)
        c = (4, 4, 4)

        for dir in (1, 2, 3)
            u = ntuple(i -> i == dir ? 1.0 : 0.0, 3)
            m = uniform(mesh, u)
            B = similar(m)
            demagfield!(B, m, plan)

            # In the centre of a uniformly magnetized cube the demag field is
            # antiparallel to m with magnitude μ0 Msat /3.
            expected = -μ0 * Msat / 3
            @test B[dir, c...] ≈ expected rtol = 0.05
            # Transverse components vanish at the symmetry centre.
            for other in setdiff(1:3, dir)
                @test abs(B[other, c...]) < 1e-3 * abs(expected)
            end
        end
    end

    @testset "cube demag factors sum to μ0 Msat (trace rule on the field)" begin
        # Summing the centre-cell field magnitude over the three magnetization
        # directions must give μ0 Msat, since Nxx+Nyy+Nzz = 1.
        mesh = Mesh((6, 6, 6), (5e-9, 5e-9, 5e-9))
        plan = makeplan(mesh)
        c = (3, 3, 3)
        total = 0.0
        for dir in 1:3
            m = uniform(mesh, ntuple(i -> i == dir ? 1.0 : 0.0, 3))
            B = similar(m)
            demagfield!(B, m, plan)
            total += -B[dir, c...]
        end
        @test total ≈ μ0 * Msat rtol = 1e-3
    end

    @testset "flat cell stack: out-of-plane vs in-plane" begin
        # A wide, thin sample (thin along z) is nearly an infinite film: it
        # demagnetizes strongly when magnetized along z, weakly in-plane.
        mesh = Mesh((16, 16, 1), (4e-9, 4e-9, 4e-9))
        plan = makeplan(mesh)
        c = (8, 8, 1)

        mz = uniform(mesh, (0, 0, 1))
        Bz = similar(mz); demagfield!(Bz, mz, plan)

        mx = uniform(mesh, (1, 0, 0))
        Bx = similar(mx); demagfield!(Bx, mx, plan)

        # Out-of-plane demag dominates.
        @test abs(Bz[3, c...]) > 3 * abs(Bx[1, c...])
        # Both point against the magnetization.
        @test Bz[3, c...] < 0
        @test Bx[1, c...] < 0
    end

    @testset "matches a direct real-space convolution on a tiny mesh" begin
        # On a small mesh, cross-check the FFT path against a brute-force sum
        # using the same kernel — this catches indexing/padding/sign mistakes
        # independently of any analytic approximation.
        mesh = Mesh((4, 4, 2), (5e-9, 5e-9, 5e-9))
        kernel = demagkernel(mesh)
        plan = DemagPlan(kernel, mesh, mat)

        m = randommag!(zeromag(mesh))
        B = similar(m); demagfield!(B, m, plan)
        Bref = direct_demag(kernel, m, mesh, mat)

        @test B ≈ Bref rtol = 1e-8
    end

    @testset "add=true accumulates onto an existing buffer" begin
        mesh = Mesh((4, 4, 2), (5e-9, 5e-9, 5e-9))
        plan = makeplan(mesh)
        m = randommag!(zeromag(mesh))

        B1 = similar(m); demagfield!(B1, m, plan)
        B2 = fill!(similar(m), 1.5)
        demagfield!(B2, m, plan; add = true)
        @test B2 ≈ B1 .+ 1.5
    end

    @testset "Float32 plan" begin
        mesh = Mesh((4, 4, 1), (5e-9, 5e-9, 5e-9))
        mat32 = Material(Msat = 8.0f5, Aex = 1.3f-11, alpha = 0.02f0)
        plan = DemagPlan(demagkernel(Float32, mesh), mesh, mat32)
        m = uniform(Float32, mesh, (1, 0, 0))
        B = similar(m); demagfield!(B, m, plan)
        @test eltype(B) === Float32
        @test all(isfinite, B)
    end

    @testset "Fourier kernel is real (odd, 2D and periodic meshes)" begin
        # fourierkernel throws if any transformed component has an imaginary part;
        # the plan stores only the real part.
        # includes single-cell axes (wires, cross-sections, chains), where the
        # odd off-diagonal components vanish identically, and pbc ≥ 2 image sums
        for (sz, pbc) in (((7, 5, 3), (0, 0, 0)), ((8, 8, 1), (0, 0, 0)),
                          ((16, 8, 2), (1, 0, 0)), ((5, 6, 4), (0, 1, 1)),
                          ((16, 1, 4), (0, 0, 0)), ((1, 16, 4), (0, 0, 0)),
                          ((32, 1, 1), (0, 0, 0)), ((16, 1, 4), (1, 0, 0)),
                          ((8, 6, 2), (2, 0, 0)), ((6, 6, 3), (0, 0, 3)), ((1, 1, 1), (0, 0, 0)))
            mesh = Mesh(sz, (3e-9, 4e-9, 5e-9); pbc = pbc)
            K = JuliaMag.fourierkernel(demagkernel(mesh))
            @test all(k -> all(isfinite, k), K)
            @test size(K[1]) == (padsize(mesh)[1] ÷ 2 + 1, padsize(mesh)[2], padsize(mesh)[3])
        end
    end

    @testset "FFT path = direct convolution to round-off (open and periodic)" begin
        for (sz, pbc) in (((5, 4, 3), (0, 0, 0)), ((6, 4, 2), (1, 0, 0)))
            mesh = Mesh(sz, (5e-9, 4e-9, 3e-9); pbc = pbc)
            kernel = demagkernel(mesh)
            plan = DemagPlan(kernel, mesh, mat)
            m = randommag!(zeromag(mesh))
            B = similar(m); demagfield!(B, m, plan)
            Bref = direct_demag(kernel, m, mesh, mat)
            @test maximum(abs.(B .- Bref)) < 1e-12 * maximum(abs.(Bref))
            # Reusing the plan (padding never rewritten) gives the identical field.
            B2 = similar(m); demagfield!(B2, m, plan)
            @test B2 == B
        end
    end

    @testset "Float32 plan agrees with Float64 (kernel transformed in Float64)" begin
        mesh = Mesh((12, 10, 3), (4e-9, 4e-9, 4e-9))
        m = randommag!(zeromag(mesh))
        B64 = similar(m); demagfield!(B64, m, DemagPlan(demagkernel(mesh), mesh, mat))
        p32 = DemagPlan(demagkernel(Float64, mesh), mesh, Msat, Float32)
        B32 = zeromag(Float32, mesh); demagfield!(B32, Float32.(m), p32)
        @test eltype(p32.Kxx) === Float32
        @test maximum(abs.(B32 .- B64)) < 1e-5 * maximum(abs.(B64))
    end
end
