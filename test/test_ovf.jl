@testset "OVF I/O" begin
    # Write a small OVF file in a given format, then read it back and compare.
    function write_ovf_text(path, m, mesh)
        Nx, Ny, Nz = mesh.size
        cx, cy, cz = mesh.cellsize
        open(path, "w") do io
            println(io, "# OOMMF OVF 2.0")
            println(io, "# Segment count: 1")
            println(io, "# Begin: Segment")
            println(io, "# Begin: Header")
            println(io, "# xnodes: ", Nx); println(io, "# ynodes: ", Ny); println(io, "# znodes: ", Nz)
            println(io, "# xstepsize: ", cx); println(io, "# ystepsize: ", cy); println(io, "# zstepsize: ", cz)
            println(io, "# valuemultiplier: 1.0")
            println(io, "# End: Header")
            println(io, "# Begin: Data Text")
            for k in 1:Nz, j in 1:Ny, i in 1:Nx
                println(io, m[1,i,j,k], " ", m[2,i,j,k], " ", m[3,i,j,k])
            end
            println(io, "# End: Data Text")
            println(io, "# End: Segment")
        end
    end

    function write_ovf_binary(path, m, mesh; nbytes = 4)
        Nx, Ny, Nz = mesh.size
        cx, cy, cz = mesh.cellsize
        F = nbytes == 4 ? Float32 : Float64
        check = nbytes == 4 ? 1234567.0f0 : 123456789012345.0
        open(path, "w") do io
            println(io, "# OOMMF OVF 2.0")
            println(io, "# xnodes: ", Nx); println(io, "# ynodes: ", Ny); println(io, "# znodes: ", Nz)
            println(io, "# xstepsize: ", cx); println(io, "# ystepsize: ", cy); println(io, "# zstepsize: ", cz)
            println(io, "# Begin: Data Binary ", nbytes)
            write(io, F(check))
            for k in 1:Nz, j in 1:Ny, i in 1:Nx, c in 1:3
                write(io, F(m[c,i,j,k]))
            end
            println(io)
            println(io, "# End: Data Binary ", nbytes)
        end
    end

    mesh = Mesh((6, 4, 2), (5e-9, 5e-9, 5e-9))
    m = setconfig(mesh, VortexConfig(mesh; circ = 1, pol = 1))
    dir = mktempdir()

    @testset "text round-trip" begin
        path = joinpath(dir, "state_text.ovf")
        write_ovf_text(path, m, mesh)
        m2, header = loadovf(path)
        @test size(m2) == size(m)
        @test m2 ≈ m rtol = 1e-6
        @test Int(header["xnodes"]) == 6
        @test header["xstepsize"] ≈ 5e-9
    end

    @testset "binary 4-byte round-trip" begin
        path = joinpath(dir, "state_bin4.ovf")
        write_ovf_binary(path, m, mesh; nbytes = 4)
        m2, _ = loadovf(path)
        @test m2 ≈ m rtol = 1e-5          # Float32 storage
    end

    @testset "binary 8-byte round-trip" begin
        path = joinpath(dir, "state_bin8.ovf")
        write_ovf_binary(path, m, mesh; nbytes = 8)
        m2, _ = loadovf(path)
        @test m2 ≈ m rtol = 1e-12
    end

    @testset "meshfromovf reconstructs the mesh" begin
        path = joinpath(dir, "state_text.ovf")
        _, header = loadovf(path)
        m2 = meshfromovf(header)
        @test size(m2) == (6, 4, 2)
        @test all(m2.cellsize .≈ (5e-9, 5e-9, 5e-9))
    end

    @testset "loaded OVF can seed a simulation" begin
        path = joinpath(dir, "state_text.ovf")
        m2, header = loadovf(path)
        mesh2 = meshfromovf(header)
        # the loaded state is usable as an initial condition
        world = World(mesh2, Material(Msat = 8e5, Aex = 1.3e-11, alpha = 0.02); demag = false)
        B = similar(m2)
        effectivefield!(B, m2, world)
        @test all(isfinite, B)
    end
end
