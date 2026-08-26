@testset "Shapes" begin
    @testset "cuboid / rect / square" begin
        c = Cuboid(100e-9, 40e-9, 10e-9)
        @test c(0, 0, 0)                    # centre is inside
        @test c(49e-9, 19e-9, 4e-9)         # just inside
        @test !c(51e-9, 0, 0)               # beyond x half-side
        @test !c(0, 21e-9, 0)               # beyond y half-side
        @test !c(0, 0, 6e-9)                # beyond z half-side

        r = Rect(100e-9, 40e-9)
        @test r(0, 0, 1e6)                  # infinite along z
        @test !r(60e-9, 0, 0)

        @test Square(20e-9)(9e-9, 9e-9, 0)
        @test !Square(20e-9)(11e-9, 0, 0)
    end

    @testset "cylinder / circle" begin
        cyl = Cylinder(50e-9, 20e-9)        # diameter 50, height 20, axis z
        @test cyl(0, 0, 0)
        @test cyl(24e-9, 0, 9e-9)           # inside radius, inside height
        @test !cyl(26e-9, 0, 0)             # outside radius
        @test !cyl(0, 0, 11e-9)             # outside height
        # corner of the bounding box is outside the circle
        @test !cyl(24e-9, 24e-9, 0)

        @test Circle(50e-9)(0, 24e-9, 1e-3) # infinite along z
        @test !Circle(50e-9)(26e-9, 0, 0)
    end

    @testset "ellipsoid" begin
        e = Ellipsoid(100e-9, 40e-9, 10e-9)
        @test e(0, 0, 0)
        @test e(49e-9, 0, 0)                # on the long axis, just inside
        @test !e(51e-9, 0, 0)
        # a point at the box corner is outside the ellipsoid
        @test !e(49e-9, 19e-9, 4e-9)
    end

    @testset "layers (multilayer building block)" begin
        mesh = Mesh((4, 4, 6), (5e-9, 5e-9, 5e-9))   # 6 layers along z
        # Layer index is 1-based; layer 1 is the bottom.
        bottom = Layer(mesh, 1)
        top = Layer(mesh, 6)
        # Cell k centre z:
        zc(k) = (k - (6 + 1)/2) * 5e-9
        @test bottom(0, 0, zc(1))
        @test !bottom(0, 0, zc(2))
        @test top(0, 0, zc(6))
        @test !top(0, 0, zc(5))

        # A 2-layer slab.
        slab = Layers(mesh, 2, 4)           # layers 2 and 3
        @test slab(0, 0, zc(2))
        @test slab(0, 0, zc(3))
        @test !slab(0, 0, zc(1))
        @test !slab(0, 0, zc(4))
    end

    @testset "xrange / zrange" begin
        @test XRange(0, 10e-9)(5e-9, 0, 0)
        @test !XRange(0, 10e-9)(-1e-9, 0, 0)
        @test !XRange(0, 10e-9)(10e-9, 0, 0)   # upper bound exclusive
        @test ZRange(-5e-9, 5e-9)(0, 0, 0)
    end

    @testset "transforms" begin
        cyl = Cylinder(20e-9, 1e6)
        # Translate the core to x = 30 nm.
        moved = translate(cyl, 30e-9, 0, 0)
        @test moved(30e-9, 0, 0)
        @test !moved(0, 0, 0)

        # Scale doubles the radius.
        big = scale(Cylinder(20e-9, 1e6), 2, 2, 1)
        @test big(19e-9, 0, 0)              # inside the scaled radius (20 nm)
        @test !Cylinder(20e-9, 1e6)(11e-9, 0, 0)  # outside the unscaled one

        # Rotate a rectangle 90°: a point off the long axis maps inside.
        rect = Rect(100e-9, 20e-9)
        rot = rotz(rect, π/2)
        @test rot(0, 40e-9, 0)              # long axis now along y
        @test !rect(0, 40e-9, 0)
    end

    @testset "set operations" begin
        a = Cylinder(40e-9, 1e6)
        b = translate(Cylinder(40e-9, 1e6), 30e-9, 0, 0)

        u = shapeunion(a, b)
        @test u(0, 0, 0) && u(30e-9, 0, 0)

        i = shapeintersect(a, b)
        @test i(15e-9, 0, 0)                # overlap region
        @test !i(-15e-9, 0, 0)              # only in a

        d = shapediff(a, b)
        @test d(-15e-9, 0, 0)               # in a, not b
        @test !d(15e-9, 0, 0)               # in both → excluded

        comp = shapecomplement(a)
        @test comp(100e-9, 0, 0)
        @test !comp(0, 0, 0)
    end

    @testset "universe / empty" begin
        @test Universe()(1e9, -1e9, 1e9)
        @test !Empty()(0, 0, 0)
    end

    @testset "triangle" begin
        t = Triangle(-30e-9, -20e-9, 30e-9, -20e-9, 0, 40e-9)
        @test t(0, 0, 0)                    # centroid-ish, inside
        @test t(0, -19e-9, 1e6)             # near the base, infinite along z
        @test !t(0, 41e-9, 0)               # above the apex
        @test !t(40e-9, -20e-9, 0)          # beyond a base corner
    end

    @testset "line segments (capsules)" begin
        l2 = Line2D(-50e-9, 0, 50e-9, 0, 10e-9)   # along x, radius 5 nm
        @test l2(0, 0, 1e6)                 # on the axis, infinite along z
        @test l2(0, 4e-9, 0)                # within radius
        @test !l2(0, 6e-9, 0)              # beyond radius
        @test l2(50e-9, 4e-9, 0)           # round cap at the end
        @test !l2(60e-9, 0, 0)            # past the cap

        l3 = Line((0, 0, -20e-9), (0, 0, 20e-9), 8e-9)  # along z, radius 4 nm
        @test l3(0, 0, 0) && l3(3e-9, 0, 0)
        @test !l3(5e-9, 0, 0)
        @test !l3(0, 0, 25e-9)            # past the end + cap
    end

    @testset "cell by index" begin
        mesh = Mesh((10, 10, 2), (5e-9, 5e-9, 5e-9))
        c = Cell(mesh, 3, 4, 1)
        x0 = (3 - (10 + 1) / 2) * 5e-9; y0 = (4 - (10 + 1) / 2) * 5e-9; z0 = (1 - (2 + 1) / 2) * 5e-9
        @test c(x0, y0, z0)                 # the cell centre
        @test !c(x0 + 5e-9, y0, z0)         # the next cell over
    end

    @testset "rotx / roty" begin
        rect = Rect(80e-9, 20e-9)           # long axis x
        @test roty(rect, 0)(30e-9, 0, 0)    # identity rotation
        # A z-slab rotated 90° about x becomes a y-slab.
        slab = ZRange(-5e-9, 5e-9)
        rotated = rotx(slab, π/2)
        @test rotated(0, 3e-9, 0)           # now bounded in y
        @test !rotated(0, 20e-9, 0)
    end

    @testset "mirror" begin
        s = translate(Circle(20e-9), 40e-9, 0, 0)   # circle centred at +x
        @test s(40e-9, 0, 0) && !s(-40e-9, 0, 0)
        mx = mirror(s; x = true)
        @test mx(-40e-9, 0, 0) && !mx(40e-9, 0, 0)  # reflected to −x
    end

    @testset "repeat / tile" begin
        dot = Circle(10e-9)                 # a small dot at the origin
        tiled = repeat_shape(dot, 40e-9, 0, 0)   # repeat along x every 40 nm
        @test tiled(0, 0, 0)
        @test tiled(40e-9, 0, 0)            # image one period over
        @test tiled(-40e-9, 0, 0)
        @test !tiled(20e-9, 0, 0)          # midway between images
    end

    @testset "xor" begin
        a = Cylinder(40e-9, 1e6)
        b = translate(Cylinder(40e-9, 1e6), 30e-9, 0, 0)
        x = shapexor(a, b)
        @test x(-15e-9, 0, 0)              # only in a
        @test x(45e-9, 0, 0)               # only in b
        @test !x(15e-9, 0, 0)             # in both → excluded
    end
end
