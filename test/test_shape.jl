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
end
