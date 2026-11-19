# Generic Julia code traced into CasADi graphs with `Traced`.
import LinearAlgebra
import SparseArrays
import ForwardDiff
import SpecialFunctions
using StaticArrays: SVector

const traced, untraced, Traced = CasADiCodegen.traced, CasADiCodegen.untraced, CasADiCodegen.Traced

# Generic code with no reference to CasADi: a static vector, `norm` and a power.
two_body(r::SVector{3}, μ) = -μ * r / LinearAlgebra.norm(r)^3
two_body_jacobian(r, μ) = (n = LinearAlgebra.norm(r); -μ * (LinearAlgebra.I / n^3 - 3 * r * r' / n^5))
step_down(x) = x[1] > 0 ? x[1] : -x[1]

@testset "a generic Julia function traces into SX" begin
    μ, r0 = 398600.4418, [7000.0, -1200.0, 300.0]
    x = SX.sym("r", 3)
    a = untraced(two_body(SVector{3}(traced(x)), μ))
    g, _ = build_function(CasadiFunction("a", [x], [a, jacobian(a, x)]); expression = Val{false})
    value, slope = g(r0)
    @test value ≈ -μ * r0 / LinearAlgebra.norm(r0)^3 rtol = 1e-14
    @test slope ≈ two_body_jacobian(r0, μ) rtol = 1e-12
end

@testset "a branch on a traced value throws" begin
    @test_throws TypeError step_down(traced(SX.sym("x", 1)))
end

@testset "min, max, clamp and ifelse trace without a branch" begin
    x = SX.sym("x", 1)
    t = only(traced(x))
    outputs = [untraced(min(t, 1)), untraced(max(2t, -1)), untraced(clamp(t, -0.5, 0.5)), untraced(ifelse(t < 0, -t, t^2))]
    g, _ = build_function(CasadiFunction("s", [x], [vcat(outputs...)]); expression = Val{false})
    for v in (-2.0, -0.3, 0.4, 3.0)
        @test g([v]) == [min(v, 1), max(2v, -1), clamp(v, -0.5, 0.5), v < 0 ? -v : v^2]
    end
end

@testset "MX values trace through a callback" begin
    scale = callback(v -> 3 .* v, "scale", (2,), (2,); jacobian = v -> [3.0 0.0; 0.0 3.0])
    x = MX.sym("x", 2)
    y = traced(scale(x))
    @test eltype(y) == Traced{MX}
    z = untraced([y[1] * y[2], sin(y[1]) + 1])
    g, _ = build_function(CasadiFunction("m", [x], [z, jacobian(z, x)]); expression = Val{false})
    value, slope = g([0.2, 0.5])
    @test value ≈ [0.6 * 1.5, sin(0.6) + 1]
    @test slope ≈ [4.5 1.8; 3cos(0.6) 0.0]
end

@testset "traced_function traces a function of the state, the parameters and the time" begin
    # In place on a matrix state, with the time.
    function spread!(du, u, p, t)
        du[:, 1] .= p[1] .* u[:, 2]
        du[:, 2] .= -u[:, 1] .* cos(t)
    end
    u0 = [1.0 2.0; 3.0 4.0]
    f = CasADiCodegen.traced_function("spread", spread!, u0, [0.5])
    du = similar(u0)
    spread!(du, u0, [0.5], 0.3)
    @test build_function(f; expression = Val{false})[1](vec(u0), [0.5], 0.3) ≈ vec(du)
    # Out of place, with a scalar parameter and without the time.
    g = CasADiCodegen.traced_function("decay", (u, k) -> -k * u, [1.0, 2.0], 0.5; inplace = false, time = false)
    @test build_function(g; expression = Val{false})[1]([1.0, 2.0], 0.5) ≈ [-0.5, -1.0]
end

@testset "rounding, remainders and roots of traced values" begin
    x, y = SX.sym("x"), SX.sym("y")
    a, b = only(traced(x)), only(traced(y))
    outputs = [floor(a), ceil(a), trunc(a), rem(a, b), mod(a, b), copysign(b, a), cbrt(a), (a < 0) & (b > 0) | false]
    g, _ = build_function(CasadiFunction("r", [x, y], [untraced(outputs)]); expression = Val{false})
    for (u, v) in ((-2.5, 0.75), (3.25, -1.5), (0.0, 2.0), (-8.0, 3.0))
        @test g([u], [v]) ≈ [floor(u), ceil(u), trunc(u), rem(u, v), mod(u, v), copysign(v, u), cbrt(u), (u < 0) & (v > 0)] atol = 1e-14
    end
end

@testset "mod and clamp of traced values are those of Julia, bit for bit" begin
    x, y, z = SX.sym("x"), SX.sym("y"), SX.sym("z")
    a, b, c = only(traced(x)), only(traced(y)), only(traced(z))
    m = CasadiFunction("m", [x, y], [untraced(mod(a, b))])
    g, _ = build_function(m; expression = Val{false})
    @testset "mod($u, $v)" for (u, v) in JULIA_MOD_CASES
        @test isequal(only(g([u], [v])), mod(u, v))
        @test isequal(only(only(casadi_eval(m, u, v))), mod(u, v))
    end
    k = CasadiFunction("k", [x, y, z], [untraced(clamp(a, b, c))])
    h, _ = build_function(k; expression = Val{false})
    @testset "clamp($u, $lo, $hi)" for (u, lo, hi) in JULIA_CLAMP_CASES
        @test isequal(only(h([u], [lo], [hi])), clamp(u, lo, hi))
    end
    t = CasadiFunction("t", [x], [untraced(trunc(a))])
    @test isequal(only(build_function(t; expression = Val{false})[1]([-0.5])), -0.0)
end

@testset "determinant, inverse and solve of a traced matrix" begin
    # LinearAlgebra factorizes with pivoting, which compares values; CasADi forms them from the graph.
    z = SX.sym("z", 4)
    A = reshape(traced(z), 2, 2)
    # A view with index vectors is not strided, so it takes the other method of `inv`.
    outputs = [LinearAlgebra.det(A); vec(LinearAlgebra.inv(A)); A \ [1.0, 2.0]; vec(inv(view(A, [2, 1], :)))]
    g, _ = build_function(CasadiFunction("m", [z], [untraced(outputs)]); expression = Val{false})
    M = [2.0 1.0; -1.0 3.0]
    @test g(vec(M)) ≈ [LinearAlgebra.det(M); vec(inv(M)); M \ [1.0, 2.0]; vec(inv(M[[2, 1], :]))]
end

@testset "a structured or sparse traced matrix solves as the matrix it stands for" begin
    LA, SA = LinearAlgebra, SparseArrays
    # `sparse` of a dense matrix asks whether each entry is zero, which a traced value cannot
    # answer, so the sparse matrix is made from its entries.
    stored(M) = SA.sparse([1, 2, 3, 1, 3], [1, 2, 3, 3, 1], [M[1, 1], M[2, 2], M[3, 3], M[1, 3], M[3, 1]], 3, 3)
    structures = Any[LA.UpperTriangular, LA.LowerTriangular, LA.UnitUpperTriangular, LA.UnitLowerTriangular,
                     LA.Symmetric, LA.Hermitian, stored,
                     M -> LA.Diagonal(M[1:3]), M -> LA.Bidiagonal(M[1:3], M[4:5], :U),
                     M -> LA.Tridiagonal(M[4:5], M[1:3], M[6:7]), M -> LA.SymTridiagonal(M[1:3], M[4:5])]
    z = SX.sym("z", 9)
    A = reshape(traced(z), 3, 3)
    b = [1.0, 2.0, 3.0]
    g, _ = build_function(CasadiFunction("s", [z], [untraced(reduce(vcat, [W(A) \ b for W in structures]))]); expression = Val{false})
    M = [4.0 0.5 1.0; 1.0 5.0 0.25; -1.0 2.0 6.0]
    # LinearAlgebra solves each structure with a method of its own; CasADi factorizes the full matrix.
    @test g(vec(M)) ≈ reduce(vcat, [W(M) \ b for W in structures]) rtol = 1e-12
end

@testset "traced_function takes tuples of parameters and names its symbols" begin
    function rober!(du, u, p, t)
        y₁, y₂, y₃ = u
        k₁, k₂, k₃ = p
        du[1] = -k₁ * y₁ + k₃ * y₂ * y₃
        du[2] = k₁ * y₁ - k₂ * y₂^2 - k₃ * y₂ * y₃
        du[3] = k₂ * y₂^2
        nothing
    end
    u0, p = [1.0, 0.5, 0.25], (0.04, 3e7, 1e4)
    f = CasADiCodegen.traced_function("rober", rober!, u0, p; states = [:y₁, :y₂, :y₃])
    du = zeros(3)
    rober!(du, u0, p, 0.0)
    @test build_function(f; expression = Val{false})[1](u0, collect(p), 0.0) ≈ du
    @test [casadi.name(e) for e in casadi.symvar(casadi.sx_in(f, 0))] == ["y₁", "y₂", "y₃"]
    # A named tuple gives the names of the parameters, as `p.k` gives the value.
    g = CasADiCodegen.traced_function("decay", (u, p) -> -p.k * u, [1.0], (k = 0.5,); inplace = false, time = false)
    @test casadi.name(casadi.sx_in(g, 1)) == "k" && build_function(g; expression = Val{false})[1]([2.0], [0.5]) ≈ [-1.0]
    @test_throws ArgumentError CasADiCodegen.traced_function("rober", rober!, u0, p; states = [:a, :b])
end

# Generic code writes a piecewise function as conditions times values; in Julia a `Bool`
# is a strong zero, so `false * Inf` is zero where CasADi gives `0 * Inf == NaN`.
@testset "a condition times a value is the value or zero" begin
    piecewise(x) = (0 <= x) * (x <= 300) * 30 * x + (300 < x) * (x <= 400) * 31 * x + (x < 0) * Inf + Inf * (x > 400)
    @testset "$S" for S in (SX, MX)
        x = S.sym("x", 1)
        y = untraced(piecewise(only(traced(x))))
        g, _ = build_function(CasadiFunction("f", [x], [y, jacobian(y, x)]); expression = Val{false})
        for v in (-1.0, 100.0, 350.0, 500.0)
            value, slope = g([v])
            @test isequal(value[1], piecewise(v))
            @test slope[1] == (0 <= v <= 300 ? 30.0 : 300 < v <= 400 ? 31.0 : 0.0)
        end
    end
end

@testset "a BigFloat constant enters the graph as a Float64" begin
    x = SX.sym("x", 1)
    g, _ = build_function(CasadiFunction("f", [x], [untraced(only(traced(x)) * big(0.25) + big(1))]); expression = Val{false})
    @test g([2.0]) == [1.5]
end

@testset "error functions of traced values" begin
    x = SX.sym("x", 1)
    t = only(traced(x))
    y = vcat(untraced(SpecialFunctions.erf(t)), untraced(SpecialFunctions.erfc(t)), untraced(SpecialFunctions.erfinv(t / 2)))
    g, _ = build_function(CasadiFunction("f", [x], [y, jacobian(y, x)]); expression = Val{false})
    for v in (-0.7, 0.3, 1.5)
        value, slope = g([v])
        @test value ≈ [SpecialFunctions.erf(v), SpecialFunctions.erfc(v), SpecialFunctions.erfinv(v / 2)] rtol = 1e-12
        @test slope ≈ ForwardDiff.derivative.([SpecialFunctions.erf, SpecialFunctions.erfc, z -> SpecialFunctions.erfinv(z / 2)], v) rtol = 1e-10
    end
end
