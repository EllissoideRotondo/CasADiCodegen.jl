# Generic Julia code traced into CasADi graphs with `Traced`.
import LinearAlgebra
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
    g, _ = build_function(CasadiFunction("s", [x], [vertcat(outputs...)]); expression = Val{false})
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

@testset "determinant, inverse and solve of a traced matrix" begin
    # LinearAlgebra factorizes with pivoting, which compares values; CasADi forms them from the graph.
    z = SX.sym("z", 4)
    A = reshape(traced(z), 2, 2)
    outputs = [LinearAlgebra.det(A); vec(LinearAlgebra.inv(A)); A \ [1.0, 2.0]]
    g, _ = build_function(CasadiFunction("m", [z], [untraced(outputs)]); expression = Val{false})
    M = [2.0 1.0; -1.0 3.0]
    @test g(vec(M)) ≈ [LinearAlgebra.det(M); vec(inv(M)); M \ [1.0, 2.0]]
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
    @test [C.name(e) for e in C.symvar(C.sx_in(f, 0))] == ["y₁", "y₂", "y₃"]
    # A named tuple gives the names of the parameters, as `p.k` gives the value.
    g = CasADiCodegen.traced_function("decay", (u, p) -> -p.k * u, [1.0], (k = 0.5,); inplace = false, time = false)
    @test C.name(C.sx_in(g, 1)) == "k" && build_function(g; expression = Val{false})[1]([2.0], [0.5]) ≈ [-1.0]
    @test_throws ArgumentError CasADiCodegen.traced_function("rober", rober!, u0, p; states = [:a, :b])
end
