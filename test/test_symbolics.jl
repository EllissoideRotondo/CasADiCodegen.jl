# Symbolics.jl expressions translated into CasADi functions by `casadi_function`.
import Symbolics
using Symbolics: @variables, @register_symbolic, Differential

@variables x y p[1:2] q[1:2, 1:2]

# The value of a Symbolics expression at `point`, a Dict from its symbols to numbers, from the code that Symbolics builds.
reference(expr, point) = Symbolics.build_function(expr, collect(keys(point)); expression = Val{false})(collect(values(point)))

@testset "scalar expression, its Jacobian and compiled code" begin
    expr = sin(x) * y^2 + exp(-x) / (1 + y^2) + abs(x - y)
    f = casadi_function("f", [x, y], [expr, Symbolics.jacobian([expr], [x, y])])
    point = Dict(x => 0.3, y => -1.2)
    value, jac = casadi_eval(f, 0.3, -1.2)
    @test only(value) ≈ reference(expr, point)
    @test jac ≈ [reference(d, point) for d in Symbolics.gradient(expr, [x, y])]
    g = build_function(f; nonzeros = true, expression = Val{false})
    @test g([zeros(1), zeros(2)], ([0.3], [-1.2]))[1] ≈ [reference(expr, point)]
end

@testset "array inputs and outputs keep their shape and names" begin
    mat = [p[1]*q[1, 1]+q[2, 1] p[2]^2; q[1, 2] p[1]*p[2]*q[2, 2]]
    f = casadi_function("m", [p, q], [mat, p[1] + p[2]]; input_names = ["p", "q"], output_names = ["m", "s"])
    @test (C.size1(C.sparsity_out(f, 0)), C.size2(C.sparsity_out(f, 0))) == (2, 2)
    @test C.name_in(f, 1) == "q" && C.name_out(f, 1) == "s"
    pv, qv = [0.7, -1.1], [0.5 2.0; -0.4 1.3]
    result = f(C.DM(pv), C.DM(qv))
    @test Matrix(result[1]) ≈ [pv[1]*qv[1, 1]+qv[2, 1] pv[2]^2; qv[1, 2] pv[1]*pv[2]*qv[2, 2]]
    @test only(C.nonzeros(result[2])) ≈ sum(pv)
end

@testset "supported operations agree with Julia" begin
    unary = [sin, cos, tan, asin, acos, atan, sinh, cosh, tanh, asinh, exp, log, log2, log10, sqrt, abs, sign, floor, ceil, expm1, log1p, inv]
    @testset "$f" for f in unary
        point = f in (acos, asin) ? 0.3 : 0.6
        h = casadi_function("h", [x], [f(x)])
        @test only(casadi_eval(h, point)[1]) ≈ f(point)
    end
    binary = [+, -, *, /, ^, min, max, atan, hypot]
    @testset "$f" for f in binary
        h = casadi_function("h", [x, y], [f(x, y)])
        @test only(casadi_eval(h, 0.7, 1.9)[1]) ≈ f(0.7, 1.9)
    end
end

@testset "piecewise expressions" begin
    expr = ifelse(x < y, x^2, ifelse((x >= 1) & (y != 0), -x, 5.0))
    f = casadi_function("pw", [x, y], [expr])
    @testset "x = $a, y = $b" for (a, b) in ((0.5, 2.0), (3.0, 1.0), (0.5, 0.2))
        @test only(casadi_eval(f, a, b)[1]) ≈ reference(expr, Dict(x => a, y => b))
    end
end

@testset "constants" begin
    f = casadi_function("c", [x], [(1 // 2) * x + 3, 2 + 0 * x])
    @test casadi_eval(f, 4.0) == [[5.0], [2.0]]
end

@testset "expressions that cannot be translated are rejected with their name" begin
    @variables t
    @register_symbolic unknown(x)
    @test_throws ArgumentError("casadi_function: the symbol y is not an input") casadi_function("f", [x], [x + y])
    @test_throws r"unknown" casadi_function("f", [x], [unknown(x)])
    @test_throws r"Differential" casadi_function("f", [x, t], [Differential(t)(x)])
    @test_throws r"Complex" casadi_function("f", [x], [x + 2im])
end
