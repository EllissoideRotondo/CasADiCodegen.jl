# CasADi functions with rules for ChainRulesCore.jl, so that an automatic
# differentiation of Julia code takes the derivatives that CasADi generates.
# The Jacobians of CasADi are the oracle.
import ChainRulesCore
using ChainRulesCore: NoTangent, ZeroTangent, Tangent, frule, rrule

@testset "a function with rules" begin
    x, P = MX.sym("x", 3), MX.sym("P", casadi.Sparsity.lower(2))
    y1, y2 = vcat(sin(x[1]) * x[2], x[3]^2 * P[1, 1]), casadi.mtimes(P, x[1:2]) * x[3]
    f = CasADiCodegen.Function("model", [x, P], [y1, y2])
    g = CasADiCodegen.differentiable(f)
    xv, Pv = [0.4, -1.2, 0.7], [1.5 0.0; -0.3 2.0]
    @test g(xv, Pv) == build_function(f; expression = Val{false})[1](xv, Pv)
    # The dense Jacobians: rows of y1 then of y2; columns of x, then of the four places of P.
    jac = CasADiCodegen.Function("jacobians", [x, P], [jacobian(vcat(y1, y2), x), jacobian(vcat(y1, y2), P)])
    J = hcat((reshape(dense_values(block), 4, :) for block in jac(casadi.DM(xv), casadi.DM(casadi.Sparsity.lower(2), casadi.DM([1.5, -0.3, 2.0]))))...)

    dx, dP = [0.3, 0.1, -0.5], [0.2 9.0; 0.7 -0.4]              # the entry above the diagonal has no effect
    value, change = frule((NoTangent(), dx, dP), g, xv, Pv)
    @test value == g(xv, Pv)
    @test vcat(change[1], change[2]) ≈ J * vcat(dx, vec(dP))
    _, by_state = frule((NoTangent(), dx, ZeroTangent()), g, xv, Pv)
    @test vcat(by_state[1], by_state[2]) ≈ J[:, 1:3] * dx

    value, back = rrule(g, xv, Pv)
    @test value == g(xv, Pv)
    seeds = ([0.6, -0.2], [1.1, 0.5])
    own, xbar, Pbar = back(Tangent{typeof(value)}(seeds...))
    @test own === NoTangent()
    @test vcat(xbar, vec(Pbar)) ≈ J' * vcat(seeds...)
    @test Pbar[1, 2] == 0
    _, xbar2, _ = back(Tangent{typeof(value)}(ZeroTangent(), ChainRulesCore.@thunk(seeds[2])))
    @test xbar2 ≈ J[3:4, 1:3]' * seeds[2]
    @test all(iszero, back(ZeroTangent())[2])
end

@testset "one argument and one result" begin
    x = MX.sym("x", 2)
    g = CasADiCodegen.differentiable(CasADiCodegen.Function("norm2", [x], [casadi.dot(x, x) * x]); force_SA = true)
    point = [0.5, -2.0]
    J = 2 * point * point' + (point' * point) * [1 0; 0 1]
    value, change = frule((NoTangent(), [1.0, 0.0]), g, point)
    @test value ≈ (point' * point) * point && change ≈ J[:, 1]
    @test rrule(g, point)[2]([0.0, 1.0])[2] ≈ J[2, :]
end

@testset "the code of a rule is made once" begin
    x = SX.sym("x", 2)
    g = CasADiCodegen.differentiable(CasADiCodegen.Function("twice", [x], [2 * x]))
    @test isempty(g.derivatives)
    forward = CasADiCodegen.Codegen.directional(g, :pushforward)
    @test CasADiCodegen.Codegen.directional(g, :pushforward) === forward && collect(keys(g.derivatives)) == [:pushforward]
end
