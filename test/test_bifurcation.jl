# CasADi functions as the vector fields of bifurcation problems of BifurcationKit.jl.
import SciMLBase
import BifurcationKit as BK

# The normal form of a pitchfork, μ x - x³ + α y = 0, -y = 0, with the observed variable x².
x, y, μ, α = SX.sym("x"), SX.sym("y"), SX.sym("μ"), SX.sym("α")
pitchfork = CasadiFunction("pitchfork", [vertcat(x, y), vertcat(μ, α)], [vertcat(μ * x - x^3 + α * y, -y), x^2],
                           ["u", "p"], ["residual", "square"])
pitchfork_problem = BK.BifurcationProblem(pitchfork, [:y => 0.5, :x => 1.0], [:μ => -1.0, :α => 1.0], :μ;
                                          record_from_solution = (:x, :μ, :square))

@testset "BifurcationProblem from a CasADi function by names" begin
    pb = pitchfork_problem
    @test BK.getu0(pb) == [1.0, 0.5] && BK.getparams(pb) == [-1.0, 1.0] && BK.getparam(pb) == -1.0
    u, p = [0.3, 0.2], [0.5, 2.0]
    d1, d2, d3 = [1.0, 2.0], [0.5, -1.0], [2.0, 1.0]
    @test BK.residual(pb, u, p) ≈ [0.5 * 0.3 - 0.3^3 + 2.0 * 0.2, -0.2]
    @test BK.jacobian(pb, u, p) ≈ [0.5-3*0.3^2 2.0; 0.0 -1.0]
    # The derivatives of the jet against the derivatives by hand, -6x for the second and -6 for the third.
    @test BK.dF(pb, u, p, d1) ≈ BK.jacobian(pb, u, p) * d1
    @test BK.d2F(pb, u, p, d1, d2) ≈ [-6 * 0.3 * 1.0 * 0.5, 0.0]
    @test BK.d3F(pb, u, p, d1, d2, d3) ≈ [-6 * 1.0 * 0.5 * 2.0, 0.0]
    @test BK.R01(pb, u, p) ≈ [0.3, 0.0] && BK.R11(pb, u, p, d1) ≈ [1.0, 0.0]
    # Complex directions, as for a Hopf point, give the complex bilinear form.
    @test BK.d2F(pb, u, p, d1 .+ im .* d2, d2) ≈ BK.d2F(pb, u, p, d1, d2) .+ im .* BK.d2F(pb, u, p, d2, d2)
    # The record holds a state, a parameter and an observed variable by name, at the value of the
    # continued parameter that BifurcationKit.jl passes.
    @test BK.record_from_solution(pb, u, 0.25) == (x = 0.3, μ = 0.25, square = 0.3^2)
end

@testset "branch point and Hopf point of a continuation" begin
    options = BK.ContinuationPar(p_min = -1.0, p_max = 1.0, nev = 2)
    branch = BK.continuation(pitchfork_problem, BK.PALC(), options)
    points = filter(point -> point.type == :bp, branch.specialpoint)
    @test length(points) == 1 && abs(only(points).param) < 1e-3
    @test propertynames(branch.branch) ⊇ (:x, :μ, :square) && branch.branch.μ ≈ branch.param
    # The right side f(u, p, t) of an ODE without the time, in MX symbols: a Hopf point at μ = 0.
    u, ν, t = MX.sym("u", 2), MX.sym("ν"), MX.sym("t")
    r2 = u[1]^2 + u[2]^2
    rhs = CasadiFunction("hopf", [u, ν, t], [vertcat(ν * u[1] - u[2] - u[1] * r2, u[1] + ν * u[2] - u[2] * r2)])
    hopf = BK.BifurcationProblem(rhs, [0.0, 0.0], [-1.0], :ν; record_from_solution = :u_0)
    @test BK.d2F(hopf, [0.1, 0.2], [0.3], [1.0, 0.0], [0.0, 1.0]) ≈ [-2 * 0.2, -2 * 0.1]
    branch = BK.continuation(hopf, BK.PALC(), options)
    points = filter(point -> point.type == :hopf, branch.specialpoint)
    @test length(points) == 1 && abs(only(points).param) < 1e-3
end

@testset "BifurcationProblem checks its arguments" begin
    t = SX.sym("t")
    @test_throws "does not depend on the time" BK.BifurcationProblem(CasadiFunction("forced", [x, μ, t], [μ * x - t]), [0.0], [0.0], 1)
    @test_throws "no value to the elements [2]" BK.BifurcationProblem(pitchfork, [:x => 1.0], [-1.0, 1.0], :μ)
    @test_throws "not a parameter" BK.BifurcationProblem(pitchfork, [1.0, 0.5], [-1.0, 1.0], :x)
end
