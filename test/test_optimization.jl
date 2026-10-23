# A solver of CasADi for optimization problems (`nlpsol`) in the graph of a
# function, replaced in the generated code by a solver of the SciML ecosystem
# that the user selects with `CasADiCodegen.Solvers`. The problems have exact
# solutions, which are the oracles: CasADi in this build cannot run its solvers.
import ForwardDiff
import SciMLBase
import OptimizationIpopt
import SparseArrays
using CasADiCodegen: Solvers, Solver
const K = CasADiCodegen.Codegen

native(f; kwargs...) = build_function(f; nonzeros = true, expression = Val{false}, expand = false, kwargs...)
evaluated(g, f, nz) = g([fill(NaN, n) for n in nnz_out(f)], Tuple(nz))
ipopt(options...) = OptimizationIpopt.IpoptOptimizer(additional_options = Dict{String,Any}("print_level" => 0, "tol" => 1e-12, options...))
const SOLVERS = Solvers(optimization = ipopt())

"The results of the solver of CasADi `solver` for the given arguments, by name."
solved(solver; arguments...) = solver(Dict{String,Any}(String(key) => value for (key, value) in arguments))

# The point of the line x1 + x2 = c that is nearest to q: the multiplier of the
# constraint is s = q1 + q2 - c, and the objective at the solution is s^2 / 2.
@testset "equality constraint" begin
    v, w = MX.sym("v", 2), MX.sym("w", 2)
    solver = C.nlpsol("nearest", "ipopt", Dict("x" => v, "p" => w, "f" => C.sumsqr(v - w), "g" => v[1] + v[2]))
    q, c = MX.sym("q", 2), MX.sym("c")
    r = solved(solver; x0 = MX(C.DM([0.0, 0.0])), p = q, lbg = c, ubg = c)
    f = CasADiCodegen.Function("f", [q, c], [r["x"], r["f"], r["g"], r["lam_x"], r["lam_g"], r["lam_p"],
                                      C.jacobian(r["x"], q), C.jacobian(r["x"], c), C.jacobian(r["f"], q), C.jacobian(r["lam_g"], c)])
    @test_throws "CasADiCodegen.Solvers(optimization = ...)" K.algorithm(f; expand = false)
    g = native(f; solvers = SOLVERS)
    values = [[0.7, -0.2], [2.0]]
    s = 0.7 - 0.2 - 2.0
    x, cost, constraint, lam_x, lam_g, lam_p, by_q, by_c, cost_by_q, lam_by_c = evaluated(g, f, values)
    @test x ≈ [0.7, -0.2] .- s / 2 rtol = 1e-9
    @test cost ≈ [s^2 / 2] rtol = 1e-9
    @test constraint ≈ [2.0] rtol = 1e-9
    @test lam_x == [0.0, 0.0]
    @test lam_g ≈ [s] rtol = 1e-8
    # The multiplier of the parameters is the slope of the best objective by them.
    @test lam_p ≈ [s, s] rtol = 1e-8
    @test cost_by_q ≈ [s, s] rtol = 1e-7
    @test reshape(by_q, 2, 2) ≈ [0.5 -0.5; -0.5 0.5] rtol = 1e-7
    @test by_c ≈ [0.5, 0.5] rtol = 1e-7
    @test lam_by_c ≈ [-1.0] rtol = 1e-7
    @testset "a written module has the solver" begin
        text = sprint(io -> generate_julia(f, io; module_name = :Written, name = :run!, expand = false, solvers = SOLVERS))
        src = Module()
        Base.include_string(src, text)
        written = Base.invokelatest((r, a) -> src.Written.run_nonzeros!(r, a), [fill(NaN, n) for n in nnz_out(f)], Tuple(values))
        @test written[1] ≈ x && written[7] ≈ by_q
    end
    @testset "numbers that are not Float64 are refused" begin
        @test_throws "takes floating-point numbers only" g([zeros(ForwardDiff.Dual{Nothing,Float64,1}, n) for n in nnz_out(f)], (ForwardDiff.Dual.([0.7, -0.2], 1.0), [2.0]))
    end
    @testset "a solver that fails" begin
        # The solution alone keeps the compilation short.
        solution = CasADiCodegen.Function("f", [q, c], [r["x"]])
        short = native(solution; solvers = Solvers(optimization = ipopt("max_iter" => 0)))
        @test_throws "solver \"nearest\": the solver stopped with the return code" evaluated(short, solution, values)
    end
end

# A solver that takes first derivatives only gets no code for the Hessians: their dense
# form has one n×n matrix per constraint, which for a large problem cannot be built.
@testset "a solver of the first order gets no Hessians" begin
    v, w = MX.sym("v", 2), MX.sym("w", 2)
    solver = C.nlpsol("nearest", "ipopt", Dict("x" => v, "p" => w, "f" => C.sumsqr(v - w), "g" => v[1] + v[2]))
    first_order = ipopt("hessian_approximation" => "limited-memory")
    functions = K.optimization_functions(K.optimization_model(solver); hessians = K.needs_hessians(first_order))
    @test functions.hess === nothing && functions.cons_h === nothing
    q, c = MX.sym("q", 2), MX.sym("c")
    r = solved(solver; x0 = MX(C.DM([0.0, 0.0])), p = q, lbg = c, ubg = c)
    f = CasADiCodegen.Function("f", [q, c], [r["x"], r["lam_g"]])
    x, lam_g = evaluated(native(f; solvers = Solvers(optimization = first_order)), f, [[0.7, -0.2], [2.0]])
    @test x ≈ [1.45, 0.55] rtol = 1e-8
    @test lam_g ≈ [0.7 - 0.2 - 2.0] rtol = 1e-6
end

# The same objective with an upper bound u on x1 below q1: the bound is active
# with the multiplier 2 (q1 - u), and the constraint x1 + x2 <= 100 is not.
@testset "active bound and constraint that is not active" begin
    v, w = MX.sym("v", 2), MX.sym("w", 2)
    solver = C.nlpsol("bounded", "ipopt", Dict("x" => v, "p" => w, "f" => C.sumsqr(v - w), "g" => v[1] + v[2]))
    q, u = MX.sym("q", 2), MX.sym("u")
    r = solved(solver; x0 = MX(C.DM([0.0, 0.0])), p = q, ubx = C.vertcat(MX[u, MX(Inf)]), ubg = MX(100.0))
    f = CasADiCodegen.Function("f", [q, u], [r["x"], r["lam_x"], r["lam_g"], C.jacobian(r["x"], q), C.jacobian(r["x"], u)])
    x, lam_x, lam_g, by_q, by_u = evaluated(native(f; solvers = SOLVERS), f, [[0.7, -0.2], [0.3]])
    @test x ≈ [0.3, -0.2] rtol = 1e-7
    @test lam_x[1] ≈ 0.8 rtol = 1e-6
    @test lam_x[2] == 0.0 && lam_g == [0.0]
    @test reshape(by_q, 2, 2) ≈ [0.0 0.0; 0.0 1.0] atol = 1e-7
    @test by_u ≈ [1.0, 0.0] atol = 1e-7
end

@testset "multipliers from the active bounds" begin
    multipliers = Base.get_extension(CasADiCodegen, :CasADiCodegenSciMLBaseExt).multipliers
    J, none = zeros(0, 2), Float64[]
    # gradient + lam_x = 0: positive at the upper bound, negative at the lower one, zero away from the bounds.
    @test multipliers([-0.8, 0.5, 0.3], zeros(0, 3), [1.0, 2.0, 0.0], [-Inf, 2.0, -1.0], [1.0, Inf, 1.0], none, none, none) == ([0.8, -0.5, 0.0], none)
    # A bound that the objective pulls away from, or does not push on, is not active.
    @test multipliers([0.8, 0.0], J, [1.0, 1.0], [-Inf, -Inf], [1.0, 1.0], none, none, none) == ([0.0, 0.0], none)
    # An unknown with one number for its two bounds takes a multiplier of any sign.
    @test multipliers([0.8, 0.0], J, [1.0, 1.0], [1.0, -Inf], [1.0, Inf], none, none, none) == ([-0.8, 0.0], none)
    # A constraint at its bound shares the gradient with the bound of an unknown.
    lam_x, lam_g = multipliers([-1.0, -3.0], [1.0 1.0], [1.0, 1.0], [-Inf, -Inf], [Inf, 1.0], [2.0], [-Inf], [2.0])
    @test lam_x ≈ [0.0, 2.0] && lam_g ≈ [1.0]
end

@testset "the problem of the solver, solved by the user" begin
    v, w = MX.sym("v", 2), MX.sym("w", 2)
    solver = C.nlpsol("nearest", "ipopt", Dict("x" => v, "p" => w, "f" => C.sumsqr(v - w), "g" => v[1] + v[2]))
    problem = SciMLBase.OptimizationProblem(solver, [0.0, 0.0], [0.7, -0.2]; lbg = [2.0], ubg = [2.0])
    # The function holds its derivatives, so Optimization.jl logs no warning about automatic differentiation.
    solution = @test_logs SciMLBase.solve(problem, ipopt())
    @test SciMLBase.successful_retcode(solution.retcode)
    @test solution.u ≈ [1.45, 0.55] rtol = 1e-9
    # Ipopt takes the Hessian of the Lagrange function of the sparse form, which has no Hessians of the constraints.
    sparse = SciMLBase.OptimizationProblem(solver, [0.0, 0.0], [0.7, -0.2]; lbg = [2.0], ubg = [2.0], sparse = true)
    @test sparse.f.cons_h === nothing
    @test SciMLBase.solve(sparse, ipopt()).u ≈ [1.45, 0.55] rtol = 1e-9
    # Without parameters Ipopt calls lag_h without them. The minimum of the Rosenbrock function is at (1, 1).
    z = SX.sym("z", 2)
    rosenbrock = CasADiCodegen.Function("rosenbrock", [z], [(1 - z[1])^2 + 100 * (z[2] - z[1]^2)^2])
    free = SciMLBase.OptimizationProblem(SciMLBase.OptimizationFunction(rosenbrock; sparse = true), [-1.2, 1.0])
    @test SciMLBase.solve(free, ipopt()).u ≈ [1.0, 1.0] rtol = 1e-9
end

# The extension for OptimizationBase.jl gives a solver the lag_h(h, x, σ, μ[, p]) of a SciML function of
# this package. The objective q1 x1² + q2 x2² - 2 x1 - 2 x2 has a Hessian that depends on q; on the line
# x1 + x2 = 2 its least value is at x = 2 w / (w1 + w2) with w = 1 ./ q, and without the constraint at w.
@testset "lag_h of the problem in OptimizationBase.jl" begin
    ext = Base.get_extension(CasADiCodegen, :CasADiCodegenOptimizationBaseExt)
    OptimizationBase = OptimizationIpopt.OptimizationBase
    # The extension rebuilds an instantiated function from these keywords; a release of SciMLBase.jl that
    # changes the fields of OptimizationFunction must fail here, not in the solve of a user.
    @test Set(fieldnames(SciMLBase.OptimizationFunction)) == Set((:f, :adtype, ext.KEYWORDS...))
    ambiguous = Test.detect_ambiguities(ext, OptimizationBase, SciMLBase, OptimizationIpopt)
    @test isempty(filter(pair -> any(m -> m.module === ext, pair), ambiguous))
    x, q = SX.sym("x", 2), SX.sym("q", 2)
    objective = CasADiCodegen.Function("objective", [x, q], [q[1] * x[1]^2 + q[2] * x[2]^2 - 2 * x[1] - 2 * x[2]])
    line = CasADiCodegen.Function("line", [x, q], [x[1] + x[2]])
    f = SciMLBase.OptimizationFunction(objective, line; sparse = true)
    @test f isa ext.WithLagrangeHessian && !(SciMLBase.OptimizationFunction(objective, line) isa ext.WithLagrangeHessian)
    nearest(q) = 2 ./ q ./ sum(1 ./ q)
    on_line(p) = SciMLBase.OptimizationProblem(f, [0.0, 0.0], p; lcons = [2.0], ucons = [2.0])
    problem = on_line([2.0, 4.0])
    @test SciMLBase.solve(problem, ipopt()).u ≈ nearest([2.0, 4.0]) rtol = 1e-9
    # A solve with new parameters takes the steps of a new problem with them, which a stale Hessian would change.
    same(solution, p) = (fresh = SciMLBase.solve(on_line(p), ipopt()); solution.u == fresh.u && solution.stats.iterations == fresh.stats.iterations)
    problem.p .= [1.0, 3.0]
    @test same(SciMLBase.solve(problem, ipopt()), [1.0, 3.0])
    @test same(SciMLBase.solve(SciMLBase.remake(problem; p = [3.0, 0.5]), ipopt()), [3.0, 0.5])
    cache = SciMLBase.init(problem, ipopt())
    SciMLBase.solve!(cache)
    SciMLBase.reinit!(cache; p = [0.5, 2.0])
    @test same(SciMLBase.solve!(cache), [0.5, 2.0])
    problem.p .= [4.0, 1.0]
    @test same(SciMLBase.solve!(SciMLBase.init(problem, ipopt())), [4.0, 1.0])
    # OptimizationMOI.jl calls grad(G, x) and lag_h(h, x, σ, μ) without p after a reinit!, which replaces the p of the cache.
    reinit = OptimizationBase.ReInitCache([0.0, 0.0], [2.0, 4.0])
    instantiated = OptimizationBase.instantiate_function(problem.f, reinit, problem.f.adtype, 1)
    reinit.p = [3.0, 0.5]
    without, with, gradient = zeros(2), zeros(2), zeros(2)
    instantiated.lag_h(without, [0.3, 0.1], 1.0, [0.5])
    instantiated.lag_h(with, [0.3, 0.1], 1.0, [0.5], [3.0, 0.5])
    instantiated.grad(gradient, [0.3, 0.1])
    @test without == with == [6.0, 1.0]
    @test gradient ≈ [-0.2, -1.9]
    # With no constraint, the solver calls lag_h with no multiplier μ.
    free = SciMLBase.OptimizationProblem(SciMLBase.OptimizationFunction(objective; sparse = true), [0.0, 0.0], [2.0, 4.0])
    @test SciMLBase.solve(free, ipopt()).u ≈ [0.5, 0.25] rtol = 1e-9
end

# x^4 / 4 - q x has its least value at the cube root of q.
@testset "no constraint, and derivatives of the second order" begin
    v, w = MX.sym("v"), MX.sym("w")
    solver = C.nlpsol("cube", "ipopt", Dict("x" => v, "p" => w, "f" => v^4 / 4 - w * v))
    q = MX.sym("q")
    x = solved(solver; x0 = MX(1.0), p = q)["x"]
    f = CasADiCodegen.Function("f", [q], [x, C.jacobian(x, q), C.jacobian(C.jacobian(x, q), q)])
    root, slope, curve = evaluated(native(f; solvers = SOLVERS), f, [[2.0]])
    @test root ≈ [cbrt(2.0)] rtol = 1e-9
    @test slope ≈ [2.0^(-2 / 3) / 3] rtol = 1e-7
    @test curve ≈ [-2 / 9 * 2.0^(-5 / 3)] rtol = 1e-6
end

@testset "no parameter, and one solver for one function" begin
    v = MX.sym("v", 2)
    solver = C.nlpsol("plain", "ipopt", Dict("x" => v, "f" => C.sumsqr(v - C.DM([3.0, -1.0]))))
    a = MX.sym("a")
    f = CasADiCodegen.Function("f", [a], [a * solved(solver; x0 = MX(C.DM([0.0, 0.0])))["x"]])
    solvers = Solvers(overrides = (Solver(solver, ipopt()),))
    @test evaluated(native(f; solvers), f, [[2.0]])[1] ≈ [6.0, -2.0] rtol = 1e-8
end

# A solver of CasADi for quadratic problems (`conic`) is a problem with the
# objective x' H x / 2 + g' x and the constraints A x. CasADi has no derivatives
# of such a solver, and its own solver `qrqp` is the oracle for the values.
@testset "solver of quadratic problems" begin
    quiet = Dict("print_iter" => false, "print_header" => false)
    solver = C.conic("quadratic", "qrqp", Dict("h" => C.Sparsity.diag(2), "a" => C.Sparsity.dense(1, 2)), quiet)
    d, q, c, u = MX.sym("d", C.Sparsity.diag(2)), MX.sym("q", 2), MX.sym("c"), MX.sym("u")
    # The upper bound `u` of the first unknown is active for the values below.
    r = solved(solver; h = d, g = q, a = MX(C.DM([1.0 1.0])), lba = c, uba = c, ubx = C.vertcat(MX[u, MX(Inf)]))
    f = CasADiCodegen.Function("f", [d, q, c, u], [r["x"], r["cost"], r["lam_a"], r["lam_x"]])
    @test_throws "CasADiCodegen.Solvers(optimization = ...)" K.algorithm(f; expand = false)
    values = [[2.0, 4.0], [-1.4, 0.4], [2.0], [0.5]]
    expected = [vec(Array(C.DM(v))) for v in f(C.DM([2.0 0.0; 0.0 4.0]), map(C.DM, values[2:4])...)]
    x, cost, lam_a, lam_x = evaluated(native(f; solvers = SOLVERS), f, values)
    # The interior point method keeps a distance of about 1e-8 from the active bound.
    @test x ≈ [0.5, 1.5] rtol = 1e-7
    @test x ≈ expected[1] rtol = 1e-7
    @test cost ≈ expected[2] rtol = 1e-7
    @test lam_a ≈ expected[3] rtol = 1e-7
    @test lam_x ≈ expected[4] atol = 1e-7
    @testset "the problem of the solver, solved by the user" begin
        # The parameters are the nonzeros of h, then g, then the nonzeros of a.
        problem = SciMLBase.OptimizationProblem(solver, [0.0, 0.0], [2.0, 4.0, -1.4, 0.4, 1.0, 1.0]; lbg = [2.0], ubg = [2.0], ubx = [0.5, Inf])
        @test SciMLBase.solve(problem, ipopt()).u ≈ [0.5, 1.5] rtol = 1e-7
    end
end
