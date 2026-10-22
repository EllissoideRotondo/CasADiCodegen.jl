# Roots of equations found by a SciML solver inside CasADi graphs and inside
# the generated code, with derivatives from the implicit function theorem.
import ForwardDiff
using SimpleNonlinearSolve: SimpleNewtonRaphson
using StaticArrays: SVector, SMatrix

"The eccentric anomaly of the equation of Kepler, E - e sin(E) = M, by the iteration of Newton."
function eccentric_anomaly(M, e)
    E = M
    for _ in 1:50
        E -= (E - e * sin(E) - M) / (1 - e * cos(E))
    end
    E
end

@testset "root of one equation with its derivatives" begin
    E, M, e = SX.sym("E"), SX.sym("M"), SX.sym("e")
    residual = CasADiCodegen.Function("kepler", [E, M, e], [E - e * sin(E) - M])
    R = rootfinder("anomaly", SimpleNewtonRaphson(), residual; abstol = 1e-14)
    @test (C.n_in(R), C.n_out(R)) == (3, 1)
    guess, m, ec = MX.sym("guess"), MX.sym("m"), MX.sym("ec")
    root = R(guess, m, ec)
    first_order = [jacobian(root, v) for v in (guess, m, ec)]
    f = CasADiCodegen.Function("f", [guess, m, ec], [root, first_order..., jacobian(first_order[2], m), jacobian(first_order[2], ec)])
    Mv, ev = 1.0, 0.3
    Ev = eccentric_anomaly(Mv, ev)
    d = 1 - ev * cos(Ev)
    expected = [[Ev], [0.0], [1 / d], [sin(Ev) / d], [-ev * sin(Ev) / d^3], [(cos(Ev) - ev) / d^3]]
    g = build_function(f; nonzeros = true, expression = Val{false})
    blank() = [zeros(C.nnz_out(f, j)) for j in 0:5]
    ours = g(blank(), ([Mv], [Mv], [ev]))
    @test all(length(a) == length(b) && all(isapprox.(a, b; atol = 1e-11)) for (a, b) in zip(ours, expected))
    @test all(all(isapprox.(a, b; atol = 1e-11)) for (a, b) in zip(casadi_eval(f, Mv, Mv, ev), expected))
    # Matrix code.
    h = build_function(f; nonzeros = true, expression = Val{false}, expand = false)
    @test all(all(isapprox.(a, b; atol = 1e-11)) for (a, b) in zip(h(blank(), ([Mv], [Mv], [ev])), expected))
    # The in-place function on arrays and the function on nonzeros allocate no memory, also
    # for the root and one derivative alone (regression: 96 bytes per call).
    _, inplace = build_function(f; expression = Val{false})
    out, arguments = Tuple(zeros(1, 1) for _ in 1:6), ([Mv], [Mv], [ev])
    inplace(out, arguments)
    @test (@allocated inplace(out, arguments)) == 0
    nonzeros = blank()
    g(nonzeros, arguments)
    @test (@allocated g(nonzeros, arguments)) == 0
    _, small! = build_function(CasADiCodegen.Function("small", [guess, m, ec], [root, first_order[2]]); expression = Val{false})
    small_out = (zeros(1), zeros(1))
    small!(small_out, arguments)
    @test (@allocated small!(small_out, arguments)) == 0
    # Dual numbers pass through the solver.
    value(m) = (r = [zeros(typeof(m), C.nnz_out(f, j)) for j in 0:5]; g(r, ([m], [m], [ev])); r[1][1])
    @test isapprox(ForwardDiff.derivative(value, Mv), 1 / d; atol = 1e-10)
end

@testset "root of a system, with a matrix parameter" begin
    z, A, b = SX.sym("z", 2), SX.sym("A", 2, 2), SX.sym("b", 2)
    residual = CasADiCodegen.Function("r", [z, A, b], [mtimes(A, z) + vertcat([z[1]^3, 0]) - b])
    R = rootfinder("sys", SimpleNewtonRaphson(), residual; abstol = 1e-13)
    z0, Am, bv = MX.sym("z0", 2), MX.sym("A", 2, 2), MX.sym("b", 2)
    root = R(z0, Am, bv)
    f = CasADiCodegen.Function("f", [z0, Am, bv], [root, jacobian(root, bv), jacobian(root, Am)])
    Av, bn = [2.0 0.5; 0.3 1.5], [1.0, 2.0]
    ours = build_function(f; expression = Val{false})[1]([0.0, 0.0], Av, bn)
    zv = ours[1]
    @test maximum(abs, Av * zv + [zv[1]^3, 0] - bn) < 1e-11
    slope = inv(Av + [3 * zv[1]^2 0; 0 0])                 # dz/db
    @test isapprox(ours[2], slope; atol = 1e-10)
    @test isapprox(ours[3], -slope * kron(zv', [1.0 0; 0 1.0]); atol = 1e-10)
end

# Above 16 unknowns the SciML problem is on vectors, with the Jacobian of the generated code.
@testset "root of a large system, on vectors" begin
    n = 40
    z, q = SX.sym("z", n), SX.sym("q")
    at(i) = 1 <= i <= n ? z[i] : SX(0.0)
    residual = CasADiCodegen.Function("bratu", [z, q], [vertcat([at(i - 1) - 2 * z[i] + at(i + 1) + q * exp(z[i]) / (n + 1)^2 for i in 1:n]...)])
    R = rootfinder("bratu", SimpleNewtonRaphson(), residual; abstol = 1e-12)
    newton = C.rootfinder("bratu_newton", "newton", residual)
    x, λ = MX.sym("x", n), MX.sym("λ")
    ours = CasADiCodegen.Function("ours", [x, λ], [R(x, λ), jacobian(R(x, λ), λ)])
    theirs = CasADiCodegen.Function("theirs", [x, λ], [newton(x, λ), jacobian(newton(x, λ), λ)])
    values = (zeros(n), [2.0])
    g = build_function(ours; nonzeros = true, expression = Val{false})
    @test all(isapprox.(g([zeros(n), zeros(n)], values), casadi_eval(theirs, values...); atol = 1e-9))     # each stops at a residual of 1e-12
end

@testset "a rootfinder made by hand" begin
    # The iteration of Newton with the emitted residual and its emitted Jacobian.
    newton(residual, slope, n) = function (z0, parameters...)
        z = SVector{n}(z0)
        for _ in 1:30
            z -= SMatrix{n,n}(slope(z, parameters...)) \ residual(z, parameters...)
        end
        z
    end
    E, M, e = SX.sym("E"), SX.sym("M"), SX.sym("e")
    R = CasADiCodegen.Codegen.root_function("hand", CasADiCodegen.Function("kepler", [E, M, e], [E - e * sin(E) - M]), newton)
    m, ec = MX.sym("m"), MX.sym("ec")
    f = CasADiCodegen.Function("f", [m, ec], [R(m, m, ec), jacobian(R(m, m, ec), ec)])
    Ev = eccentric_anomaly(1.0, 0.3)
    ours = build_function(f; nonzeros = true, expression = Val{false})([zeros(1), zeros(1)], ([1.0], [0.3]))
    @test isapprox(ours[1][1], Ev; atol = 1e-13) && isapprox(ours[2][1], sin(Ev) / (1 - 0.3 * cos(Ev)); atol = 1e-12)
    # A residual that CasADi cannot expand: the derivative keeps a linear solve.
    shift = callback(v -> [v[1] + 1], "shift", (1,), (1,); jacobian = v -> [1.0;;])
    z, q = MX.sym("z", 1), MX.sym("q", 1)
    S = CasADiCodegen.Codegen.root_function("opaque", CasADiCodegen.Function("r", [z, q], [shift(z) * 2 - q]), newton)
    g = CasADiCodegen.Function("g", [q], [S(q, q), jacobian(S(q, q), q)])
    @test build_function(g; nonzeros = true, expression = Val{false})([zeros(1), zeros(1)], ([5.0],)) == [[1.5], [0.5]]
end

@testset "limits of the rootfinder" begin
    z, p = SX.sym("z", 2), SX.sym("p")
    @test_throws ArgumentError rootfinder("not a name", SimpleNewtonRaphson(), CasADiCodegen.Function("r", [z, p], [z * p]))
    @test_throws ArgumentError rootfinder("r", SimpleNewtonRaphson(), CasADiCodegen.Function("r", [z, p], [z[1] * p]))          # one equation, two unknowns
    @test_throws ArgumentError rootfinder("r", SimpleNewtonRaphson(), CasADiCodegen.Function("r", [z, p], [z * p, z]))         # two results
    Z = SX.sym("Z", 2, 2)
    @test_throws ArgumentError rootfinder("r", SimpleNewtonRaphson(), CasADiCodegen.Function("r", [Z], [Z]))                   # a matrix of unknowns
    # An equation with no root: the error names the rootfinder and the return code of the solver.
    none = rootfinder("none", SimpleNewtonRaphson(), CasADiCodegen.Function("r", [p], [p^2 + 1]); maxiters = 20)
    w = MX.sym("w")
    g = build_function(CasADiCodegen.Function("f", [w], [none(w)]); nonzeros = true, expression = Val{false})
    err = try g([zeros(1)], ([0.5],)); nothing catch e; e end
    @test err isa ErrorException && occursin("rootfinder \"none\"", err.msg)
end
