# Nodes and functions of CasADi that have no instruction list: a B-spline, an
# interpolant, a tensor product, an assertion. CasADi keeps their data to
# itself, so the package reads it from the C code that CasADi generates, and
# the generated Julia code does not call CasADi. CasADi is the oracle.
import ForwardDiff
const K = CasADiCodegen.Codegen

node_ops(f) = Set(ins.op for ins in K.MXAlgorithm(f).instructions)
native(f; kwargs...) = build_function(f; nonzeros = true, expression = Val{false}, expand = false, kwargs...)
evaluated(g, f, nz) = g([fill(NaN, n) for n in nnz_out(f)], Tuple(nz))
agree(a, e; rtol = 1e-12) = length(a) == length(e) && all(isapprox(x, y; rtol, atol = 1e-13, nans = true) for (x, y) in zip(a, e))
all_agree(actual, expected; kwargs...) = all(agree(a, e; kwargs...) for (a, e) in zip(actual, expected))

"The function as a written module, evaluated on one vector of nonzeros per input: no part of it can call CasADi."
function written(f, nz; kwargs...)
    text = sprint(io -> generate_julia(f, io; module_name = :Written, name = :run!, expand = false, kwargs...))
    src = Module()
    Base.include_string(src, text)
    Base.invokelatest((r, a) -> src.Written.run_nonzeros!(r, a), [fill(NaN, n) for n in nnz_out(f)], Tuple(nz))
end

"Captured standard output of `run()`."
function printed(run)
    text = ""
    mktemp() do path, io
        redirect_stdout(run, io)
        close(io)
        text = read(path, String)
    end
    text
end

# The forms of a number in the C code of CasADi that no model of these tests holds.
@testset "numbers in the C code of CasADi" begin
    @test K.c_real("casadi_inf") == Inf && K.c_real("-INF") == -Inf && isnan(K.c_real("NAN")) && K.c_real("5.") == 5.0
end

# Points in a knot span, at a knot, at both ends, and out of the range.
const SPLINE_POINTS = [[0.4, 0.3], [1.0, 1.0], [0.0, 0.0], [3.0, 2.0], [2.5, 1.7], [2.9, 0.01], [-0.5, 1.0], [1.0, 2.5], [0.0, 2.0], [1.7, 1.2]]
const SPLINE_KNOTS = [[0.0, 0, 0, 1, 2.5, 3, 3, 3], [0.0, 0, 1, 2, 2]]       # degrees 2 and 1: 5 and 3 basis functions

@testset "B-spline node" begin
    x, c = MX.sym("x", 2), MX.sym("c", 30)
    coefficients = [sin(0.7k) + 0.1k for k in 1:30]
    @testset "fixed coefficients, lookup $mode" for mode in ("linear", "binary")
        options = Dict("lookup_mode" => [mode, mode])
        value = C.bspline(x, C.DM(coefficients), SPLINE_KNOTS, [2, 1], 2, options)
        f = CasADiCodegen.Function("spline", [x], [value, C.jacobian(value, x)])
        @test :OP_BSPLINE in node_ops(f)
        g = native(f)
        @test all(all_agree(evaluated(g, f, [p]), mx_casadi(f, [p])) for p in SPLINE_POINTS)
        mode == "binary" && continue        # the lookup mode changes the search only
        @test all(all_agree(written(f, [p]), mx_casadi(f, [p])) for p in SPLINE_POINTS[1:2])
        results, arguments = [zeros(2), zeros(4)], ([0.4, 0.3],)
        g(results, arguments)
        @test @allocated(g(results, arguments)) == 0
        # Dual numbers pass through the generated code.
        J = ForwardDiff.jacobian(p -> g([similar(p, 2), similar(p, 4)], (p,))[1], [0.4, 0.3])
        @test agree(vec(J), mx_casadi(f, [[0.4, 0.3]])[2])
    end
    @testset "coefficients that are an argument" begin
        value = C.bspline(x, c, SPLINE_KNOTS, [2, 1], 2, Dict())
        f = CasADiCodegen.Function("spline", [x, c], [value, C.jacobian(value, x), C.jacobian(value, c)])
        @test :OP_BSPLINE in node_ops(f)
        g = native(f)
        @test all(all_agree(evaluated(g, f, [p, coefficients]), mx_casadi(f, [p, coefficients])) for p in SPLINE_POINTS)
        @test all_agree(written(f, [[0.4, 0.3], coefficients]), mx_casadi(f, [[0.4, 0.3], coefficients]))
    end
    @testset "one dimension, degree $degree" for degree in 0:3
        knots = [fill(0.0, degree); [0.0, 1.0, 1.5, 4.0]; fill(4.0, degree)]
        t = MX.sym("t")
        f = CasADiCodegen.Function("curve", [t], [C.bspline(t, C.DM([1.0 + k^2 for k in 1:3+degree]), [knots], [degree], 1, Dict())])
        g = native(f)
        # At a knot the constant spline of CasADi writes out of its work space, so those points are left out for degree 0.
        points = degree == 0 ? (0.5, 1.2, 3.9, -1.0, 5.0) : (0.0, 0.5, 1.0, 1.2, 1.5, 3.9, 4.0, -1.0, 5.0)
        @test all(all_agree(evaluated(g, f, [[p]]), mx_casadi(f, [[p]])) for p in points)
    end
end

@testset "place in a grid, lookup $mode" for mode in ("linear", "exact", "binary")
    v, x = MX.sym("v", 6), MX.sym("x", 5)
    f = CasADiCodegen.Function("place", [v, x], [C.low(v, x, Dict("lookup_mode" => mode))])
    grid = [-1.0, -0.4, 0.2, 0.8, 1.4, 2.0]
    for numbers in ([-3.0, -1.0, -0.4, 0.21, 0.8], [1.39, 1.4, 2.0, 2.5, 0.5], [0.2, 0.19999999999999998, 0.20000000000000004, 1.9, -0.99], [NaN, Inf, -Inf, 0.0, 1.0])
        @test isequal(evaluated(native(f), f, [grid, numbers]), mx_casadi(f, [grid, numbers]))
    end
end

@testset "sum of blocks" begin
    B = MX.sym("B", 2, 6)
    S = MX.sym("S", C.repmat(C.Sparsity.lower(2), 1, 3))
    for (f, n) in ((CasADiCodegen.Function("blocks", [B], [C.repsum(B, 1, 3), C.repsum(B, 2, 1)]), 12),
                   (CasADiCodegen.Function("sparse_blocks", [S], [C.repsum(S, 1, 3)]), 9))
        values = [[0.37k - 1.3 for k in 1:n]]
        @test :OP_HORZREPSUM in node_ops(f)
        @test isequal(evaluated(native(f), f, values), mx_casadi(f, values))
        @test isequal(written(f, values), mx_casadi(f, values))
    end
end

@testset "assertion" begin
    x, y = MX.sym("x", 3), MX.sym("y", 3)
    f = CasADiCodegen.Function("checked", [x, y], [2 * C.attachAssert(x, y[1] > 0, "the first number must be positive")])
    @test :OP_ASSERTION in node_ops(f)
    g = native(f)
    @test evaluated(g, f, [[1.0, 2.0, 3.0], [0.5, 0.0, 0.0]]) == [[2.0, 4.0, 6.0]]
    @test_throws AssertionError("the first number must be positive") evaluated(g, f, [[1.0, 2.0, 3.0], [-0.5, 0.0, 0.0]])
    @test_throws AssertionError written(f, [[1.0, 2.0, 3.0], [NaN, 0.0, 0.0]])
    @test written(f, [[1.0, 2.0, 3.0], [0.5, 0.0, 0.0]]) == [[2.0, 4.0, 6.0]]
    # An assertion without a text, on the value itself.
    s = MX.sym("s")
    plain = CasADiCodegen.Function("plain", [s], [C.attachAssert(s, s)])
    @test evaluated(native(plain), plain, [[1.0]]) == [[1.0]]
    @test_throws AssertionError("") evaluated(native(plain), plain, [[2.0]])
end

@testset "first nonzero" begin
    pattern = C.Sparsity.triplet(6, 1, [1, 3, 4], [0, 0, 0])
    s, v = MX.sym("s", pattern), MX.sym("v", 3)
    f = CasADiCodegen.Function("first", [s, v], [C.find(s), C.find(v)])
    @test :OP_FIND in node_ops(f)
    g = native(f)
    for values in ([0.0, 0, 0], [0.0, 2, 0], [0.0, 0, 5], [1.0, 0, 0], [0.0, NaN, 1.0])
        @test isequal(evaluated(g, f, [values, values]), mx_casadi(f, [values, values]))
    end
    @test isequal(written(f, [[0.0, 2, 0], [0.0, 0, 5]]), mx_casadi(f, [[0.0, 2, 0], [0.0, 0, 5]]))
end

@testset "print and pass" begin
    x, y = MX.sym("x", 2), MX.sym("y", 2)
    f = CasADiCodegen.Function("shown", [x, y], [2 * C.printme(x, y)])
    @test :OP_PRINTME in node_ops(f)
    g = native(f)
    result = [zeros(2)]
    @test printed(() -> g(result, ([1.5, 2.0], [7.0, 9.25]))) == "|> 7.0 : 1.5\n|> 9.25 : 2.0\n"
    @test result == [[3.0, 4.0]]
    # The scalar form of the same operation.
    a, b = SX.sym("a"), SX.sym("b")
    scalar = build_function(CasADiCodegen.Function("shown", [a, b], [C.printme(a, b) + 1]); nonzeros = true, expression = Val{false})
    @test printed(() -> scalar(result, ([0.5], [3.0]))) == "|> 3.0 : 0.5\n"
    @test result[1][1] == 1.5
end

@testset "tensor product" begin
    A, B, y = MX.sym("A", 24), MX.sym("B", 12), MX.sym("y", 2)
    cases = ["one index left" => CasADiCodegen.Function("t", [A, B, y], [C.einstein(A, B, [2, 3, 4], [4, 3], [2], [-1, -2, -3], [-3, -2], [-1])]),
             "added to a value" => CasADiCodegen.Function("t", [A, B, y], [C.einstein(A, B, y, [2, 3, 4], [4, 3], [2], [-1, -2, -3], [-3, -2], [-1])]),
             "outer product" => CasADiCodegen.Function("t", [A, B, y], [C.einstein(B, y, [12], [2], [12, 2], [-1], [-2], [-1, -2])]),
             "its gradient" => CasADiCodegen.Function("t", [A, B, y], [C.gradient(C.dot(C.einstein(A, B, [2, 3, 4], [4, 3], [2], [-1, -2, -3], [-3, -2], [-1]), y), A)])]
    @testset "$label" for (label, f) in cases
        @test :OP_EINSTEIN in node_ops(f)
        values = mx_inputs(f)
        @test isequal(evaluated(native(f), f, values), mx_casadi(f, values))
        label == "one index left" && @test isequal(written(f, values), mx_casadi(f, values))
    end
end

@testset "contracted Kronecker product" begin
    A, B, D = MX.sym("A", 4, 6), MX.sym("B", 2, 3), MX.sym("D", 2, 2)
    f = CasADiCodegen.Function("contract", [A, B, D], [C.kron_contract(A, B, true), C.kron_contract(A, D, false)])
    @test :OP_KRON_CONTRACT in node_ops(f)
    values = mx_inputs(f)
    @test isequal(evaluated(native(f), f, values), mx_casadi(f, values))
    @test isequal(written(f, values), mx_casadi(f, values))
end

@testset "a value written to a file" begin
    pattern = C.Sparsity.lower(2)
    v = MX.sym("v", pattern)
    cd(mktempdir()) do
        f = CasADiCodegen.Function("traced", [v], [2 * C.dump(v, "trace")])
        @test :OP_DUMP in node_ops(f)
        g = native(f)
        @test evaluated(g, f, [[1.0, 3.0, 4.5]]) == [[2.0, 6.0, 9.0]]
        evaluated(g, f, [[5.0, 7.0, 8.0]])
        ours = [read("trace.00000$k.mtx", String) for k in 0:1]
        foreach(rm, readdir())
        f(C.DM(pattern, C.DM([1.0, 3.0, 4.5])))
        f(C.DM(pattern, C.DM([5.0, 7.0, 8.0])))
        @test ours == [read("trace.00000$k.mtx", String) for k in 0:1]
    end
end

# The emitter is tested on an algorithm made by hand for the one node that the
# front end cannot build: the lifted value of the lifted Newton method, which
# is its first operand.
@testset "lifted value" begin
    dense(n) = C.Sparsity.dense(n, 1)
    alg = K.MXAlgorithm("lifted", [K.MXInstruction(:OP_INPUT, [1], Int[], (1, 0)), K.MXInstruction(:OP_INPUT, [2], Int[], (2, 0)),
                                   K.MXInstruction(:OP_LIFT, [3], [1, 2], nothing), K.MXInstruction(:OP_OUTPUT, Int[], [3], (1, 0))],
                        [2, 2, 2], [dense(2), dense(2)], [dense(2)], ["x", "guess"], ["y"])
    @test compiled(alg)([zeros(2)], ([1.5, -2.0], [9.0, 9.0])) == [[1.5, -2.0]]
end

const TABLE_GRID = [[0.0, 1.0, 2.0, 3.5, 5.0], [0.0, 1.0, 2.0, 4.0]]
const TABLE_POINTS = [[0.3, 0.7], [1.0, 2.0], [4.99, 3.9], [-1.0, 5.0], [6.0, -2.0], [2.0, 1.0], [0.0, 0.0], [5.0, 4.0], [3.5, 2.0], [2.7, 3.1]]

@testset "interpolant functions" begin
    x, c = MX.sym("x", 2), MX.sym("c", 40)
    values = [sin(0.9k) + 0.2k for k in 1:20]
    coefficients = [cos(0.4k) + 0.1k for k in 1:40]
    @testset "$kind with fixed values, lookup $modes" for (kind, modes) in (("linear", ["linear", "linear"]), ("bspline", ["binary", "linear"]))
        table = C.interpolant("table", kind, TABLE_GRID, values, Dict("lookup_mode" => modes))
        f = CasADiCodegen.Function("f", [x], [table(x) * 2, C.jacobian(table(x), x), CasADiCodegen.hessian(table(x), x)])
        g = native(f)
        @test all(all_agree(evaluated(g, f, [p]), mx_casadi(f, [p]); rtol = 1e-10) for p in TABLE_POINTS)
        kind == "bspline" && continue       # the written module and the dual numbers take one path for both kinds
        @test all_agree(written(f, [[0.3, 0.7]]), mx_casadi(f, [[0.3, 0.7]]); rtol = 1e-10)
        J = ForwardDiff.jacobian(p -> g([similar(p, n) for n in nnz_out(f)], (p,))[1], [0.3, 0.7])
        @test agree(vec(J), 2 .* mx_casadi(f, [[0.3, 0.7]])[2]; rtol = 1e-10)
    end
    @testset "$kind with values that are an argument" for kind in ("linear", "bspline")
        table = C.interpolant("table", kind, TABLE_GRID, 2)
        f = CasADiCodegen.Function("f", [x, c], [table(x, c), C.jacobian(table(x, c), x), C.jacobian(table(x, c), c)])
        g = native(f)
        @test all(all_agree(evaluated(g, f, [p, coefficients]), mx_casadi(f, [p, coefficients]); rtol = 1e-8) for p in TABLE_POINTS)
    end
    @testset "a grid with equal steps and the exact lookup" begin
        table = C.interpolant("table", "linear", [[0.0, 0.5, 1.0, 1.5], [1.0, 2.0, 3.0]], collect(1.0:12.0) .^ 1.5, Dict("lookup_mode" => ["exact", "exact"]))
        f = CasADiCodegen.Function("f", [x], [table(x), C.gradient(table(x), x)])
        @test all(all_agree(evaluated(native(f), f, [p]), mx_casadi(f, [p]); rtol = 1e-10) for p in ([0.2, 1.3], [1.2, 2.9], [0.5, 2.0], [-1.0, 0.0], [2.0, 3.5]))
    end
    @testset "the interpolant itself" begin
        table = C.interpolant("table", "bspline", TABLE_GRID, values)
        out = build_function(table; expression = Val{false})[1]
        @test out([0.3, 0.7])[1] ≈ dense_values(table(C.DM([0.3, 0.7])))[1] rtol = 1e-12
    end
end

# A map calls its body in one loop, so the size of the code does not depend on
# the number of columns (bug: every map was unrolled).
@testset "a map is a loop" begin
    x, s = MX.sym("x", 2), MX.sym("s", C.Sparsity.lower(2))
    body = CasADiCodegen.Function("body", [x, s], [C.sin(x) * x[1] + s[2, 1], C.mtimes(s, x)])
    mapped(n, kind) = let X = MX.sym("X", 2, n), S = MX.sym("S", C.repmat(C.Sparsity.lower(2), 1, n))
        CasADiCodegen.Function("f", [X, S], [C.map(body, n, kind)(X, S)...])
    end
    code_length(f) = length(string(build_function(f; nonzeros = true, expression = Val{true}, expand = false)))
    @testset "$kind" for kind in ("serial", "thread", "openmp")
        f = mapped(3, kind)
        @test isequal(evaluated(native(f), f, mx_inputs(f)), mx_casadi(f, mx_inputs(f)))
        @test code_length(mapped(300, kind)) <= 1.1 * code_length(f)     # longer numbers only
    end
    @testset "derivatives, which are maps of the derivatives of the body" begin
        f = mapped(3, "serial")
        X, S = C.mx_in(f)
        y = f(X, S)
        J = CasADiCodegen.Function("J", [X, S], [C.jacobian(y[1], X), CasADiCodegen.hessian(C.sumsqr(y[2]), X)[1]])
        @test all_agree(evaluated(native(J), J, mx_inputs(J)), mx_casadi(J, mx_inputs(J)); rtol = 1e-12)
    end
end

# A function of a compiled library is called directly. The library here is the
# C code that CasADi generates for a function, compiled with the C compiler of
# the machine; without a compiler the scenario is skipped.
@testset "external function" begin
    compiler = something(Sys.which("cc"), Sys.which("gcc"), Some(nothing))
    if compiler === nothing
        @test_skip "no C compiler"
    else
        directory = mktempdir()
        a, q = SX.sym("a", 2), SX.sym("q")
        model = CasADiCodegen.Function("model", [a, q], [C.sin(a) * q, C.dot(a, a), SX(2, 1)])
        cd(directory) do
            C.generate(model, "model.c", Dict("with_header" => false))
            run(`$compiler -shared -fPIC -O1 model.c -o libmodel.so`)
        end
        external = C.external("model", joinpath(directory, "libmodel.so"))
        @test K.external_library(external) == joinpath(directory, "libmodel.so")
        x, p = MX.sym("x", 2), MX.sym("p")
        f = CasADiCodegen.Function("f", [x, p], [external(x, p)[1] * 2 + external(2 * x, p)[2], external(x, p)[3]])
        g = native(f)
        values = [[0.4, -1.1], [2.5]]
        @test isequal(evaluated(g, f, values), mx_casadi(f, values))
        results, arguments = [zeros(2), zeros(0)], Tuple(values)
        g(results, arguments)
        @test @allocated(g(results, arguments)) <= 512        # 256 bytes on Julia 1.13, for the two calls of the library; the bound leaves room for another version
        @test isequal(written(f, values), mx_casadi(f, values))
        dual = ForwardDiff.Dual(0.4, 1.0)
        @test_throws "\"model\" is compiled code, which takes Float64 only" g([fill(dual, 2), fill(dual, 0)], ([dual, dual], [dual]))
        @test_throws "is not a file" K.external_library(model)
        # A library without the optional functions, and a function that fails.
        write(joinpath(directory, "bare.c"), "int bare(const double** arg, double** res, long long* iw, double* w, int mem) { res[0][0] = 3 * arg[0][0]; return arg[0][0] < 0; }")
        run(`$compiler -shared -fPIC $(joinpath(directory, "bare.c")) -o $(joinpath(directory, "libbare.so"))`)
        bare = K.ExternalCall("bare", joinpath(directory, "libbare.so"), [1], [1])
        @test bare([2.0]) == ([6.0],)
        @test_throws "failed with the code 1" bare([-2.0])
    end
end

# The Newton rootfinder of CasADi is the same iteration in Julia code.
@testset "Newton rootfinder of CasADi" begin
    x, L, a = MX.sym("x", 2), MX.sym("L", C.Sparsity.lower(2)), MX.sym("a")
    residual = CasADiCodegen.Function("residual", [x, L], [C.mtimes(L, x) + x^3 - 1])
    values = [[0.4, 0.9], [1.3, 0.2, 0.7]]
    @testset "$label" for (label, options) in ("the options of CasADi" => Dict(), "no line search" => Dict("line_search" => false),
                                                "other tolerances" => Dict("abstol" => 1e-6, "abstolStep" => 1e-3))
        newton = C.rootfinder("newton", "newton", residual, options)
        f = CasADiCodegen.Function("f", [x, L], [newton(x, L), C.jacobian(newton(x, L), L)])
        g = native(f)
        @test all_agree(evaluated(g, f, values), mx_casadi(f, values); rtol = 1e-11)
        isempty(options) || continue        # the options change numbers of the iteration only
        @test all_agree(written(f, values), mx_casadi(f, values); rtol = 1e-11)
        results, arguments = [zeros(2), zeros(6)], Tuple(values)
        g(results, arguments)
        @test @allocated(g(results, arguments)) <= 1024       # 304 bytes on Julia 1.13, for the calls of the solver; the bound leaves room for another version
    end
    # The unknown is the second input, the residual is the second result, and the other result is given at the root.
    shifted = CasADiCodegen.Function("shifted", [a, x], [x * a, x^2 - a])
    newton = C.rootfinder("newton", "newton", shifted, Dict("implicit_input" => true, "implicit_output" => true))
    f = CasADiCodegen.Function("f", [a, x], collect(newton(a, x)))
    g = native(f)
    @test all_agree(evaluated(g, f, [[2.0], [3.0, 1.0]]), mx_casadi(f, [[2.0], [3.0, 1.0]]); rtol = 1e-12)
    # Dual numbers go through the iteration: the derivative of the root of x^2 = a is 1 / (2 x).
    slope = ForwardDiff.derivative(t -> g([zeros(typeof(t), 2), zeros(typeof(t), 2)], ([t], [3.0, 1.0]))[2][1], 2.0)
    @test slope ≈ 1 / (2 * sqrt(2.0)) rtol = 1e-10
    # The front end gives a whole number as 1, so this is a limit of one iteration: CasADi fails, and so does the code.
    limited(options) = (r = C.rootfinder("limited", "newton", shifted, merge(Dict{String,Any}("implicit_input" => true, "implicit_output" => true), options));
                        CasADiCodegen.Function("f", [a, x], collect(r(a, x))))
    @test_throws "the Newton rootfinder \"limited\" failed" evaluated(native(limited(Dict("max_iter" => 1))), limited(Dict("max_iter" => 1)), [[2.0], [3.0, 1.0]])
    quiet = limited(Dict("max_iter" => 1, "error_on_fail" => false))
    @test all_agree(evaluated(native(quiet), quiet, [[2.0], [3.0, 1.0]]), mx_casadi(quiet, [[2.0], [3.0, 1.0]]); rtol = 1e-12)
    # No step above the tolerance makes this residual smaller, so the line search fails as that of CasADi does.
    stuck = C.rootfinder("stuck", "newton", CasADiCodegen.Function("r", [a], [a^2 + 1]), Dict("error_on_fail" => false, "abstolStep" => 1.0))
    h = CasADiCodegen.Function("h", [a], [stuck(a)])
    @test all(all_agree(evaluated(native(h), h, [[start]]), mx_casadi(h, [[start]]); rtol = 1e-12) for start in (0.1, 0.5))
    @test evaluated(native(h), h, [[0.1]]) == [[0.1]]
end

# A Jacobian of 60 rows and about 3 elements per row gets the sparse LU factorization.
@testset "Newton rootfinder of CasADi with a sparse Jacobian" begin
    n = 60
    z, q = SX.sym("z", n), SX.sym("q")
    at(i) = 1 <= i <= n ? z[i] : SX(0.0)
    bratu = [at(i - 1) - 2 * z[i] + at(i + 1) + q * exp(z[i]) / (n + 1)^2 for i in 1:n]
    # Row i holds equation i + 2, so the diagonal of the Jacobian is zero and the factorization must exchange rows.
    residual = CasADiCodegen.Function("bratu", [z, q], [vertcat(bratu[[3:n; 1; 2]]...)])
    newton = C.rootfinder("bratu_newton", "newton", residual)
    x, λ = MX.sym("x", n), MX.sym("λ")
    f = CasADiCodegen.Function("f", [x, λ], [newton(x, λ), C.jacobian(newton(x, λ), λ)])
    values = [zeros(n), [2.0]]
    g = native(f)
    @test all_agree(evaluated(g, f, values), mx_casadi(f, values); rtol = 1e-10)
    # Dual numbers go through the sparse factorization.
    slope = ForwardDiff.derivative(t -> g([zeros(typeof(t), n), zeros(typeof(t), n)], (zeros(n), [t]))[1][n ÷ 2], 2.0)
    @test slope ≈ mx_casadi(f, values)[2][n÷2] rtol = 1e-8
end

# An integrator of CasADi with a fixed step is its loop of steps, written out.
@testset "integrators of CasADi with a fixed step" begin
    x, p, u, t = MX.sym("x", 2), MX.sym("p"), MX.sym("u", 1, 3), MX.sym("t")
    control = MX.sym("c")
    dae = Dict("x" => x, "p" => p, "u" => control, "t" => t, "ode" => C.vertcat([x[2], -p * C.sin(x[1]) + control * t]), "quad" => C.dot(x, x))
    values = [[0.4, -0.3], [1.7], [0.2, -0.1, 0.5]]
    @testset "$plugin" for plugin in ("rk", "collocation")
        # Seven elements from the start time 0.25 give one, two and four steps, so both branches of the loop run.
        rule = C.integrator("rule", plugin, dae, 0.25, [0.5, 1.0, 2.0], Dict("number_of_finite_elements" => 7))
        run(x0, parameter, controls) = rule(x0, MX(0, 1), parameter, controls, MX(), MX(), MX())
        final, quadrature = run(x, p, u)[1], run(x, p, u)[3]
        f = CasADiCodegen.Function("f", [x, p, u], [final, quadrature, C.jacobian(final, x), C.jacobian(quadrature, p), C.jacobian(final[:, 3], u)])
        g = native(f)
        @test all_agree(evaluated(g, f, values), mx_casadi(f, values); rtol = 1e-10)
        # Dual numbers go through the loop of steps, which both methods share, so one method shows it.
        if plugin == "rk"
            slope = ForwardDiff.derivative(s -> g([zeros(typeof(s), n) for n in nnz_out(f)], (values[1], [s], values[3]))[2][3], values[2][1])
            @test slope ≈ mx_casadi(f, values)[4][3] rtol = 1e-8
        end
        # The adjoint sensitivities are the reverse derivative of the same loop, so one method shows them too.
        if plugin == "collocation"
            cost = CasADiCodegen.Function("cost", [x, p, u], [C.gradient(C.sumsqr(final) + quadrature[3], C.vertcat([x, p, C.vec(u)]))])
            @test all_agree(evaluated(native(cost), cost, values), mx_casadi(cost, values); rtol = 1e-9)
        end
    end
    # Collocation has unknowns for the algebraic states at each point; z = p x1 makes x1' = -z a decay.
    @testset "algebraic states" begin
        z = MX.sym("z")
        rule = C.integrator("rule", "collocation", Dict("x" => x, "z" => z, "p" => p, "ode" => C.vertcat([-z, x[1]]), "alg" => z - p * x[1]), 0.0, [0.5, 1.0],
                            Dict("number_of_finite_elements" => 2))
        reached = rule(x, MX(0.3), p, MX(0, 1), MX(), MX(), MX())
        f = CasADiCodegen.Function("f", [x, p], [reached[1], reached[2], C.jacobian(reached[2], p)])
        @test all_agree(evaluated(native(f), f, values[1:2]), mx_casadi(f, values[1:2]); rtol = 1e-10)
    end
    @testset "two adjoint directions at once" begin
        # CasADi asks for them for a Jacobian with two rows and many columns.
        wide = MX.sym("wide", 6)
        many = C.integrator("many", "collocation", Dict("x" => x, "p" => wide, "ode" => -C.sum1(wide) * x^2 + wide[1:2], "quad" => x[1] * wide[3]), 0.0, [0.5, 1.0],
                            Dict("number_of_finite_elements" => 4))
        reached = many(x, MX(0, 1), wide, MX(0, 1), MX(), MX(), MX())
        rows = CasADiCodegen.Function("rows", [x, wide], [C.jacobian(reached[1][:, 2], wide), C.jacobian(C.vertcat([reached[3][2], reached[1][1, 1]]), wide)])
        point = [values[1], [0.3, -0.2, 0.5, 0.1, 0.4, 0.2]]
        @test all_agree(evaluated(native(rows), rows, point), mx_casadi(rows, point); rtol = 1e-9)
    end
    # The derivatives of the second order go through the same loop for both methods; rk keeps the code small.
    @testset "derivatives of the second order" begin
        w = MX.sym("w", 2)
        rule = C.integrator("rule", "rk", Dict("x" => x, "p" => w, "u" => control, "ode" => -w[1] * x^2 * w[2] + control, "quad" => x[1] * w[1]), 0.0, [0.5, 1.0],
                            Dict("number_of_finite_elements" => 2))
        result = rule(x, MX(0, 1), w, u[1:2], MX(), MX(), MX())
        cost = C.sumsqr(result[1]) + result[3][2]
        f = CasADiCodegen.Function("f", [x, w, u], [CasADiCodegen.hessian(cost, w), C.jacobian(C.jacobian(result[1][:, 2], w), x), C.jacobian(C.gradient(cost, x), u)])
        points = [values[1], [1.7, 0.6], values[3]]
        @test all_agree(evaluated(native(f), f, points), mx_casadi(f, points); rtol = 1e-8)
    end
end

# A matrix that CasADi makes positive definite: by a shift of its diagonal, or
# by a change of the eigenvalues of each block of it.
@testset "matrix made positive definite" begin
    blocks = C.Sparsity(5, 5, [0, 2, 4, 5, 7, 9], [0, 1, 0, 1, 2, 3, 4, 3, 4])
    dense(n, shift) = [sin(1.3 * (i + j)) + 0.4 * cos(i * j) + (i == j) * shift for i in 1:n, j in 1:n][:]
    # The values: with eigenvalues of both signs, positive definite, with equal eigenvalues, and all zero.
    cases = ["dense" => (C.Sparsity.dense(4, 4), [dense(4, 0.0), dense(4, 6.0), [1.0, 0, 0, 0, 0, 1, 0, 0, 0, 0, -2, 0, 0, 0, 0, -2], zeros(16)]),
             "blocks" => (blocks, [[1.0, 3, 3, -2, -0.5, 4, 1, 1, 0.2], [2.0, 0, 0, 2, 3, 1, 0, 0, 1]]),
             "lower triangle" => (C.Sparsity.lower(3), [[1.0, 2, -3, -4, 0.5, 0.1], [5.0, 1, 0, 4, 1, 3]]),
             "upper triangle" => (C.Sparsity.upper(3), [[1.0, 2, -3, -4, 0.5, 0.1]]),
             "lower triangle of blocks" => (C.tril(blocks), [[1.0, 3, -2, -0.5, 4, 1, 0.2]]),
             "one number" => (C.Sparsity.dense(1, 1), [[-3.0], [2.0]])]
    @testset "$label, $strategy" for (label, (pattern, values)) in cases, strategy in ("regularize", "eigen-clip", "eigen-reflect")
        strategy == "regularize" && occursin("triangle", label) && continue       # CasADi refuses it
        strategy == "eigen-reflect" && label != "dense" && continue                # it differs from eigen-clip by the sign of an eigenvalue only
        H = MX.sym("H", pattern)
        f = CasADiCodegen.Function("convex", [H], [C.convexify(2 * H, Dict("strategy" => strategy, "margin" => 1e-3)), C.convexify(H, Dict("strategy" => strategy))])
        @test :OP_CONVEXIFY in node_ops(f)
        g = native(f)
        @test all(all_agree(evaluated(g, f, [v]), mx_casadi(f, [v]); rtol = 1e-11) for v in values)
        label == "dense" && @test all_agree(written(f, values[1:1]), mx_casadi(f, values[1:1]); rtol = 1e-11)
    end
    @testset "dual numbers" begin
        H = MX.sym("H", 3, 3)
        f = CasADiCodegen.Function("convex", [H], [C.convexify(H, Dict("strategy" => "eigen-reflect"))])
        g, value, direction = native(f), [2.0, 1, 0.3, 1, -3, 0.5, 0.3, 0.5, 1], [0.1, 0.2, 0, 0.2, -0.3, 0.1, 0, 0.1, 0.4]
        run(s) = g([zeros(typeof(s), 9)], (value .+ s .* direction,))[1]
        h = 1e-6
        @test ForwardDiff.derivative(run, 0.0) ≈ (run(h) - run(-h)) / 2h rtol = 1e-6
    end
    @testset "no convergence is an error" begin
        # The front end gives the whole number of an option as 1, so the method has one iteration only.
        H = MX.sym("H", 4, 4)
        f = CasADiCodegen.Function("convex", [H], [C.convexify(H, Dict("strategy" => "eigen-clip", "max_iter_eig" => 1))])
        @test_throws Exception f(C.DM(reshape(dense(4, 0.0), 4, 4)))
        @test_throws "the eigenvalues of a block did not converge in 1 iterations" evaluated(native(f), f, [dense(4, 0.0)])
    end
end
