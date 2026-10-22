# Generated code kept on disk as a precompiled package. The functions come from
# a module that is loaded inside the test set, so the tests call them with
# `Base.invokelatest`, as running code must.
include("ode_helpers.jl")
import LinearAlgebra
using SimpleNonlinearSolve: SimpleNewtonRaphson
using StaticArrays: SVector, SMatrix

# A function in a module of Main, which is not in a package.
module CacheModels
drag(v) = [v[1]^2 * v[2], sin(v[1]) + v[3]]
end

"Does the method of `g` hold compiled code for these argument types before any call?"
compiled_for(g, types) = any(mi -> mi.specTypes <: Tuple{typeof(g),types...}, Iterators.flatten(map(Base.specializations, methods(g))))

@testset "cache of generated code on disk" begin
    directory = mktempdir()
    x, p = SX.sym("x", 3), SX.sym("p", 2, 2)
    y = vertcat(sin(x[1]) * p[1, 1] + x[3], x[2]^2 * p[2, 2])
    f = CasADiCodegen.Function("f", [x, p], [y, jacobian(y, x)])
    g, g! = build_function(f; expression = Val{false})
    h, h! = build_function(f; expression = Val{false}, cache = directory)
    packages = readdir(directory)
    @test length(packages) == 1 && startswith(packages[1], "CasADiCodegenCache_")
    Arrays = (Vector{Float64}, Matrix{Float64})
    @test compiled_for(h!, (Tuple{Arrays...}, Tuple{Arrays...}))
    @test compiled_for(h, Arrays)

    xv, pv = [0.3, 1.7, -0.4], [2.0 0.5; 0.1 3.0]
    @test Base.invokelatest(h, xv, pv) == g(xv, pv)
    res = (zeros(2), zeros(2, 3))
    Base.invokelatest(h!, res, (xv, pv))
    @test res == g(xv, pv)
    @test Base.invokelatest((r, a) -> @allocated(h!(r, a)), res, (xv, pv)) == 0

    # The same function again: the same package and the same functions.
    again, again! = build_function(f; expression = Val{false}, cache = directory)
    @test again === h && again! === h!
    @test length(readdir(directory)) == 1

    # Other options give another package; the form with static arrays is compiled with force_SA.
    s, _ = build_function(f; expression = Val{false}, cache = directory, force_SA = true, nanmath = false)
    @test length(readdir(directory)) == 2
    @test compiled_for(s, (SVector{3,Float64}, SMatrix{2,2,Float64,4}))
    @test Base.invokelatest(s, SVector{3}(xv), SMatrix{2,2}(pv)) == g(SVector{3}(xv), SMatrix{2,2}(pv))

    # A new Julia process with the same model finds the package and compiles nothing.
    # (A package name that depends on the session would make a third package here.)
    compiled_files(g) = sort!([(file, stat(file).inode) for file in Base.find_all_in_cache_path(Base.PkgId(parentmodule(parentmodule(g))))])
    before = compiled_files(h!)
    script = """
        using CasADiCodegen
        x, p = SX.sym("x", 3), SX.sym("p", 2, 2)
        y = vertcat(sin(x[1]) * p[1, 1] + x[3], x[2]^2 * p[2, 2])
        f = CasADiCodegen.Function("f", [x, p], [y, jacobian(y, x)])
        h, h! = build_function(f; expression = Val{false}, cache = $(repr(directory)))
        print(length(readdir($(repr(directory)))), " ", Base.invokelatest(h, $xv, $pv)[1] == $(g(xv, pv)[1]))
        """
    seconds = @elapsed out = read(`$(Base.julia_cmd()) --startup-file=no --project=$(Base.active_project()) -e $script`, String)
    haskey(ENV, "CASADI_TEST_VERBOSE") && println("cache in a new process: ", round(seconds; digits = 1), " s")
    @test out == "2 true"
    # A compilation writes a new file; a load leaves the file in its place.
    @test !isempty(before) && compiled_files(h!) == before
end

@testset "cache: function on nonzeros and a Julia function of a package" begin
    directory = mktempdir()
    len = callback(LinearAlgebra.norm, "len", (3,), (1,))
    v = MX.sym("v", 3)
    f = CasADiCodegen.Function("f", [v], [len(2 * v) + v[1]])
    n = build_function(f; expression = Val{false}, cache = directory, nonzeros = true)
    @test compiled_for(n, (Vector{Vector{Float64}}, Tuple{Vector{Float64}}))
    @test Base.invokelatest(n, [zeros(1)], ([3.0, 0.0, 4.0],)) == [[13.0]]
    package = only(readdir(directory))
    @test occursin("LinearAlgebra = ", read(joinpath(directory, package, "Project.toml"), String))
end

@testset "cache: wrong use" begin
    x = SX.sym("x", 2)
    f = CasADiCodegen.Function("f", [x], [x[1] * x[2]])
    directory = mktempdir()
    @test_throws ArgumentError build_function(f; cache = directory)                 # an expression has no cache
    local_function = callback(v -> [v[1]], "local", (2,), (1,))
    v = MX.sym("v", 2)
    @test_throws ArgumentError build_function(CasADiCodegen.Function("f", [v], [local_function(v)]);
                                              expression = Val{false}, cache = directory)
    main = callback(CacheModels.drag, "drag", (3,), (2,))          # a function of Main is not in a package
    w = MX.sym("w", 3)
    @test_throws ArgumentError build_function(CasADiCodegen.Function("f", [w], [main(w)]); expression = Val{false}, cache = directory)
    @test isempty(readdir(directory))
    @test startswith(CasADiCodegen.Codegen.cache_directory(true), first(DEPOT_PATH))
    @test CasADiCodegen.Codegen.cache_directory("somewhere") == "somewhere"
    # A package that is loaded and is not a dependency of CasADiCodegen.jl, and one that does not exist.
    @test CasADiCodegen.Codegen.dependency(:Test) == Base.PkgId(Test)
    @test_throws ArgumentError CasADiCodegen.Codegen.dependency(:NoSuchPackage)
end

# Integrators and rootfinders in source files. The written module makes the
# solvers again from their algorithms and options, and it holds the code of
# the right sides, so it gives the numbers of `build_function` exactly.
written_nonzeros(owner, name, f, arguments) =
    Base.invokelatest((r, a) -> getfield(getfield(owner, name[1]), name[2])(r, a), [zeros(C.nnz_out(f, i - 1)) for i in 1:C.n_out(f)], arguments)

@testset "integrators and rootfinders in a written module" begin
    F = integrator("WF", Vern9(), two_body(), 0.0, 3.0; abstol = 1e-12, reltol = 1e-12, specialize = SciMLBase.AutoSpecialize)
    f = with_derivatives(F, 1)
    text = sprint(io -> generate_julia(f, io; module_name = :Orbit, name = :arc!))
    src = Module()
    Base.include_string(src, text)
    emitted = ode_emitted(f, (ODE_X0, ODE_MU))
    @test written_nonzeros(src, (:Orbit, :arc_nonzeros!), f, (ODE_X0, ODE_MU)) == emitted
    @test Base.invokelatest((a, b) -> getfield(getfield(src, :Orbit), :arc)(a, b), ODE_X0, ODE_MU)[1] == emitted[1]

    # An event with adjoint derivatives and a rootfinder, in matrix code; the module above has the scalar code.
    y, z, a = SX.sym("y", 2), MX.sym("z"), MX.sym("a")
    swing = Dict("x" => y, "ode" => vertcat([y[2], -y[1]]), "zero" => y[1])
    E = integrator("WE", Vern9(), swing; adjoint = true, direction = -1, TIGHT...)
    root = rootfinder("WR", SimpleNewtonRaphson(), CasADiCodegen.Function("cube", [z, a], [z^3 + z - a]); abstol = 1e-13)
    y0 = MX.sym("y0", 2)
    xe, te = E(y0, MX(0.0), MX(50.0))
    solved = root(MX(1.0), te + xe[2])
    g = CasADiCodegen.Function("mixed", [y0], [xe, te, solved, gradient(solved, y0)])
    text = sprint(io -> generate_julia(g, io; module_name = :Mixed, name = :mixed!, expand = false))
    src = Module()
    Base.include_string(src, text)
    # A solver that ran before starts with its last step, so the numbers agree to its tolerance.
    @test closeto(written_nonzeros(src, (:Mixed, :mixed_nonzeros!), g, ([1.0, 0.2],)), ode_emitted(g, ([1.0, 0.2],); expand = false), 1e-10)

    # What a source file cannot hold: a solver that is a local function, and a call through CasADi.
    local_solver(rhs!, n, m) = (x0, q) -> x0
    x, p = SX.sym("x", 1), SX.sym("p", 1)
    H = CasADiCodegen.Codegen.ode_integrator("WH", Dict("x" => x, "p" => p, "ode" => -p * x), 0.0, 1.0, local_solver)
    x0, q = MX.sym("x0", 1), MX.sym("q", 1)
    @test_throws "named function" generate_julia(CasADiCodegen.Function("f", [x0, q], [H(x0, q)]), IOBuffer())
    odd = integrator("WO", Vern9(), delete!(copy(swing), "zero"), 0.0, 1.0; TIGHT..., callback = Ref(1))
    @test_throws "a source file cannot hold the value" generate_julia(CasADiCodegen.Function("f", [y0], [odd(y0)]), IOBuffer())
end

@testset "cache: a function with an integrator" begin
    directory = mktempdir()
    F = integrator("CF", Vern9(), two_body(), 0.0, 3.0; abstol = 1e-12, reltol = 1e-12, specialize = SciMLBase.AutoSpecialize)
    f = with_derivatives(F, 1)
    cached = build_function(f; expression = Val{false}, cache = directory, nonzeros = true)
    result = [zeros(C.nnz_out(f, i - 1)) for i in 1:C.n_out(f)]
    @test Base.invokelatest(cached, result, (ODE_X0, ODE_MU)) == ode_emitted(f, (ODE_X0, ODE_MU))
    project = read(joinpath(directory, only(readdir(directory)), "Project.toml"), String)
    @test occursin("SciMLBase = ", project) && occursin("OrdinaryDiffEqVerner = ", project)
end
