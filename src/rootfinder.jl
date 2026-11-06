# Roots of equations in CasADi graphs. A solver that is a plain Julia function
# finds the root with the emitted residual and its emitted Jacobian, and the
# implicit function theorem gives CasADi the derivatives of the root.

"""
    root_function(name, f, make_solver) -> CasADiCodegen.Function

Return the CasADi function `(z0, p1, p2, ...) -> z` that solves
`f(z, p1, p2, ...) = 0` for `z` from the first guess `z0`. `f` is a CasADi
function with one result, the residual, which has the size of its first
argument, the unknown. The unknown is a dense column vector that is not empty;
the other arguments are dense.

`make_solver(residual, slope, n)` gets two Julia functions of
`(z, p1, p2, ...)` for `n` unknowns: the residual and its Jacobian by `z`. They
return static arrays when `z` is a static vector. The Jacobian of one equation
is then a vector of one element, as each column is. `make_solver` returns the
Julia function `(z0, p1, p2, ...) -> root`.

The root does not depend on the guess. Its derivative by a parameter `p` is the
solution of `(df/dz) X = -df/dp` at the root, which CasADi differentiates again
for the higher orders.
"""
function root_function(name::AbstractString, f::C.Function, make_solver)
    @requires Base.isidentifier(name) "\"$name\" is not a valid function name"
    @requires C.n_in(f) >= 1 && C.n_out(f) == 1 "rootfinder \"$name\": the function must have the unknown as its first argument and the residual as its only result"
    patterns = [C.sparsity_in(f, i) for i in 0:C.n_in(f)-1]
    n = C.size1(patterns[1])
    @requires C.size2(patterns[1]) == 1 && n > 0 "rootfinder \"$name\": the unknown must be a column vector that is not empty"
    @requires all(C.is_dense, patterns) "rootfinder \"$name\": the arguments must be dense"
    @requires (C.size1(C.sparsity_out(f, 0)), C.size2(C.sparsity_out(f, 0))) == (n, 1) "rootfinder \"$name\": the residual must have the size of the unknown"

    symbols(prefix) = [MX.sym("$prefix$i", sp) for (i, sp) in enumerate(patterns)]
    arguments = symbols("a")
    value = Front.densify(f(arguments...))
    # The solver calls the residual and its slope inside the function that calls the solver.
    emitted(g) = Emitted(Base.ScopedValues.with(() -> build_function(g; expression = Val{false})[1], CONTEXT => Nested), g, :outofplace)
    residual = emitted(Front.Function("residual", arguments, [value]))
    slope = emitted(Front.Function("slope", arguments, [Front.densify(Front.jacobian(value, arguments[1]))]))
    implicit_function(name, f, make_solver(residual, slope, n))
end

"""
    implicit_function(name, f, solve) -> CasADiCodegen.Function

Return the CasADi function of `root_function` for the Julia function
`solve(z0, p1, p2, ...) -> root`, which finds the root of `f` by any means.
This function does not check `f`; the caller does (see `root_function`).
"""
function implicit_function(name::AbstractString, f::C.Function, solve)
    patterns = [C.sparsity_in(f, i) for i in 0:C.n_in(f)-1]
    n = C.size1(patterns[1])
    symbols(prefix) = [MX.sym("$prefix$i", sp) for (i, sp) in enumerate(patterns)]
    # The blocks of the Jacobian as one function of the root and the parameters. With scalar
    # (SX) symbols the linear system is solved in symbols, so that CasADi can expand the code
    # around the rootfinder; a function that CasADi cannot expand keeps a linear solve.
    scalar = expanded_algorithm(f) === nothing ? nothing : C.expand(f)
    kind = scalar === nothing ? MX : SX
    at = [kind.sym("a$i", sp) for (i, sp) in enumerate(patterns)]
    g = Front.densify((scalar === nothing ? f : scalar)(at...))
    by_root = Front.jacobian(g, at[1])
    slopes = Front.Function("slopes", at, [-Front.solve(by_root, Front.jacobian(g, parameter)) for parameter in at[2:end]])

    jacobian = function (jacobian_name)
        inputs, root = symbols("i"), MX.sym("z", n)
        blocks = MX[MX(n, n)]                    # the root does not depend on the guess
        length(inputs) > 1 && append!(blocks, as_list(slopes(root, inputs[2:end]...)))
        Front.Function(jacobian_name, [inputs..., root], blocks, Dict("always_inline" => true))
    end

    shapes_in = [C.size2(sp) == 1 ? (C.size1(sp),) : (C.size1(sp), C.size2(sp)) for sp in patterns]
    call = JuliaCall(String(name), solve, length(shapes_in), shapes_in, [(n,)], map(prod, shapes_in), [n])
    recorded_function(SymbolicCallback(call, solve, jacobian))
end
