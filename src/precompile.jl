# Compile the emitters when the package is precompiled, so that the first
# `build_function` of a session does not pay for them.
@compile_workload begin
    C.__init__()    # Julia runs it after the module is complete; the workload needs it before
    x = SX.sym("x", 2)
    y = [sin(x[1]) * x[2] + 1; x[1]^2 / x[2]]
    scalar = Function("f", [x], [y, jacobian(y, x)])
    A, v = MX.sym("A", C.Sparsity.lower(2)), MX.sym("v", 2)
    z = C.mtimes(A, sin.(v))
    matrix = Function("m", [A, v], [z, jacobian(z, v)])
    for (f, arg, expand) in ((scalar, ([0.3, 1.7],), true), (matrix, ([1.0 0.0; 2.0 3.0], [0.3, 1.7]), false))
        g, g! = build_function(f; expression = Val{false}, expand)
        g!(g(arg...), arg)
        build_function(f; expression = Val{false}, expand, nonzeros = true, chunk_size = 2)
        build_function(f; expand)
        generate_julia(f, IOBuffer(); expand)
    end
    empty!(Codegen.OP_NAMES)    # read from the library again in each session
end
