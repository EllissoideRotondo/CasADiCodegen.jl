# The Newton rootfinder of CasADi (`rootfinder(name, "newton", f)`) as Julia
# code: the iteration of CasADi 3.8.1 (`Newton::solve`), with the emitted
# residual and the emitted Jacobian, so that generated code does not call
# CasADi for it.

"The work space of one task and one number type for a Newton rootfinder."
struct NewtonWork{T,R<:Tuple,F<:Union{Nothing,SparseLU{T}}}
    inputs::Vector{Vector{T}}                   # a copy of each argument, so that every call has plain vectors
    x::Vector{T}
    f::Vector{T}
    x_trial::Vector{T}
    f_trial::Vector{T}
    matrix::Matrix{T}                           # the dense Jacobian; empty when `factors` holds a sparse one
    factors::F
    res_step::Vector{Vector{T}}                 # the Jacobian, then the results with the residual in `f`
    res_trial::Vector{Vector{T}}                # the results with the residual in `f_trial`
    returned::R
end

"""
    NewtonCall

The CasADi Newton rootfinder for the function `g`, as the Julia function that
generated code calls. It takes the inputs of `g`, with a first guess in the
place of the unknown. It returns the results of `g` at the root, with the root
in the place of the residual.

- `step(res, arg)` writes the nonzeros of the Jacobian of the residual by the
  unknown, then those of each result of `g`.
- `residual(res, arg)` writes the nonzeros of the results only.
- The unknown is input `iin`, and the residual is result `iout`.
- `rows[k]` and `columns[k]` give the place of nonzero `k` of the Jacobian,
  which has `n` rows and columns.

The iteration ports `Newton::solve` of CasADi 3.8.1. It stops when the largest
residual is at most `abstol`, or when the largest element of the step is at
most `abstol_step`. With `line_search` the step is halved until the largest
residual falls by the factor `1 - alpha / 2`.

The linear system is solved by elimination with row exchanges, where CasADi
uses a QR factorization, so the results agree with CasADi to rounding. A
Jacobian with at most a quarter of its elements stored, and at least
`SPARSE_NEWTON_SIZE` rows, gets a sparse LU factorization (see `SparseLU`). A call
allocates no memory after the first one of a task. Number types other than
`Float64`, such as dual numbers, go through the same iteration.
"""
mutable struct NewtonCall{S,G}
    const name::String
    const step::S
    const residual::G
    const n::Int
    const rows::Vector{Int}
    const columns::Vector{Int}
    const iin::Int
    const iout::Int
    const nnz_in::Vector{Int}
    const nnz_out::Vector{Int}
    const max_iter::Int
    const abstol::Float64
    const abstol_step::Float64
    const line_search::Bool
    const error_on_fail::Bool
end

function newton_work(s::NewtonCall, ::Type{T}) where {T}
    works = get!(() -> IdDict{Any,Any}(), task_local_storage(), s)::IdDict{Any,Any}
    get!(works, T) do
        vector(n) = Vector{T}(undef, n)
        x, f, f_trial = vector(s.n), vector(s.n), vector(s.n)
        results = [vector(n) for n in s.nnz_out]
        with(residual) = [o == s.iout ? residual : r for (o, r) in enumerate(results)]
        sparse = sparse_newton(s.n, length(s.rows))
        matrix = sparse ? Matrix{T}(undef, 0, 0) : Matrix{T}(undef, s.n, s.n)
        factors = sparse ? SparseLU{T}(s.n, s.rows, s.columns) : nothing
        NewtonWork(map(vector, s.nnz_in), x, f, vector(s.n), f_trial, matrix, factors, [[vector(length(s.rows))]; with(f)], with(f_trial),
                   Tuple(o == s.iout ? x : r for (o, r) in enumerate(results) if s.nnz_out[o] > 0))
    end
end

"""
The largest absolute value, as CasADi computes it: a NaN element is not counted.
"""
function largest(values)
    result = zero(eltype(values))
    for v in values
        abs(v) > result && (result = abs(v))
    end
    result
end

"""
    eliminate!(A, b) -> b

Solve `A x = b` in-place by Gaussian elimination with partial pivoting: `b`
becomes the solution and `A` is overwritten. `A` must be square with the size
of `b`; the body does not check it. A singular matrix gives non-finite numbers
and no error.
"""
function eliminate!(A::Matrix, b::Vector)
    n = length(b)
    @inbounds for k in 1:n
        # Loop invariant: the columns before k are zero below the diagonal.
        pivot = k
        for i in k+1:n
            abs(A[i, k]) > abs(A[pivot, k]) && (pivot = i)
        end
        if pivot != k
            for j in k:n
                A[k, j], A[pivot, j] = A[pivot, j], A[k, j]
            end
            b[k], b[pivot] = b[pivot], b[k]
        end
        for i in k+1:n
            factor = A[i, k] / A[k, k]
            for j in k+1:n
                A[i, j] -= factor * A[k, j]
            end
            b[i] -= factor * b[k]
        end
    end
    @inbounds for k in n:-1:1
        for j in k+1:n
            b[k] -= A[k, j] * b[j]
        end
        b[k] /= A[k, k]
    end
    b
end

"""
    newton_step!(m::NewtonWork, s::NewtonCall)

Overwrite the residual `m.f` with the Newton step: the solution of `J x = f`, where the
Jacobian `J` has its nonzeros in `m.res_step[1]`.
"""
function newton_step!(m::NewtonWork{T,R,Nothing}, s::NewtonCall) where {T,R}
    fill!(m.matrix, zero(T))
    for (k, value) in enumerate(m.res_step[1])
        m.matrix[s.rows[k], s.columns[k]] = value
    end
    eliminate!(m.matrix, m.f)
end
newton_step!(m::NewtonWork{T,R,<:SparseLU}, s::NewtonCall) where {T,R} = (factor!(m.factors, m.res_step[1]); solve!(m.factors, m.f))

function (s::NewtonCall)(arguments::Vararg{Any,N}) where {N}
    T = promote_type(Float64, map(eltype, arguments)...)
    m = newton_work(s, T)::NewtonWork{T}
    foreach(copyto!, m.inputs, arguments)
    at(x) = ntuple(i -> i == s.iin ? x : m.inputs[i], Val(N))
    copyto!(m.x, m.inputs[s.iin])
    iteration, success = 0, true
    while true
        # Loop invariant: `m.x` is the iterate after `iteration` evaluations of the Jacobian.
        if iteration >= s.max_iter
            success = false
            break
        end
        iteration += 1
        s.step(m.res_step, at(m.x))
        abstol = largest(m.f)
        s.abstol != Inf && abstol <= s.abstol && break
        newton_step!(m, s)
        abstol_step = largest(m.f)
        s.abstol_step != Inf && abstol_step <= s.abstol_step && break
        if s.line_search
            alpha = 1.0
            while true
                # Loop invariant: no step longer than `alpha` times the Newton step gave the required decrease.
                m.x_trial .= m.x .- alpha .* m.f
                s.residual(m.res_trial, at(m.x_trial))
                if largest(m.f_trial) <= (1 - alpha / 2) * abstol
                    copyto!(m.x, m.x_trial)
                    break
                end
                if alpha * abstol_step <= s.abstol_step
                    success = false
                    break
                end
                alpha *= 0.5
            end
            success || break
        else
            m.x .-= m.f
        end
    end
    success || !s.error_on_fail || error("the Newton rootfinder \"$(s.name)\" failed: the limit of $(s.max_iter) iterations, or no step that makes the residual smaller")
    m.returned
end

"""
    newton_call(f) -> JuliaCall

Return the record of the CasADi Newton rootfinder `f` as a direct call of
Julia code (see `NewtonCall`). CasADi gives the residual and its Jacobian as
functions. The options are read from the serial form of `f` (see
`serial_value`). The option `constraints` is not read: the Newton method of
CasADi does not use it.
"""
function newton_call(f::C.Function)
    bytes = serial_bytes(f)
    field(name; kwargs...) = serial_value(bytes, name; kwargs...)
    emitted(g) = Emitted(build_function(g; nonzeros = true, expression = Val{false}), g, :nonzeros)
    jacobian = C.get_function(f, "jac_g_x")
    rows, columns = positions(C.sparsity_out(jacobian, 0))
    nnz_in, nnz_out = [C.nnz_in(f, i) for i in 0:C.n_in(f)-1], [C.nnz_out(f, o) for o in 0:C.n_out(f)-1]
    call = NewtonCall(C.name(f), emitted(jacobian), emitted(C.get_function(f, "g")), field("Rootfinder::n"), rows, columns,
                      field("Rootfinder::iin") + 1, field("Rootfinder::iout") + 1, nnz_in, nnz_out, field("Newton::max_iter"), field("Newton::abstol"),
                      field("Newton::abstolStep"), field("Newton::line_search"), field("ProtoFunction::error_on_fail"; first = true))
    JuliaCall(C.name(f), call, length(nnz_in), Tuple{Vararg{Int}}[(n,) for n in nnz_in], Tuple{Vararg{Int}}[(n,) for n in nnz_out], nnz_in, nnz_out)
end
