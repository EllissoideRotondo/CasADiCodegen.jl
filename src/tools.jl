# Tools to look into a function: what its code is made of, and where a NaN
# comes from.

operations(alg::Algorithm) = Symbol[ins.op for ins in alg.instructions]
operations(alg::MXAlgorithm) = Symbol[ins.op for ins in alg.instructions]
operations(alg::StagedAlgorithm) =
    vcat((vcat(operations(stage.pre), :OP_CALL) for stage in alg.stages if stage.live)..., operations(alg.post))

"""
    instruction_histogram(f; expand = true) -> Vector{Pair{Symbol,Int}}

Count the instructions of the code that [`build_function`](@ref) emits for the CasADi
function `f`, by operation. Return the pairs `operation => count`, the most frequent first.

Matrix code counts the instructions of `f` itself, not those of the functions that it
calls. `expand` is that of `build_function`.

# Examples
```jldoctest
julia> using CasADiCodegen

julia> x = SX.sym("x", 2);

julia> f = CasadiFunction("f", [x], [sin(x[1]) * x[2] + x[1] * x[2]]);

julia> CasADiCodegen.instruction_histogram(f)
5-element Vector{Pair{Symbol, Int64}}:
  :OP_INPUT => 2
    :OP_MUL => 2
    :OP_ADD => 1
 :OP_OUTPUT => 1
    :OP_SIN => 1
```
"""
function instruction_histogram(f; expand::Bool = true)
    counts = Dict{Symbol,Int}()
    for op in operations(algorithm(f; expand))
        counts[op] = get(counts, op, 0) + 1
    end
    sort!(collect(counts); by = entry -> (-entry.second, entry.first))
end

# One Julia function per operation, built from the expressions of the emitter.
const OPERATION_FUNCTIONS = Dict{Symbol,Function}(
    op => Core.eval(@__MODULE__, entry.arity == 1 ? :(a -> $(entry.build(global_ref, :a))) :
                                                   :((a, b) -> $(entry.build(global_ref, :a, :b))))
    for (op, entry) in OPERATIONS)

"""
    nan_trace(f, arguments...) -> NamedTuple or nothing

Find where the first NaN appears when the CasADi function `f` runs on numbers. Return the
first instruction of the scalar code of `f` whose result is NaN while no operand is, or the
first NaN among the arguments. Return `nothing` when there is no NaN.

Give one argument per input: an array with the shape of the input, or the vector of its
structural nonzeros. CasADi must be able to expand `f` to scalar code, so `f` cannot call a
Julia function.

The result has these fields:

- `instruction`: the place of the instruction in the scalar code.
- `operation`: the operation, for example `:OP_SQRT`.
- `operands`: the values of the operands.
- `result`: the result of the operation.
- `text`: the operation on those values, or the place of the NaN in the arguments.

# Examples
```jldoctest
julia> using CasADiCodegen

julia> x = SX.sym("x", 2);

julia> f = CasadiFunction("f", [x], [sqrt(x[1]) + x[2]]);

julia> CasADiCodegen.nan_trace(f, [-1.0, 2.0])
(instruction = 2, operation = :OP_SQRT, operands = (-1.0,), result = NaN, text = "NaNMath.sqrt(-1.0)")

julia> CasADiCodegen.nan_trace(f, [1.0, 2.0]) === nothing
true
```
"""
function nan_trace(f, arguments...)
    alg = expanded_algorithm(f)
    @requires alg !== nothing "nan_trace needs a function that CasADi can expand to scalar code"
    @requires length(arguments) == length(alg.nnz_in) "$(alg.name): expected $(length(alg.nnz_in)) arguments, got $(length(arguments))"
    values = map(eachindex(arguments)) do i
        argument, sp = arguments[i], alg.sparsity_in[i]
        length(argument) == alg.nnz_in[i] && return Float64[argument...]
        @requires length(argument) == C.size1(sp) * C.size2(sp) "$(alg.name): argument $i ($(alg.names_in[i])) needs $(C.size1(sp) * C.size2(sp)) elements or $(alg.nnz_in[i]) nonzeros, got $(length(argument))"
        Float64[argument[k] for k in nonzero_indices(sp)]
    end
    w = zeros(alg.nwork)
    for (k, ins) in enumerate(alg.instructions)
        # Loop invariant: no instruction before `k` made a NaN from operands without one.
        op = ins.op
        if op === :OP_INPUT
            w[ins.out] = values[ins.a][ins.b]
            isnan(w[ins.out]) && return (instruction = k, operation = op, operands = (), result = NaN,
                                         text = "nonzero $(ins.b) of argument $(ins.a) ($(alg.names_in[ins.a]))")
        elseif op === :OP_CONST
            w[ins.out] = ins.constant
        elseif op !== :OP_OUTPUT
            operands = OPERATIONS[op].arity == 1 ? (w[ins.a],) : (w[ins.a], w[ins.b])
            result = Float64(OPERATION_FUNCTIONS[op](operands...))
            w[ins.out] = result
            isnan(result) && !any(isnan, operands) &&
                return (; instruction = k, operation = op, operands, result, text = string(OPERATIONS[op].build(global_ref, operands...)))
        end
    end
    nothing
end
