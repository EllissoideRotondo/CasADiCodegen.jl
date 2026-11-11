"""
    Instruction(op, out, a, b, constant)

One step of a scalar algorithm. All indices are 1-based.

| `op`          | `out`        | `a`             | `b`             |
|---------------|--------------|-----------------|-----------------|
| `:OP_INPUT`   | work slot    | argument index  | nonzero index   |
| `:OP_OUTPUT`  | result index | work slot       | nonzero index   |
| `:OP_CONST`   | work slot    | 0               | 0               |
| unary         | work slot    | work slot       | 0               |
| binary        | work slot    | work slot       | work slot       |

`constant` is meaningful for `:OP_CONST` only.
"""
struct Instruction
    op::Symbol
    out::Int
    a::Int
    b::Int
    constant::Float64
end

Instruction(op::Symbol, out::Integer, a::Integer=0, b::Integer=0) =
    Instruction(op, out, a, b, 0.0)

"""
    Algorithm

The instruction list of a scalar (SX) CasADi function, which CasADi calls its
algorithm, copied into Julia values. The function maps the nonzeros of its
inputs to the nonzeros of its outputs, with `nwork` slots of work space.

Invariants (checked by the constructor):
- every work slot lies in `1:nwork` and is written before it is read;
- every `:OP_INPUT` reads an existing nonzero of an existing argument;
- every `:OP_OUTPUT` writes an existing nonzero of an existing result;
- every operation is known to the operation table.
"""
struct Algorithm
    name::String
    instructions::Vector{Instruction}
    nwork::Int
    nnz_in::Vector{Int}
    nnz_out::Vector{Int}
    sparsity_in::Vector{casadi.Sparsity}
    sparsity_out::Vector{casadi.Sparsity}
    names_in::Vector{String}
    names_out::Vector{String}
    function Algorithm(name, instructions, nwork, sparsity_in, sparsity_out, names_in, names_out)
        @requires length(names_in) == length(sparsity_in) "one name per input is required"
        @requires length(names_out) == length(sparsity_out) "one name per output is required"
        alg = new(String(name), collect(Instruction, instructions), nwork,
                  Int[casadi.nnz(sp) for sp in sparsity_in], Int[casadi.nnz(sp) for sp in sparsity_out],
                  collect(casadi.Sparsity, sparsity_in), collect(casadi.Sparsity, sparsity_out),
                  collect(String, names_in), collect(String, names_out))
        check_algorithm(alg)
        alg
    end
end

"""
    check_algorithm(alg) -> nothing

Verify the `Algorithm` invariants; throw `ArgumentError` on the first violation.
"""
function check_algorithm(alg::Algorithm)
    defined = falses(alg.nwork)
    slot_ok(s) = 1 <= s <= alg.nwork
    for (k, ins) in enumerate(alg.instructions)
        op = ins.op
        if op === :OP_INPUT
            @requires slot_ok(ins.out) "instruction $k: work slot $(ins.out) out of range"
            @requires 1 <= ins.a <= length(alg.nnz_in) "instruction $k: no argument $(ins.a)"
            @requires 1 <= ins.b <= alg.nnz_in[ins.a] "instruction $k: argument $(ins.a) has no nonzero $(ins.b)"
            defined[ins.out] = true
        elseif op === :OP_OUTPUT
            @requires 1 <= ins.out <= length(alg.nnz_out) "instruction $k: no result $(ins.out)"
            @requires 1 <= ins.b <= alg.nnz_out[ins.out] "instruction $k: result $(ins.out) has no nonzero $(ins.b)"
            @requires slot_ok(ins.a) && defined[ins.a] "instruction $k: reads undefined work slot $(ins.a)"
        elseif op === :OP_CONST
            @requires slot_ok(ins.out) "instruction $k: work slot $(ins.out) out of range"
            defined[ins.out] = true
        else
            @requires haskey(OPERATIONS, op) "instruction $k: unsupported operation $op"
            n = OPERATIONS[op].arity
            @requires slot_ok(ins.out) "instruction $k: work slot $(ins.out) out of range"
            @requires slot_ok(ins.a) && defined[ins.a] "instruction $k: reads undefined work slot $(ins.a)"
            @requires n == 1 || (slot_ok(ins.b) && defined[ins.b]) "instruction $k: reads undefined work slot $(ins.b)"
            defined[ins.out] = true
        end
        # Loop invariant: `defined` marks exactly the slots written by instructions 1:k.
    end
    nothing
end
