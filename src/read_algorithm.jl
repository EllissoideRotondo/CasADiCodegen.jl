# CasADi operation code => operation name, read once from the loaded library.
const OP_NAMES = Dict{Int,Symbol}()

function op_names()
    if isempty(OP_NAMES)
        for n in names(casadi; all = true)
            startswith(string(n), "OP_") && (OP_NAMES[getfield(casadi, n)::Int] = n)
        end
    end
    OP_NAMES
end

"""
    Algorithm(f::CasADiCodegen.Function)

Copy the instruction list of the scalar (SX) function `f`, one instruction per
CasADi instruction. `f` must be an SX function with no free symbols.
"""
function Algorithm(f::casadi.Function)
    @requires casadi.is_a(f, "SXFunction") "function \"$(casadi.name(f))\" is not a scalar (SX) function"
    @requires !casadi.has_free(f) "function \"$(casadi.name(f))\" has free symbols: $(casadi.get_free(f))"
    names = op_names()
    n = casadi.n_instructions(f)
    instructions = Vector{Instruction}(undef, n)
    for k in 1:n
        op = names[casadi.instruction_id(f, k - 1)]
        i = casadi.instruction_input(f, k - 1)
        o = casadi.instruction_output(f, k - 1)
        instructions[k] = if op === :OP_INPUT
            Instruction(op, o[1] + 1, i[1] + 1, i[2] + 1)
        elseif op === :OP_OUTPUT
            Instruction(op, o[1] + 1, i[1] + 1, o[2] + 1)
        elseif op === :OP_CONST
            Instruction(op, o[1] + 1, 0, 0, casadi.instruction_constant(f, k - 1))
        else
            @requires haskey(OPERATIONS, op) "function \"$(casadi.name(f))\": unsupported operation $op at instruction $k"
            arity = OPERATIONS[op].arity
            Instruction(op, o[1] + 1, i[1] + 1, arity == 2 ? i[2] + 1 : 0)
        end
    end
    Algorithm(casadi.name(f), instructions, casadi.sz_w(f),
              [casadi.sparsity_in(f, j) for j in 0:casadi.n_in(f)-1],
              [casadi.sparsity_out(f, j) for j in 0:casadi.n_out(f)-1],
              [casadi.name_in(f, j) for j in 0:casadi.n_in(f)-1],
              [casadi.name_out(f, j) for j in 0:casadi.n_out(f)-1])
end
