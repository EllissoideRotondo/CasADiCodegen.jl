module CasADiCodegenSymbolicsExt

import CasADiCodegen
import Symbolics
using CasADiCodegen: SX
using CasADiCodegen.Codegen: @requires, @ensures

const SU = Symbolics.SymbolicUtils

"The operations of Symbolics.jl expressions that have a method for CasADi scalars."
const OPERATIONS = Set{Any}([
    +, -, *, /, ^, sin, cos, tan, asin, acos, atan, sinh, cosh, tanh, asinh, acosh, atanh,
    exp, log, log2, log10, sqrt, abs, sign, floor, ceil, expm1, log1p, exp2, inv,
    min, max, hypot, ifelse, <, >, <=, >=, ==, !=, &, |, !,
])

"The operations of Symbolics.jl that CasADi scalars spell differently, such as the `signbit` in the derivative of `abs`."
const SUBSTITUTES = Dict{Any,Function}(signbit => a -> a < 0)

"The scalar symbols and expressions of `a`, in column-major order, and the size of `a`."
function elements(a)
    a isa Union{Symbolics.Num,Symbolics.Arr,AbstractArray{<:Symbolics.Num}} ||
        throw(ArgumentError("casadi_function: $(typeof(a)) is not a symbolic number or an array of them; complex values have no CasADi counterpart"))
    a isa Symbolics.Num && return [a], ()
    scalars = Symbolics.scalarize(a)
    vec(scalars), size(scalars)
end

"""
    assemble(values, shape) -> SX

The scalar, the column or the matrix of the CasADi scalars `values`, which `shape` gives.
"""
function assemble(values, shape)
    isempty(shape) && return only(values)
    column(v) = length(v) == 1 ? v[1] : CasADiCodegen.vertcat(v...)
    length(shape) == 1 && return column(values)
    @requires length(shape) == 2 "casadi_function: an array of $(length(shape)) dimensions has no CasADi shape; use a vector or a matrix"
    columns = [column(values[(j-1)*shape[1]+1:j*shape[1]]) for j in 1:shape[2]]
    length(columns) == 1 ? columns[1] : CasADiCodegen.horzcat(columns...)
end

"""
    translate(e, known) -> SX or Number

The CasADi value of the Symbolics expression `e`, from the values `known` of its symbols.
`known` also holds the value of each subexpression that has been translated, so a shared
subexpression is translated once.
"""
function translate(e, known::Dict)
    haskey(known, e) && return known[e]
    value = if SU.isconst(e)
        constant = SU.unwrap_const(e)
        @requires constant isa Real "casadi_function: the complex or non-numeric constant $constant has no CasADi value"
        constant isa Rational ? float(constant) : constant
    elseif SU.iscall(e)
        op = SU.operation(e)
        @requires op in OPERATIONS || haskey(SUBSTITUTES, op) "casadi_function: the operation $op has no CasADi counterpart"
        get(SUBSTITUTES, op, op)(map(a -> translate(a, known), SU.arguments(e))...)
    else
        throw(ArgumentError("casadi_function: the symbol $e is not an input"))
    end
    known[e] = value
end

"The CasADi symbols for `inputs`, the Dict that maps the Symbolics symbols to them, and the arguments."
function symbols(inputs)
    known = Dict{Any,Any}()
    arguments = map(inputs) do input
        scalars, shape = elements(input)
        syms = [SX.sym(string(s)) for s in scalars]
        for (s, sym) in zip(scalars, syms)
            known[Symbolics.unwrap(s)] = sym
        end
        assemble(syms, shape)
    end
    known, arguments
end

function CasADiCodegen.casadi_function(name::AbstractString, inputs::AbstractVector, outputs::AbstractVector;
                                       input_names = nothing, output_names = nothing)
    known, arguments = symbols(inputs)
    results = map(outputs) do output
        scalars, shape = elements(output)
        assemble([SX(translate(Symbolics.unwrap(s), known)) for s in scalars], shape)
    end
    @ensures length(results) == length(outputs) "casadi_function: $(length(results)) results for $(length(outputs)) outputs"
    input_names === nothing && output_names === nothing && return CasADiCodegen.Function(name, arguments, results)
    CasADiCodegen.Function(name, arguments, results,
                           something(input_names, ["i$i" for i in 0:length(arguments)-1]),
                           something(output_names, ["o$i" for i in 0:length(results)-1]))
end

end # module
