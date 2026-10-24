# The changes to the front end, made when its generated module `casadi` is included:
# the numbers in a Dict of options keep their type (see options.jl), `der` of a
# `DaeBuilder` and a matrix symbol calls the overload of CasADi that can declare a
# differential state (see `derivative_definition!`), and `jacobian`
# of a graph with an integrator of CasADi with an event function gives the Jacobian of
# forward directions, because CasADi 3.8.1 gives such an integrator the sparsity
# pattern of the model without events (see `Codegen.event_jacobian`). The methods of
# `integrator` with a plugin name first check the model for switches on the state and
# read the option "nonsmooth" (see `Codegen.check_native_integrator`).

# The code generation, which holds the methods that the changed `jacobian` calls. The
# module is defined after the front end, so the call finds it when it runs.
const CODEGEN = GlobalRef(@__MODULE__, :Codegen)

"""
    event_definitions!(ex, count::Ref{Int}) -> ex

Rewrite the methods `jacobian(ex::MX, arg::MX...)` and `jacobian(self::Function)` in `ex`,
so that they first call `Codegen.event_jacobian` and `Codegen.check_jacobian`; `count` is
increased by the number of rewritten methods.
"""
function event_definitions!(ex, count::Ref{Int})
    ex isa Expr || return ex
    if Meta.isexpr(ex, (:(=), :function)) && (signature = call_signature(ex.args[1])) !== nothing
        signature.args[1] === :jacobian || return ex
        arguments = signature.args[2:end]
        if length(arguments) >= 2 && arguments[1] == :(ex::MX) && arguments[2] == :(arg::MX)
            ex.args[2] = Expr(:block, :((J = $(Expr(:., CODEGEN, QuoteNode(:event_jacobian)))(ex, arg)) === nothing || return J), ex.args[2])
            count[] += 1
        elseif arguments == [:(self::Function)]
            ex.args[2] = Expr(:block, :($(Expr(:., CODEGEN, QuoteNode(:check_jacobian)))(self)), ex.args[2])
            count[] += 1
        end
        return ex
    end
    foreach(a -> event_definitions!(a, count), ex.args)
    ex
end

"""
    native_integrator_definitions!(ex, count::Ref{Int}) -> ex

Rewrite the methods `integrator(name::AbstractString, solver::AbstractString, dae, ...)` in
`ex`, so that they first call `Codegen.check_native_integrator`, which warns about a switch
on the state in the model and removes the option `"nonsmooth"`; `count` is increased by
the number of rewritten methods.
"""
function native_integrator_definitions!(ex, count::Ref{Int})
    ex isa Expr || return ex
    if Meta.isexpr(ex, (:(=), :function)) && (signature = call_signature(ex.args[1])) !== nothing
        arguments = signature.args[2:end]
        if signature.args[1] === :integrator && length(arguments) >= 3 && arguments[1:2] == [:(name::AbstractString), :(solver::AbstractString)]
            check = Expr(:., CODEGEN, QuoteNode(:check_native_integrator))
            names = map(argument_name, arguments)
            names[3] === :dae || throw(AssertionError("the method integrator(name, solver, ...) of the front end has changed"))
            first_call = :opts in names ? :(opts = $check(name, dae, opts)) : :($check(name, dae, nothing))
            ex.args[2] = Expr(:block, first_call, ex.args[2])
            count[] += 1
        end
        return ex
    end
    foreach(a -> native_integrator_definitions!(a, count), ex.args)
    ex
end

"""
    derivative_definition!(ex, count::Ref{Int}) -> ex

Rewrite the method `der(self::DaeBuilder, v::MX)` in `ex` so that it always calls the
non-const overload `MX DaeBuilder::der(const MX&)` of CasADi; `count` is increased by the
number of rewritten methods.

The generated method calls the const overload whenever SWIG can convert the arguments,
which is always. That overload only reads the derivative of a variable and throws
"Variable x has no time derivative" for a variable that has none. The non-const overload,
which Python and C++ call, also makes the derivative, so that `eq(dae, der(dae, x), rhs)`
declares `x` a differential state.
"""
function derivative_definition!(ex, count::Ref{Int})
    ex isa Expr || return ex
    if Meta.isexpr(ex, (:(=), :function)) && (signature = call_signature(ex.args[1])) !== nothing
        if signature.args[1] === :der && signature.args[2:end] == [:(self::DaeBuilder), :(v::MX)]
            branch = only(filter(a -> Meta.isexpr(a, :if), ex.args[2].args))
            # The `else` branch of the generated dispatch calls the non-const overload, `_SWIG_3`.
            occursin("der__SWIG_3", string(branch.args[3])) ||
                throw(AssertionError("the method der(::DaeBuilder, ::MX) of the front end has changed"))
            ex.args[2] = branch.args[3]
            count[] += 1
        end
        return ex
    end
    foreach(a -> derivative_definition!(a, count), ex.args)
    ex
end

"The number of methods that `shaped_zero_definitions!` has rewritten: 2 when the front end is loaded."
const SHAPED_ZEROS = Ref(0)

"""
    shaped_zero_definitions!(ex) -> ex

Rewrite the methods `zero(::T)` and `one(::T)` of the matrix types `T` in `ex`, which return
a scalar for a matrix of any shape, so that `zero(x)` is the zero of the shape of
`x` and `one(x)` the identity of the square matrix `x`, as for a Julia matrix (see
`Codegen.zero_matrix` and `Codegen.identity_matrix`). The methods of the types,
`zero(::Type{T})` and `one(::Type{T})`, keep their scalars. `SHAPED_ZEROS` counts the
rewritten methods.
"""
function shaped_zero_definitions!(ex)
    ex isa Expr || return ex
    if Meta.isexpr(ex, :(=)) && Meta.isexpr(ex.args[1], :call) && length(ex.args[1].args) == 2
        name, argument = ex.args[1].args
        if name in (:(Base.zero), :(Base.one)) && Meta.isexpr(argument, :(::), 1) && Meta.isexpr(argument.args[1], :$)
            method = name == :(Base.zero) ? :zero_matrix : :identity_matrix
            ex.args[1].args[2] = Expr(:(::), :x, argument.args[1])
            ex.args[2] = :($(Expr(:., CODEGEN, QuoteNode(method)))(x))
            SHAPED_ZEROS[] += 1
        end
        return ex
    end
    foreach(shaped_zero_definitions!, ex.args)
    ex
end

"""
    typed_front_end(ex)

The expression `ex` of the convenience layer of the front end, with the include of
the generated module `casadi` given `typed_module` as its `mapexpr`, and with the methods
`zero` and `one` of a matrix rewritten by `shaped_zero_definitions!`.
"""
function typed_front_end(ex)
    Meta.isexpr(ex, :call) && ex.args[1] === :include && length(ex.args) == 2 || return shaped_zero_definitions!(ex)
    Expr(:call, :include, typed_module, ex.args[2])
end

function typed_module(ex)
    Meta.isexpr(ex, :module) || return ex
    count = Ref(0)
    typed_definitions!(ex, count)
    count[] > 100 || throw(AssertionError("only $(count[]) functions of the front end take options; the front end has changed"))
    jacobians = Ref(0)
    event_definitions!(ex, jacobians)
    jacobians[] == 3 || throw(AssertionError("$(jacobians[]) methods of jacobian of the front end were found, not 3; the front end has changed"))
    integrators = Ref(0)
    native_integrator_definitions!(ex, integrators)
    integrators[] == 12 || throw(AssertionError("$(integrators[]) methods of integrator with a plugin name were found in the front end, not 12; the front end has changed"))
    derivatives = Ref(0)
    derivative_definition!(ex, derivatives)
    derivatives[] == 1 || throw(AssertionError("$(derivatives[]) methods der(::DaeBuilder, ::MX) of the front end were found, not 1; the front end has changed"))
    ex
end
