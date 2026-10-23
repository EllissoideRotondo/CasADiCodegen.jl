module CasADiCodegenOptimizationBaseExt

import CasADiCodegen
import OptimizationBase
import SciMLBase
using CasADiCodegen.Codegen: @requires
using DifferentiationInterface: SecondOrder

const G = CasADiCodegen.Codegen

"""
    GENERATED

The `adtype` that a problem of a SciML function of this package takes in Optimization.jl: no
automatic differentiation of the first or the second order, since the function holds the
generated derivatives. A solver that needs the second derivatives finds them declared, so
OptimizationBase.jl does not warn that a `SecondOrder` type is missing.
"""
const GENERATED = SecondOrder(SciMLBase.NoAD(), SciMLBase.NoAD())
const Generated = typeof(GENERATED)

"""
    KEYWORDS

The keywords of the constructor of `SciMLBase.OptimizationFunction` that `rebuilt` passes on:
each field of the function but `f` and `adtype`. The tests compare this list with the fields
of the release of SciMLBase.jl in use.
"""
const KEYWORDS = (:grad, :fg, :hess, :fgh, :hv, :cons, :cons_j, :cons_jvp, :cons_vjp, :cons_h, :hess_prototype,
                  :cons_jac_prototype, :cons_hess_prototype, :observed, :expr, :cons_expr, :sys, :lag_h,
                  :lag_hess_prototype, :hess_colorvec, :cons_jac_colorvec, :cons_hess_colorvec, :lag_hess_colorvec,
                  :initialization_data)

"""
    rebuilt(f::SciMLBase.OptimizationFunction, adtype; changes...)

Return `f` with `adtype` and the fields in `changes` replaced, through the public constructor
of `SciMLBase.OptimizationFunction`. The other fields in `KEYWORDS` are kept.
"""
function rebuilt(f::SciMLBase.OptimizationFunction{iip}, adtype; changes...) where {iip}
    @requires all(in(KEYWORDS), keys(changes)) "rebuilt: $(keys(changes)) are not all fields of an OptimizationFunction"
    kept = NamedTuple{KEYWORDS}(map(name -> getproperty(f, name), KEYWORDS))
    SciMLBase.OptimizationFunction{iip,SciMLBase.specialization(f)}(f.f, adtype; kept..., changes...)
end

# A problem of a SciML function of this package takes the adtype `GENERATED`, which tells the
# cache of Optimization.jl that the second derivatives are there. The SciML function itself
# keeps `NoAD`, since the extension for SciMLBase.jl cannot depend on DifferentiationInterface.jl.
SciMLBase.OptimizationProblem{true}(f::SciMLBase.OptimizationFunction{true,SciMLBase.NoAD,<:G.GeneratedObjective}, u0,
                                    p = SciMLBase.NullParameters(); kwargs...) =
    SciMLBase.OptimizationProblem{true}(rebuilt(f, GENERATED), u0, p; kwargs...)

"""
    WithLagrangeHessian{AD}

A `SciMLBase.OptimizationFunction` with the `adtype` of type `AD` and a `lag_h` of this package.
"""
const WithLagrangeHessian{AD} = SciMLBase.OptimizationFunction{true,AD,F,Gr,FG,H,FGH,HV,C,CJ,CJV,CVJ,CH,HP,CJP,CHP,O,EX,CEX,SYS,<:G.GeneratedLagrangeHessian} where {F,Gr,FG,H,FGH,HV,C,CJ,CJV,CVJ,CH,HP,CJP,CHP,O,EX,CEX,SYS}

"""
    Current{N}(f, source)

The function `f(args..., p)` of an instantiated SciML function, which a solver calls with `N`
arguments and no parameters, or with `N` arguments and the parameters `p`. Without them it
takes the current parameters of `source`: the `p` of an `OptimizationBase.ReInitCache`, which
`reinit!` replaces, or the value of a `Ref`.
"""
struct Current{N,F,S}
    f::F
    source::S
end
Current{N}(f, source) where {N} = Current{N,typeof(f),typeof(source)}(f, source)
(c::Current{N})(arguments::Vararg{Any,N}) where {N} = c.f(arguments..., parameters(c.source))
function (c::Current{N})(arguments::Vararg{Any,M}) where {N,M}
    @requires M == N + 1 "a function of the OptimizationFunction takes $N arguments and the parameters; got $M arguments"
    c.f(arguments...)
end
parameters(cache::OptimizationBase.ReInitCache) = cache.p
parameters(p::Base.RefValue) = p[]

"""
    PARAMETRIC

The number of arguments before the parameters `p` of each function that OptimizationBase.jl
instantiates from a given function with `NoAD`, as `grad(G, x[, p])`.
"""
const PARAMETRIC = (grad = 2, fg = 2, hess = 2, fgh = 3, hv = 3, cons = 2, cons_j = 2, cons_jvp = 3, cons_vjp = 3, cons_h = 2)

"""
    current(instantiated, f, cache::OptimizationBase.ReInitCache)

Return the function `instantiated` from `f` with each function given in `f` called through
`Current`, so that a solver that calls it without the parameters gets the `p` of `cache`.
OptimizationBase.jl 5 gives such a solver the parameters of the time of the instantiation, which
OptimizationMOI.jl uses after `reinit!` replaces them.
"""
function current(instantiated::SciMLBase.OptimizationFunction, f::SciMLBase.OptimizationFunction, cache::OptimizationBase.ReInitCache)
    cache.p isa SciMLBase.NullParameters && return instantiated
    changes = (; (name => Current{n}(getproperty(instantiated, name), cache) for (name, n) in pairs(PARAMETRIC) if getproperty(f, name) !== nothing)...)
    f isa WithLagrangeHessian && (changes = (; changes..., lag_h = Current{4}(f.lag_h, cache)))
    rebuilt(instantiated, instantiated.adtype; changes...)
end

# `GENERATED` is instantiated as `NoAD`, by OptimizationBase.jl or, for a `lag_h` of this
# package, by the methods further below.
OptimizationBase.instantiate_function(f::SciMLBase.OptimizationFunction{true,Generated,<:G.GeneratedObjective}, x,
                                      ::Generated, p, num_cons = 0; kwargs...) =
    OptimizationBase.instantiate_function(rebuilt(f, SciMLBase.NoAD()), x, SciMLBase.NoAD(), p, num_cons; kwargs...)

OptimizationBase.instantiate_function(f::SciMLBase.OptimizationFunction{true,Generated,<:G.GeneratedObjective},
                                      cache::OptimizationBase.ReInitCache, ::Generated, num_cons = 0; kwargs...) =
    current(OptimizationBase.instantiate_function(rebuilt(f, SciMLBase.NoAD()), cache.u0, SciMLBase.NoAD(), cache.p, num_cons; kwargs...), f, cache)

# For `NoAD`, OptimizationBase.jl 5 gives the solvers a `lag_h(h, x[, p])` that cannot take the
# arguments (h, x, σ, μ[, p]) of a solver, and it leaves out `lag_hess_prototype`. For a `lag_h`
# of this package, the methods below pass both on.
function OptimizationBase.instantiate_function(f::WithLagrangeHessian{SciMLBase.NoAD}, x, ::SciMLBase.NoAD, p, num_cons = 0; kwargs...)
    @requires f.lag_hess_prototype !== nothing "OptimizationFunction: lag_h needs its lag_hess_prototype"
    instantiated = invoke(OptimizationBase.instantiate_function, Tuple{SciMLBase.OptimizationFunction{true},Any,SciMLBase.NoAD,Any,Any},
                          f, x, SciMLBase.NoAD(), p, num_cons; kwargs...)
    rebuilt(instantiated, SciMLBase.NoAD(); lag_h = Current{4}(f.lag_h, Ref(p)), lag_hess_prototype = similar(f.lag_hess_prototype, eltype(x)))
end

OptimizationBase.instantiate_function(f::WithLagrangeHessian{SciMLBase.NoAD}, cache::OptimizationBase.ReInitCache, ::SciMLBase.NoAD,
                                      num_cons = 0; kwargs...) =
    current(OptimizationBase.instantiate_function(f, cache.u0, SciMLBase.NoAD(), cache.p, num_cons; kwargs...), f, cache)

end # module CasADiCodegenOptimizationBaseExt
