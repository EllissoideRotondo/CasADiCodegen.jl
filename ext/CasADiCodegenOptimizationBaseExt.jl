module CasADiCodegenOptimizationBaseExt

import CasADiCodegen
import OptimizationBase
import SciMLBase
using CasADiCodegen.Codegen: @requires

const G = CasADiCodegen.Codegen

# For `NoAD`, OptimizationBase.jl 5 gives the solvers a `lag_h(h, x[, p])` that cannot take the
# arguments (h, x, σ, μ[, p]) of a solver, and it leaves out `lag_hess_prototype`. For a SciML
# function of this package, the method below passes both on as they are.

"""
    WithParameters(lag_h, p)

The Hessian of the Lagrange function `lag_h(h, x, σ, μ, p)` of a SciML function, which a
solver calls with the parameters `p` of the problem or without them.
"""
struct WithParameters{L,P}
    lag_h::L
    p::P
end
(w::WithParameters)(h, x, σ, μ) = w.lag_h(h, x, σ, μ, w.p)
(w::WithParameters)(h, x, σ, μ, p) = w.lag_h(h, x, σ, μ, p)

function OptimizationBase.instantiate_function(f::SciMLBase.OptimizationFunction{true,SciMLBase.NoAD,<:G.GeneratedObjective},
                                               x, adtype::SciMLBase.NoAD, p, num_cons = 0; kwargs...)
    instantiated = invoke(OptimizationBase.instantiate_function, Tuple{SciMLBase.OptimizationFunction{true},Any,SciMLBase.NoAD,Any,Any},
                          f, x, adtype, p, num_cons; kwargs...)
    f.lag_h === nothing && return instantiated
    @requires f.lag_hess_prototype !== nothing "OptimizationFunction: lag_h needs its lag_hess_prototype"
    fields = (name => getfield(instantiated, name) for name in fieldnames(typeof(instantiated)) if name ∉ (:f, :adtype, :lag_h, :lag_hess_prototype))
    SciMLBase.OptimizationFunction{true}(instantiated.f, adtype; fields..., lag_h = WithParameters(f.lag_h, p),
                                         lag_hess_prototype = similar(f.lag_hess_prototype, eltype(x)))
end

# The method for the cache of a problem that is solved again, which would otherwise be ambiguous.
OptimizationBase.instantiate_function(f::SciMLBase.OptimizationFunction{true,SciMLBase.NoAD,<:G.GeneratedObjective},
                                      cache::OptimizationBase.ReInitCache, adtype::SciMLBase.NoAD, num_cons = 0; kwargs...) =
    OptimizationBase.instantiate_function(f, cache.u0, adtype, cache.p, num_cons; kwargs...)

end # module CasADiCodegenOptimizationBaseExt
