module CasADiCodegenChainRulesCoreExt

import CasADiCodegen
import ChainRulesCore
using ChainRulesCore: AbstractZero, NoTangent, Tangent, unthunk
using CasADiCodegen.Codegen: Differentiable, directional

const casadi = CasADiCodegen.casadi

array(::AbstractZero, sp) = casadi.size2(sp) == 1 ? zeros(casadi.size1(sp)) : zeros(casadi.size1(sp), casadi.size2(sp))
array(value, sp) = unthunk(value)

"""
    ChainRulesCore.frule((_, changes...), d::Differentiable, arguments...)

The results of the CasADi function and their changes for the changes of the
arguments, from one call of generated code.
"""
function ChainRulesCore.frule((_, changes...), d::Differentiable, arguments...)
    n = casadi.n_out(d.f)
    given = ntuple(i -> array(changes[i], casadi.sparsity_in(d.f, i - 1)), length(arguments))
    all = directional(d, :pushforward)(arguments..., given...)
    n == 1 && return all[1], all[2]
    value = all[1:n]
    value, Tangent{typeof(value)}(all[n+1:2n]...)
end

"""
    ChainRulesCore.rrule(d::Differentiable, arguments...)

The results of the CasADi function, and the function that maps one seed per
result to one sum per argument, from generated code for the products with the
transposed Jacobians.
"""
function ChainRulesCore.rrule(d::Differentiable, arguments...)
    n = casadi.n_out(d.f)
    function pullback(seeds)
        seeds = unthunk(seeds)
        part(o) = seeds isa AbstractZero ? seeds : n == 1 ? seeds : seeds[o]
        given = ntuple(o -> array(part(o), casadi.sparsity_out(d.f, o - 1)), n)
        sums = directional(d, :pullback)(arguments..., given...)
        (NoTangent(), (length(arguments) == 1 ? (sums,) : sums)...)
    end
    d(arguments...), pullback
end

end # module
