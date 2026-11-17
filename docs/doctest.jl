# The doctests of the docstrings only, without building the manual:
# julia --project=docs docs/doctest.jl
using Documenter, CasADiCodegen
# Symbolics.jl loads its extension here, so that its precompilation prints nothing in a doctest.
import SciMLBase, ChainRulesCore, SparseArrays, OptimizationIpopt, BifurcationKit, Symbolics

extension(name) = Base.get_extension(CasADiCodegen, name)
modules = [CasADiCodegen, CasADiCodegen.Codegen, CasADiCodegen.Models, extension(:CasADiCodegenSciMLBaseExt),
           extension(:CasADiCodegenChainRulesCoreExt), extension(:CasADiCodegenSparseArraysExt)]
doctest(nothing, modules)
