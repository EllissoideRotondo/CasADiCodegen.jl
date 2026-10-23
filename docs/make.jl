using Documenter
using CasADiCodegen
import SciMLBase, ChainRulesCore, SparseArrays, OptimizationIpopt, BifurcationKit

extension(name) = Base.get_extension(CasADiCodegen, name)

makedocs(;
    sitename = "CasADiCodegen.jl",
    modules = [CasADiCodegen, CasADiCodegen.Codegen, extension(:CasADiCodegenSciMLBaseExt), extension(:CasADiCodegenChainRulesCoreExt), extension(:CasADiCodegenSparseArraysExt), extension(:CasADiCodegenOptimizationBaseExt), extension(:CasADiCodegenBifurcationKitExt)],
    format = Documenter.HTML(;
        prettyurls = get(ENV, "CI", "false") == "true",
        canonical = "https://AstroGuild.github.io/CasADiCodegen.jl",
        edit_link = "main",
    ),
    # The front end exports names with docstrings of its own, which are not part of this manual.
    checkdocs = :none,
    pages = [
        "Home" => "index.md",
        "Getting started" => "getting_started.md",
        "Tutorials" => [
            "tutorials/ode_modeling.md",
            "tutorials/nonlinear.md",
            "tutorials/optimization.md",
            "tutorials/sparse_jacobians.md",
            "tutorials/dae_models.md",
            "tutorials/initialization.md",
            "tutorials/events.md",
            "tutorials/stochastic.md",
            "tutorials/discrete.md",
            "tutorials/callable_parameters.md",
            "tutorials/bifurcation.md",
            "tutorials/traced_problems.md",
            "tutorials/linear_analysis.md",
            "tutorials/independent_variable.md",
            "tutorials/perturbation.md",
            "tutorials/attractors.md",
            "tutorials/parameter_estimation.md",
            "tutorials/dynamic_optimization.md",
            "tutorials/large_models.md",
        ],
        "Models" => [
            "howto/load_model.md",
            "howto/traced_models.md",
            "howto/callbacks.md",
            "howto/integrators.md",
            "howto/events.md",
            "howto/rootfinders.md",
            "howto/linearize.md",
        ],
        "Solvers and SciML" => [
            "howto/casadi_solvers.md",
            "howto/sciml_functions.md",
            "howto/autodiff_rules.md",
        ],
        "Large models and debugging" => [
            "howto/large_functions.md",
            "howto/inspect.md",
        ],
        "Concepts" => [
            "explanation/matrix_code.md",
            "explanation/integrator_derivatives.md",
            "explanation/performance.md",
            "explanation/scope.md",
        ],
        "API reference" => ["api.md", "api_sciml.md"],
    ],
)

deploydocs(;
    repo = "github.com/AstroGuild/CasADiCodegen.jl",
    devbranch = "main",
)
