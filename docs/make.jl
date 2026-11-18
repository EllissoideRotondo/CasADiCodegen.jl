using Documenter
using CasADiCodegen
# Symbolics.jl loads its extension here, so that its precompilation prints nothing in a doctest.
import SciMLBase, ChainRulesCore, SparseArrays, OptimizationIpopt, BifurcationKit, Symbolics

extension(name) = Base.get_extension(CasADiCodegen, name)

# `checkdocs` cannot tell the names of the package from those that it takes from the CasADi
# front end, so the reference is checked here: each exported or public name of the package,
# and each name that `CasADiCodegen.Models` exports, must be an entry of a `@docs` block.
let front = Set(names(CasADiCodegen.Codegen.Front)),
    # `casadi` is the module of the front end, whose page is the CasADi documentation.
    own = [string(n) for n in names(CasADiCodegen) if !(n in front) && n !== :CasADiCodegen && n !== :casadi],
    models = ["Models." * string(n) for n in names(CasADiCodegen.Models) if n !== :Models],
    entries = join((m[1] for page in ("api.md", "api_sciml.md")
                    for m in eachmatch(r"```@docs\n(.*?)```"s, read(joinpath(@__DIR__, "src", page), String))), "\n")
    missing = [n for n in [own; models] if !occursin(Regex("(^|[.\\s])" * replace(n, "." => "\\.") * "([(\\s]|\$)", "m"), entries)]
    isempty(missing) || error("the reference lists no docstring of: ", join(missing, ", "))
end

extensions = [
    extension(:CasADiCodegenSciMLBaseExt),
    extension(:CasADiCodegenChainRulesCoreExt),
    extension(:CasADiCodegenSparseArraysExt),
    extension(:CasADiCodegenOptimizationBaseExt),
    extension(:CasADiCodegenBifurcationKitExt),
]

makedocs(;
    sitename = "CasADiCodegen.jl",
    modules = [CasADiCodegen, CasADiCodegen.Codegen, CasADiCodegen.Models, extensions...],
    format = Documenter.HTML(;
        prettyurls = get(ENV, "CI", "false") == "true",
        canonical = "https://orbitology.github.io/CasADiCodegen.jl",
        edit_link = "main",
        collapselevel = 1,
    ),
    # The front end exports names with docstrings of its own, which are not part of this manual.
    checkdocs = :none,
    pages = [
        "Home" => "index.md",
        "Getting started" => "getting_started.md",
        "Tutorials" => [
            "tutorials/traced_problems.md",
            "tutorials/parameter_estimation.md",
            "tutorials/dae_models.md",
            "tutorials/initialization.md",
            "tutorials/events.md",
            "tutorials/bang_bang.md",
            "tutorials/large_models.md",
        ],
        # Each group folds in the sidebar (`collapselevel = 1`) until one of its pages is open.
        "How-to guides" => [
            "Write or import a model" => [
                "howto/equations.md",
                "howto/load_model.md",
                "howto/symbolics.md",
                "howto/traced_models.md",
                "howto/callbacks.md",
            ],
            "Integrate, find roots and handle events" => [
                "howto/integrators.md",
                "howto/events.md",
                "howto/discontinuities.md",
                "howto/rootfinders.md",
                "howto/casadi_solvers.md",
            ],
            "Use the code with SciML and AD packages" => [
                "howto/sciml_functions.md",
                "howto/autodiff_rules.md",
                "howto/linearize.md",
            ],
            "Compile large models and debug" => [
                "howto/large_functions.md",
                "howto/inspect.md",
            ],
        ],
        "How the generated code works" => [
            "explanation/matrix_code.md",
            "explanation/integrator_derivatives.md",
            "explanation/performance.md",
            "explanation/scope.md",
        ],
        "API reference" => ["api.md", "api_sciml.md"],
    ],
)

deploydocs(;
    repo = "github.com/Orbitology/CasADiCodegen.jl",
    devbranch = "main",
)
