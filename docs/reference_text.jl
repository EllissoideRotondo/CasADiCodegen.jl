# The docstrings of the reference pages as plain text, one file per page, for a reader
# review: julia --project=docs docs/reference_text.jl <directory>
using CasADiCodegen, Markdown, Documenter
# Symbolics.jl loads its extension here, so that its precompilation prints nothing in a doctest.
import SciMLBase, ChainRulesCore, SparseArrays, OptimizationIpopt, BifurcationKit, Symbolics

using Documenter: DocSystem

# The docstrings of an entry of a `@docs` block, found as Documenter finds them.
function entry_doc(entry::AbstractString)
    ex = Meta.parse(entry)
    binding = DocSystem.binding(Main, ex)
    typesig = Core.eval(Main, DocSystem.signature(ex, entry))
    docs = DocSystem.getdocs(binding, typesig)
    isempty(docs) && error("no docstring for the entry $entry")
    Markdown.MD([Base.Docs.parsedoc(d) for d in docs])
end

out = mkpath(get(ARGS, 1, "reference_text"))
for page in ("api.md", "api_sciml.md")
    text = read(joinpath(@__DIR__, "src", page), String)
    open(joinpath(out, replace(page, ".md" => ".txt")), "w") do io
        # The text of the page, with each `@docs` block replaced by its docstrings.
        last_end = 1
        for block in eachmatch(r"```@docs\n(.*?)```"s, text)
            print(io, text[last_end:prevind(text, block.offset)])
            for entry in split(block[1], '\n'; keepempty = false)
                println(io, "=" ^ 92, "\n", strip(entry), "\n", "=" ^ 92)
                println(io, Markdown.plain(entry_doc(strip(entry))))
            end
            last_end = block.offset + ncodeunits(block.match)
        end
        print(io, text[last_end:end])
    end
end
