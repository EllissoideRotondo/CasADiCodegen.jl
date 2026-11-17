---
name: julia-doc-quality
description: Use when writing or changing the README, a docstring or a manual page of CasADiCodegen.jl, or when adding or renaming a public name; the rules of this repository on top of the global julia-docstrings, julia-manual, julia-readme and julia-doc-reader-review skills.
---

# CasADiCodegen documentation rules

The general rules (prose registers, docstring form, README content, reader review) are in
the global skills `julia-docstrings`, `julia-manual`, `julia-readme` and
`julia-doc-reader-review`. This file holds what is particular to CasADiCodegen.jl. The
models are the manuals of ModelingToolkit.jl and the SciML style guide: a first tutorial
that covers the common case from start to end, tutorials before reference, doctests on,
and a docstring template for every public name.

## Where a fact lives

Each fact has one home, and other places link to it.

| Fact | Home |
|---|---|
| What the package is, who needs it, install, one worked example, where to go next | `README.md` |
| The common workflow, from a CasADi function to generated code | `docs/src/getting_started.md` |
| A problem solved from start to end, after the pattern of a ModelingToolkit tutorial | `docs/src/tutorials/` |
| One task (load a model, call a Julia function, replace a solver) | `docs/src/howto/` |
| How the code is made and why, costs, limits, differences from CasADi | `docs/src/explanation/` |
| What one call takes, returns, changes and throws | its docstring |
| Internal interfaces and invariants for maintainers | private docstrings and comments |

The scope page (`explanation/scope.md`) is the one list of what is and is not supported.
Another page that names a limit links to it instead of restating the list.

## Docstring form for a public name

```julia
"""
    name(arg1, arg2; keyword = default) -> result

Return <what>, as <the CasADi or SciML counterpart, if there is one>.

<Only what the caller must know: mutation, shapes, defaults, numerical limits, edge cases.
Leave out what the signature already says.>

# Keywords
- `keyword = default`: <meaning>. Only when the keywords need more than one sentence.

Throws an `ArgumentError` if <condition>.

# Examples
```jldoctest
julia> using CasADiCodegen

julia> x = SX.sym("x");

julia> name(x)
...
```

See also [`other`](@ref).
"""
```

- Lines at most 92 characters.
- A public name is one the package exports (`build_function`, `generate_julia`,
  `CasadiFunction`, `casadi_function`), one that the manual calls with the prefix
  `CasADiCodegen.`, or a method that an extension adds to a function of SciMLBase.jl or of
  the CasADi front end. Each has a docstring and appears once in `docs/src/api.md` or
  `docs/src/api_sciml.md`; `docs/make.jl` fails the build otherwise.
- A method that an extension adds to a function of another package (`SciMLBase.ODEProblem`,
  `integrator`, `rootfinder`) gets a docstring on that method, with the signature of the
  method, and the reference lists it with its signature.
- Several methods of one call: one docstring with one signature line per call form.
- "Throws": name the error type and the condition, `Throws an ArgumentError if ...`. More
  than three conditions: `Throws an ArgumentError:` and a list, one line each. A `nothing`
  result or a warning belongs in the body.
- State units where a call has physical quantities (times of an integrator, tolerances).
- State the differences from CasADi that a caller can observe, in fixed words: "as CasADi
  does", "CasADi gives ...; this call gives ...", "This call has no counterpart in CasADi."
- `# Arguments` only when the signature and one sentence cannot carry the meaning.
- An example is wanted for a main entry point and for behaviour that is not obvious; an
  accessor (`unknowns`, `parameters`) needs none.
- Never write "Contract:" in a public docstring.

## Facts are measured, not copied

Before a docstring says what a call throws or returns:

- Provoke each error with a bad input and read the type and the message; an error often
  comes from a lower layer (the front end, SciMLBase) and differs between sibling calls.
- Never write "throws as [`other`](@ref)" without a comparison of the two in the code.
- A number in a docstring (a time, an accuracy, a limit such as 1024 elements) comes from a
  run or from a constant in the code; cite the constant when there is one.
- A kept docstring that states something false is corrected, private or not; report the
  correction.

## Examples

- `jldoctest` when the example builds in a few seconds and its output is stable: pure
  CasADi graphs and `build_function` with `expression = Val{false}`. Each doctest starts
  with `julia> using CasADiCodegen`, and a doctest that needs Symbolics.jl or SciMLBase.jl
  loads it in its own first lines.
- A plain `julia` block for an example that runs a SciML solver, compiles for long or
  depends on the machine. Run it once in a scratch script before it goes in.
- Show a result that prints the same everywhere: a comparison (`g(x) == [6.0]`, `≈` for a
  rounded value), a small `Vector`, or `round.(x; digits = 6)`. A value that two code paths
  compute can differ in the last bit: compare with `≈`.
- Write the expected output from a scratch run. Never use `doctest(...; fix = true)`.
- No `$` and no backslash in a docstring that is not `raw`.

## Private functions

A private function keeps a docstring only when it states something the code and the name do
not: a precondition, an invariant, a rule of CasADi that the code follows, a known
difference from CasADi. A one-line docstring on a private constant becomes a trailing `#`
comment or goes.

## Manual

Pages follow reader needs, not `src/` file names. A new tutorial follows the pattern of
the existing ones: the problem in one paragraph, the model, the generated code, the result
checked against CasADi or the SciML solver, and what to read next. Examples run
(`@example`, `jldoctest`); try each in a scratch script first, and after the build read
the outputs in `docs/build/` once: a result that surprises is a fault of the example or of
the text beside it.

## Checks before the work is done

```sh
julia --project=docs docs/make.jl                        # build, doctests, references, completeness
julia --project=docs docs/reference_text.jl <directory>  # the reference pages as text
```

A change of docstrings only must not change code. Prove it: parse the old and the new file
with `Meta.parseall`, replace each docstring macro call by its target, remove the
`LineNumberNode`s, and require equality.

Do not pass `warnonly = true`, and do not accept a changed doctest output without reading
why it changed. Then review the changed pages with the global `julia-doc-reader-review`
skill, with the inputs below.

## Reader review: what the reader gets

- README and manual pages: the Markdown files themselves.
- Docstrings: the files of `docs/reference_text.jl`. Split the pages between readers so
  that each reads at most about 20,000 words in full.
- A reader agent cannot write a file: ask for the full list of findings in its reply, one
  line each as `page / call: CUT | CLARIFY | MOVE | WRONG? : what`, with the quoted words.

A `WRONG?` finding is two entries that disagree. Run the call before any edit: the text can
be right and the code inconsistent, and then the text stays and the inconsistency goes to
the report.
