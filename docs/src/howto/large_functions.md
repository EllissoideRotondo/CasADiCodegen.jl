# Large functions and the cache on disk

A model with tens of thousands of instructions compiles slowly, and Julia compiles
it again in each session. This page shows the remedies of the package, which are a
split of the code into sub-functions, a lower optimization level and a cache of
compiled code on disk, and the setting of the tests that keeps the compilation short.

## Split a large function

Julia compiles one large function slowly, because the time grows faster than the
number of instructions, so above a limit the code is split into sub-functions that
share one work vector, which holds the intermediate values. The work vector belongs
to the Julia task, so a call does not allocate it, and each form of the generated
function (see [Choose a form](@ref)) calls the same sub-functions. Two keywords of
`build_function` and `generate_julia` control the split:

| Keyword | Meaning | Default |
|---|---|---|
| `split_above` | A function with more instructions is split | 50,000 for scalar code; 1,000 for matrix code |
| `chunk_size` | Largest number of instructions in one sub-function | 1,000 |

In matrix code, an instruction is one operation of the MX graph, such as a matrix
product or a call. When only `chunk_size` is given, each function with more
instructions than `chunk_size` is split; `split_above = typemax(Int)` keeps any function in one piece.

The split can shorten the first call of a large model by a factor of six, but it
makes the later calls 15 to 40 percent slower. Raise `split_above` when the speed of
the later calls matters more than the first call; [Split of a large function](@ref)
gives the measurements behind the defaults.

## Compile at a lower optimization level

For a function that is built in the session, compiling the generated body at the first
call takes most of the time, while CasADi and the emission of the code take
milliseconds. The keyword `optlevel` of `build_function` makes Julia compile the
returned functions at a lower optimization level, as `Base.Experimental.@optlevel`
sets it for a module, which suits a model that changes often:

```julia
g, g! = build_function(f; expression = Val{false}, optlevel = 1)
```

| `optlevel` | First call | Later calls |
|---|---|---|
| 2 (the default of Julia) | 0.41 s | 1.55 µs |
| 1 | 0.27 s | 1.60 µs |
| 0 | 0.21 s | 2.07 µs |

The times are those of the right-hand side of a second-order state transition
tensor of the two-body problem with the J2 term, 258 states, called with static arrays
on Julia 1.13. `generate_julia` takes the same keyword and writes the level into the
module, and with `cache` the cached package is compiled at that level.

## Keep the compiled code between sessions

`build_function` with `expression = Val{false}` returns functions made with
[RuntimeGeneratedFunctions.jl](https://github.com/SciML/RuntimeGeneratedFunctions.jl),
which Julia compiles again in each session. With
`cache = true`, `build_function` writes the generated module into a small package,
which Julia compiles once and whose machine code it stores, so that a later session
with the same model loads the functions and compiles nothing.

```julia
g, g! = build_function(f; expression = Val{false}, cache = true)
g, g! = build_function(f; expression = Val{false}, cache = "/path/to/cache")   # another directory
```

| Function | Without cache, each session | With cache, first session | With cache, later sessions |
|---|---|---|---|
| Chain of maps, 22,236 instructions | 6.2 s | 16.8 s | 1.1 s |
| Chain of maps, 67,236 instructions | 16.4 s | 19.7 s | 1.3 s |
| Two-body integrator, final state and its Jacobian | 4.1 s | 12.8 s | 2.0 s |

Each time is that of `build_function` together with the first call, and the speed of
the later calls does not change. The name of the package comes from the generated
code, so a changed model or other options give a new package, and an old one is
never used by mistake. The packages are in the directory `scratchspaces` of the first
Julia depot (`~/.julia` by default), or in the directory that `cache` gives, and they
can be deleted at any time. Four rules apply.

- The package is compiled for `Float64` on ordinary arrays; `force_SA = true` compiles
  the out-of-place function for static arrays, and `nonzeros = true` the function on
  vectors of nonzeros. Other argument types compile at their first call, as before.
- The functions belong to a module that is loaded during the call of `build_function`.
  Code that runs already, such as the function that called `build_function`, must
  call them with `Base.invokelatest`; code that starts later calls them directly.
- A Julia function in the graph must be a named function of a package, as for
  `generate_julia`. An integrator or a rootfinder goes into the package with its
  algorithm and its options, and the package also holds the machine code of the
  solver for `Float64`, so that a later session does not compile the solver again.
- An input or an output with structural zeros makes the module import SparseArrays,
  which must then be in the environment.

## Keep bounds checks off in tests

With `--check-bounds=yes`, Julia checks each array access of the generated code, so
the first call of a large function compiles much longer. `Pkg.test` sets this option on
Julia 1.12; on Julia 1.13 it keeps the setting of the session. For integrators of
CasADi with a fixed step, measured once:

| First call of | Julia 1.12 | Julia 1.12, `--check-bounds=yes` | Julia 1.13 | Julia 1.13, `--check-bounds=yes` |
|---|---|---|---|---|
| `rk` with Jacobians | 2.6 s | 23.7 s | 2.1 s | 8.3 s |
| `collocation` with Jacobians | 4.3 s | 41.5 s | 3.4 s | 11.5 s |
| `collocation` with derivatives of the second order | 16.6 s | 533 s | 14.6 s | 144 s |

On Julia 1.12, run such tests with `Pkg.test(...; julia_args = ["--check-bounds=auto"])`.
