# Large functions and the cache on disk

A model with tens of thousands of instructions compiles slowly, and Julia compiles it
again in each session; this page shows the remedies.

## Split a large function

Above a limit, the generated code is split into sub-functions that share one work
vector. Two keywords of `build_function` and `generate_julia` control the split:

| Keyword | Meaning | Default |
|---|---|---|
| `split_above` | A function with more instructions is split | 50,000 for scalar code; 1,000 for matrix code |
| `chunk_size` | Largest number of instructions in one sub-function | 1,000 |

In matrix code, an instruction is one operation of the MX graph;
`split_above = typemax(Int)` keeps any function in one piece. The split can shorten
the first call by a factor of six but makes later calls 15 to 40 percent slower (see
[Split of a large function](@ref)).

## Compile at a lower optimization level

For a model that changes often, pass `optlevel` to compile the returned functions
faster:

```julia
g, g! = build_function(f; expression = Val{false}, optlevel = 1)
```

| `optlevel` | First call | Later calls |
|---|---|---|
| 2 (the default of Julia) | 0.41 s | 1.55 µs |
| 1 | 0.27 s | 1.60 µs |
| 0 | 0.21 s | 2.07 µs |

The times are those of a two-body model with 258 states on Julia 1.13.
`generate_julia` and `cache` honor the same keyword.

## Keep the compiled code between sessions

With `cache`, `build_function` stores the generated code as a precompiled package,
so that a later session with the same model compiles nothing:

```julia
g, g! = build_function(f; expression = Val{false}, cache = true)
g, g! = build_function(f; expression = Val{false}, cache = "/path/to/cache")   # another directory
```

| Function | Without cache, each session | With cache, first session | With cache, later sessions |
|---|---|---|---|
| Chain of maps, 22,236 instructions | 6.2 s | 16.8 s | 1.1 s |
| Chain of maps, 67,236 instructions | 16.4 s | 19.7 s | 1.3 s |
| Two-body integrator, final state and its Jacobian | 4.1 s | 12.8 s | 2.0 s |

Each time includes `build_function` and the first call. A changed model gets a new
package; the packages live in `scratchspaces` of the first Julia depot and can be
deleted at any time.

- The package is compiled for `Float64` on ordinary arrays, plus static arrays with
  `force_SA = true` and vectors of nonzeros with `nonzeros = true`; other argument
  types compile at their first call.
- The caller of `build_function` must call the functions with `Base.invokelatest`.
- A Julia function in the graph must be a named function of a package.
- Structural zeros in an input or output require SparseArrays in the environment.

## Keep bounds checks off in tests

`--check-bounds=yes` makes the first call of a large function compile much longer;
`Pkg.test` sets it on Julia 1.12 but not on 1.13. For fixed-step integrators of CasADi:

| First call of | Julia 1.12 | Julia 1.12, `--check-bounds=yes` | Julia 1.13 | Julia 1.13, `--check-bounds=yes` |
|---|---|---|---|---|
| `rk` with Jacobians | 2.6 s | 23.7 s | 2.1 s | 8.3 s |
| `collocation` with Jacobians | 4.3 s | 41.5 s | 3.4 s | 11.5 s |
| `collocation` with derivatives of the second order | 16.6 s | 533 s | 14.6 s | 144 s |

On Julia 1.12, run such tests with `Pkg.test(...; julia_args = ["--check-bounds=auto"])`.
