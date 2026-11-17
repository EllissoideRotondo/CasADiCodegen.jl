---
name: julia-test-quality
description: Use when adding, changing, reviewing or trimming tests of CasADiCodegen.jl, or when a fault is fixed and needs a regression test; the rules of this repository on top of the global julia-tests skill.
---

# CasADiCodegen test rules

The general rules (one fault per test, regression tests, trimming) are in the global skill
`julia-tests`. This file holds what is particular to CasADiCodegen.jl.

## Oracle

CasADi is the oracle. A test compares the generated code with CasADi evaluating the same
function on the same inputs, through the helpers of `test/helpers.jl` (`casadi_eval`,
`mx_casadi`, `close_enough`), not with numbers written in the test. A value that CasADi
cannot give (a Julia callback, a SciML solver) is compared with the plain Julia computation.

- **Bit for bit** (`isequal`) for operations that IEEE 754 rounds exactly (`EXACT_OPS` of
  `test_casadi.jl`), for rounding, comparisons and selections, and for graphs of such
  operations. `isequal` also checks signed zeros and NaN.
- **`close_enough`** (4 ulps) for the elementary functions, which C libraries round
  differently.
- **A tolerance with a reason** for anything else: an iterative solver, a different
  factorization, an integrator. Write the reason in a comment beside it.

## Platforms

The suite runs on Linux (x86), macOS (ARM) and Windows (x86). CasADi itself differs there,
and a test must say so rather than loosen everywhere:

- The C library of Windows rounds `sin`, `cos`, `tan` and `pow` less closely; the
  comparison falls back to Julia's function there (`JULIA_OPS` of `test_casadi.jl`).
- The C compiler of CasADi on ARM fuses products and sums; a test of a routine read from
  CasADi's C code compares with `rtol = 1e-14` on `Sys.ARCH === :aarch64` only.
- A conversion of NaN or of a value out of range to an integer follows the processor; the
  generated code does too, so such tests stay exact.
- A file that CasADi writes ends its lines with CR LF on Windows.
- A test that compiles C code finds the compiler with `Sys.which` and skips without one.

Prefer inputs whose computation is exact on every platform (arithmetic only, no
elementary functions) when the test is about the code structure, not about an operation.

## Inputs

- Special values go through `SPECIAL_VALUES` of `test_casadi.jl`: signed zeros, the
  extremes, infinities, NaN, values next to the boundaries of a domain.
- A property test (Supposition.jl) covers one operation or one rule over random inputs; it
  ignores the cases that C leaves open (the sign of `fmin(0.0, -0.0)`) instead of failing on
  them, and says so in a comment.
- A translation from Julia (Traced values, Symbolics.jl) is tested against `Base` with the
  same table of cases for both front ends (`JULIA_MOD_CASES`, `JULIA_CLAMP_CASES` of
  `test/helpers.jl`).

## Groups and time

Each file `test/test_<group>.jl` runs in a process of its own (`test/runtests.jl`); a
group's time is mostly compilation. A new test goes into the group of its subject. A test
that compiles a large function or runs a solver for long goes under `EXHAUSTIVE`, unless it
is the only test of a feature. Keep a group under about two minutes on CI.

## Checks before the work is done

```sh
julia -e 'using Pkg; Pkg.test("CasADiCodegen"; test_args = ["mx"])'   # the changed groups
julia -e 'using Pkg; Pkg.test("CasADiCodegen")'                       # all groups
```

Codecov reports the coverage of each pull request: every changed line that can run should
be covered. A branch kept only for safety that no test reaches is a sign to remove it, or to
find the input that reaches it.
