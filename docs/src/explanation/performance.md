# Cost of the generated code

This page gives the measurements behind the defaults of the package, whose conditions
are in [Reproducibility](@ref): compile time at the first call, which grows with the
model, and run time at later calls, which depends on the form of the code.

## Scalar code and its forms

An instruction is one operation as CasADi counts them: a scalar operation in an SX
function, a graph operation such as a matrix product or a call in an MX function.
Scalar code, from an SX function or an expanded MX function, allocates nothing in
place or out of place with static arrays, whatever its size; matrix code is described
in [Matrix code and functions without instructions](@ref). Two scalar models with
their 6 by 6 Jacobians, measured once on Julia 1.13:

| Instructions | Form | First call | Later calls | Memory per call |
|---|---|---|---|---|
| 5,361 | Out of place, static arrays | 0.62 s | 1.1 µs | 0 B |
| 5,361 | In place, arrays | 1.04 s | 1.2 µs | 0 B |
| 5,361 | Out of place, arrays | 0.68 s | 1.2 µs | 576 B |
| 18,861 | Out of place, static arrays | 3.69 s | 3.8 µs | 0 B |
| 18,861 | In place, arrays | 6.24 s | 3.9 µs | 0 B |
| 18,861 | Out of place, arrays | 3.82 s | 4.1 µs | 576 B |

The first call is mostly compilation. For scalar code above about 50,000 instructions,
or matrix code above about 1,000, see [Large functions and the cache on disk](@ref).

## Split of a large function

Above `split_above` instructions, `build_function` and `generate_julia` split the code
into sub-functions of at most `chunk_size` instructions (see
[Large functions and the cache on disk](@ref)). The defaults rest on the measurements
below, made once on Julia 1.13 for the function on nonzeros of two models with dense
Jacobians: a chain of `map`s, each applied to the result of the one before, and the
acceleration of 40 bodies under mutual gravity.

| Model | Instructions | Sub-function size | First call | Later calls |
|---|---|---|---|---|
| Chain of maps, 6 states | 67,236 | One function | 35 s | 21 µs |
| Chain of maps, 6 states | 67,236 | 500 | 14 s | 28 µs |
| Chain of maps, 6 states | 67,236 | 1,000 | 15 s | 27 µs |
| Chain of maps, 6 states | 67,236 | 5,000 | 26 s | 28 µs |
| Chain of maps, 6 states | 67,236 | 20,000 | 34 s | 25 µs |
| Gravity of 40 bodies | 102,946 | One function | 54 s | 70 µs |
| Gravity of 40 bodies | 102,946 | 500 | 39 s | 97 µs |
| Gravity of 40 bodies | 102,946 | 1,000 | 53 s | 96 µs |
| Gravity of 40 bodies | 102,946 | 5,000 | 63 s | 98 µs |
| Chain of maps, 6 states | 224,736 | One function | 339 s | 89 µs |
| Chain of maps, 6 states | 224,736 | 1,000 | 54 s | 105 µs |
| Chain of maps, 6 states | 224,736 | 5,000 | 92 s | 102 µs |
| Chain of maps, 6 states | 224,736 | 20,000 | 186 s | 102 µs |

Once a function is split, the sub-function size barely affects later calls, which
the split makes 15 to 40 percent slower, while the first call slows as the size grows.
Whether splitting shortens the first call depends on the model: the largest chain
compiles in 54 s instead of 339 s, while the gravity model gains nothing at the
default size. Smaller versions of the two models near the default limit of 50,000
instructions, not in the table, took 18 s and 14 s unsplit against 10 s and 22 s split.
