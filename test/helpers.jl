# Names that more than one group of tests uses. Each group runs in a process
# of its own, so a group cannot take a helper from the file of another group.
using Test
using CasADiCodegen
using Supposition
using Supposition: Data

const casadi = CasADiCodegen.casadi
# The exhaustive tier: checks that repeat a behavior of the default run in a case that compiles
# much code again, mostly a solver for dual numbers. The environment variable CASADI_TEST_EXHAUSTIVE selects it.
const EXHAUSTIVE = haskey(ENV, "CASADI_TEST_EXHAUSTIVE")
const Algorithm = CasADiCodegen.Codegen.Algorithm
const I = CasADiCodegen.Codegen.Instruction

compiled(alg; kwargs...) = build_function(alg; nonzeros = true, expression = Val{false}, kwargs...)
results(alg) = [fill(NaN, n) for n in alg.nnz_out]

casadi_nonzeros(r::casadi.DM) = [casadi.nonzeros(r)]
casadi_nonzeros(r) = [casadi.nonzeros(d) for d in r]

"Evaluate the CasADi function numerically; one vector of nonzeros per output."
casadi_eval(f, args...) = casadi_nonzeros(f(map(a -> casadi.DM(a), args)...))

close_enough(a, b) = isequal(a, b) || (isnan(a) && isnan(b)) ||
                     abs(a - b) <= 4 * eps(max(abs(a), abs(b)))

# Operation => (arity, builder on CasADi scalars).
const OP_BUILDERS = Dict{Symbol,Tuple{Int,Function}}(
    :OP_ADD => (2, (x, y) -> x + y), :OP_SUB => (2, (x, y) -> x - y),
    :OP_MUL => (2, (x, y) -> x * y), :OP_DIV => (2, (x, y) -> x / y),
    :OP_NEG => (1, x -> -x), :OP_EXP => (1, x -> casadi.exp(x)), :OP_LOG => (1, x -> casadi.log(x)),
    :OP_POW => (2, (x, y) -> x^y), :OP_CONSTPOW => (1, x -> casadi.constpow(x, SX(2.5))),
    :OP_SQRT => (1, x -> casadi.sqrt(x)), :OP_SQ => (1, x -> x^2), :OP_TWICE => (1, x -> 2 * x),
    :OP_SIN => (1, x -> casadi.sin(x)), :OP_COS => (1, x -> casadi.cos(x)), :OP_TAN => (1, x -> casadi.tan(x)),
    :OP_ASIN => (1, x -> casadi.asin(x)), :OP_ACOS => (1, x -> casadi.acos(x)), :OP_ATAN => (1, x -> casadi.atan(x)),
    :OP_LT => (2, (x, y) -> casadi.lt(x, y)), :OP_LE => (2, (x, y) -> casadi.le(x, y)),
    :OP_EQ => (2, (x, y) -> casadi.eq(x, y)), :OP_NE => (2, (x, y) -> casadi.ne(x, y)),
    :OP_NOT => (1, x -> casadi.logic_not(x)), :OP_AND => (2, (x, y) -> casadi.logic_and(x, y)),
    :OP_OR => (2, (x, y) -> casadi.logic_or(x, y)),
    :OP_FLOOR => (1, x -> casadi.floor(x)), :OP_CEIL => (1, x -> casadi.ceil(x)),
    :OP_FMOD => (2, (x, y) -> casadi.fmod(x, y)), :OP_FABS => (1, x -> casadi.abs(x)),
    :OP_SIGN => (1, x -> casadi.sign(x)), :OP_COPYSIGN => (2, (x, y) -> casadi.copysign(x, y)),
    :OP_IF_ELSE_ZERO => (2, (x, y) -> casadi.if_else(x, y, SX(0.0))),
    :OP_ERF => (1, x -> casadi.erf(x)), :OP_FMIN => (2, (x, y) -> casadi.fmin(x, y)),
    :OP_FMAX => (2, (x, y) -> casadi.fmax(x, y)), :OP_INV => (1, x -> 1 / x),
    :OP_SINH => (1, x -> casadi.sinh(x)), :OP_COSH => (1, x -> casadi.cosh(x)), :OP_TANH => (1, x -> casadi.tanh(x)),
    :OP_ASINH => (1, x -> casadi.asinh(x)), :OP_ACOSH => (1, x -> casadi.acosh(x)),
    :OP_ATANH => (1, x -> casadi.atanh(x)), :OP_ATAN2 => (2, (x, y) -> casadi.atan2(x, y)),
    :OP_ERFINV => (1, x -> casadi.erfinv(x)), :OP_LOG1P => (1, x -> casadi.log1p(x)),
    :OP_EXPM1 => (1, x -> casadi.expm1(x)), :OP_HYPOT => (2, (x, y) -> casadi.hypot(x, y)),
    :OP_REMAINDER => (2, (x, y) -> casadi.remainder(x, y)),
)

# Whole range, plus values near the origin where most domains have their edges.
# Every NaN is replaced by the ordinary quiet NaN: the generator also produces
# signalling NaNs, for which C `fmin` and `fmax` return NaN instead of the other
# operand, and no real computation produces a signalling NaN.
const any_float = map(x -> isnan(x) ? NaN : x, Data.Floats{Float64}())
const small_float = map(x -> 8 * (x - 0.5), Data.Floats{Float64}(; minimum = 0.0, maximum = 1.0, nans = false))
const test_float = Data.OneOf(any_float, small_float, Data.SampledFrom([0.0, -0.0, 1.0, -1.0, 0.5, 2.0, 3.0, -2.0]))

nnz_in(f) = [casadi.nnz(casadi.sparsity_in(f, i)) for i in 0:casadi.n_in(f)-1]
nnz_out(f) = [casadi.nnz(casadi.sparsity_out(f, j)) for j in 0:casadi.n_out(f)-1]

"Evaluate the CasADi function on one vector of nonzeros per input."
mx_casadi(f, nz) = casadi_nonzeros(
    f((casadi.DM(casadi.sparsity_in(f, i - 1), casadi.DM(nz[i])) for i in eachindex(nz))...))

# Fixed inputs with distinct values; the front end turns -0.0 into 0.0, so none is zero.
mx_inputs(f) = [[0.37 * k - 1.3 + 0.11 * i for k in 1:n] for (i, n) in enumerate(nnz_in(f))]

dense_values(m) = casadi.nonzeros(casadi.densify(m))
