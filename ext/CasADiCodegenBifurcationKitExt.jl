module CasADiCodegenBifurcationKitExt

import BifurcationKit
import CasADiCodegen
import SciMLBase
using CasADiCodegen.Codegen: @requires, @ensures

const C = CasADiCodegen.C
const SII = SciMLBase.SymbolicIndexingInterface

"The extension for SciMLBase.jl, whose `NonlinearFunction` and names this extension builds on."
sciml() = Base.get_extension(CasADiCodegen, :CasADiCodegenSciMLBaseExt)

"""
    Residual(f)

The in-place residual `f(out, u, p)` of a `NonlinearFunction`, as BifurcationKit.jl calls it:
in place it returns `out`, and out of place `(u, p)` it returns a new vector of the element
type of `u` and `p`.
"""
struct Residual{F}
    f::F
    n::Int
end
(r::Residual)(out, u, p) = (r.f(out, u, p); out)
(r::Residual)(u, p) = r(zeros(promote_type(Float64, eltype(u), eltype(p)), r.n), u, p)
SciMLBase.numargs(::Residual) = [3, 2]

"""
    Derivative(g!, n)

The generated in-place function `g!((out,), (u, p, directions...))` of a derivative of the
residual, multilinear in the `directions`, called as `(u, p, directions...)`: it returns a
new vector of length `n`. Complex directions, which BifurcationKit.jl passes for the normal
form of a Hopf point, give the sum over the real and imaginary parts of each direction, each
imaginary part with a factor `im`.
"""
struct Derivative{G}
    g!::G
    n::Int
end
(d::Derivative)(u, p, directions...) = evaluate(d, u, p, directions)
function evaluate(d::Derivative, u, p, directions::Tuple{Vararg{AbstractArray{<:Real}}})
    out = zeros(promote_type(Float64, eltype(u), eltype(p), map(eltype, directions)...), d.n)
    d.g!((out,), (u, p, directions...))
    out
end
function evaluate(d::Derivative, u, p, directions::Tuple)
    k = length(directions)
    parts(mask) = ntuple(i -> (mask >> (i - 1)) & 1 == 1 ? imag.(directions[i]) : real.(directions[i]), k)
    total = sum(mask -> im^count_ones(mask) .* evaluate(d, u, p, parts(mask)), 0:(2^k - 1))
    @ensures length(total) == d.n "Derivative: the result has $(length(total)) elements, not $(d.n)"
    total
end

"""
    Record{N}(getters, p, k)

The `record_from_solution(u, q; iter, k...)` of a bifurcation problem: the named tuple of the
values `getter(u, p)` under the names `N`. BifurcationKit.jl passes the value `q` of the
continued parameter, which becomes element `k` of the parameters of the problem of `iter`, or
of `p` without `iter`; a vector `q` holds all the parameters.
"""
struct Record{N,G}
    getters::G
    p::Vector{Float64}
    k::Int
end
Record{N}(getters, p, k) where {N} = Record{N,typeof(getters)}(getters, p, k)
function (r::Record{N})(u, q; iter = nothing, k...) where {N}
    p = q isa AbstractVector ? q : parameters(r, iter, q)
    NamedTuple{N}(map(g -> g(u, p), r.getters))
end
function parameters(r::Record, iter, q::Number)
    given = iter === nothing ? r.p : BifurcationKit.getparams(BifurcationKit.getprob(iter))
    p = given isa AbstractVector{<:Real} && length(given) == length(r.p) ? collect(promote_type(eltype(given), typeof(q)), given) : collect(promote_type(Float64, typeof(q)), r.p)
    p[r.k] = q
    p
end

"""
    reader(sys, name) -> (u, p) -> value

Return the function of the state, the parameter or the observed variable `name` of the
names `sys` of a `NonlinearFunction`.
"""
function reader(sys, name::Symbol)
    SII.is_variable(sys, name) && return (i = SII.variable_index(sys, name); (u, p) -> u[i])
    SII.is_parameter(sys, name) && return (i = SII.parameter_index(sys, name); (u, p) -> p[i])
    SII.is_observed(sys, name) && return SII.observed(sys, name)
    throw(ArgumentError("BifurcationProblem: $name is neither a state, a parameter nor a further result of the function"))
end

"""
    values_by_name(kind, given, n, index) -> Vector{Float64}

Return the vector of `n` values of `given`: a vector as it is, or pairs of a name and a value,
which `index(name)` places, and which must give a value to each element.
"""
values_by_name(kind, given::AbstractVector{<:Real}, n, index) = (@requires length(given) == n "BifurcationProblem: $kind has $(length(given)) values for $n elements"; collect(float.(given)))
function values_by_name(kind, given, n, index)
    @requires all(pair -> pair isa Pair, given) "BifurcationProblem: $kind must be a vector of values or of pairs `name => value`"
    values = fill(NaN, n)
    for (name, value) in given
        values[index(name)] = value
    end
    @requires !any(isnan, values) "BifurcationProblem: $kind gives no value to the elements $(findall(isnan, values))"
    values
end

"""
    autonomous(f) -> CasADiCodegen.Function

Return `f(u, p)` itself, or for `f(u, p, t)` the function of `u` and `p` alone with the
results of `f`, which must not depend on the time.
"""
function autonomous(f::CasADiCodegen.Function)
    C.n_in(f) == 3 || return f
    sx = C.is_a(f, "SXFunction")
    inputs = [sx ? C.sx_in(f, i) : C.mx_in(f, i) for i in 0:2]
    results = C.n_out(f) == 1 ? [f(inputs...)] : collect(f(inputs...))
    @requires !C.depends_on(results[1], inputs[3]) "BifurcationProblem: the steady states of \"$(C.name(f))\" need a first result that does not depend on the time"
    CasADiCodegen.Function(C.name(f), inputs[1:2], results, [C.name_in(f, i) for i in 0:1], [C.name_out(f, o) for o in 0:C.n_out(f)-1])
end

"""
    BifurcationKit.BifurcationProblem(f::CasADiCodegen.Function, u0, p, parameter;
                                      record_from_solution = nothing, jac = true, jet = true,
                                      chunk_size = nothing, split_above = nothing,
                                      optlevel = nothing, cache = false, options...)

Make the bifurcation problem of the steady states `f(u, p) = 0` of the CasADi function `f`,
continued in the parameter `parameter`. It holds generated code for the residual, its
Jacobian (`jac`) and, with `jet`, the higher derivatives that BifurcationKit.jl uses.

`f` takes the state and the parameters, and its first result is the residual, a column
vector of the size of the state. `f(u, p, t)` with a residual that does not depend on `t`,
such as the right side of an `ODEFunction`, also serves. The states and the parameters have
the names of the scalar symbols of `f`, as for `SciMLBase.NonlinearFunction`. Further results
of `f` are observed variables, each known by the name of its result.

`u0`, the guess of a steady state, and `p`, the values of the parameters, are vectors in the
order of the symbols of `f`, or vectors of pairs `name => value`, which must give a value to
each element. `parameter` is the name of a scalar parameter, such as `:μ`, or its place in
`p`, from 1.

`record_from_solution` gives what the branch records at each point: a name, or a vector or
tuple of names, of states, parameters and observed variables, recorded as a named tuple;
a function `(u, p; k...) -> value`; or `nothing` for the default of BifurcationKit.jl, the
norm of the state.

`jac = false` leaves the Jacobian to automatic differentiation by BifurcationKit.jl. `jet`
adds the Jacobian-vector product (`jvp`), the derivatives of the second and the third order
by the state in given directions (`d2F`, `d3F`), and the derivatives by the continued
parameter (`R01`, `R11`), which the normal forms at branch points and Hopf points need;
with `jet = false`, BifurcationKit.jl computes them by automatic differentiation of the
generated residual, and `R11` by finite differences.
`chunk_size`, `split_above`, `optlevel` and `cache` go to
[`build_function`](@ref CasADiCodegen.build_function), and the other `options` to
`BifurcationKit.BifurcationProblem`.

# Examples
```julia
x, μ = SX.sym("x"), SX.sym("μ")
f = CasadiFunction("f", [x, μ], [μ * x - x^3])
problem = BifurcationKit.BifurcationProblem(f, [:x => 0.0], [:μ => -1.0], :μ; record_from_solution = :x)
```
"""
function BifurcationKit.BifurcationProblem(f::CasADiCodegen.Function, u0, p, parameter::Union{Symbol,Integer};
                                           record_from_solution = nothing, jac::Bool = true, jet::Bool = true,
                                           chunk_size = nothing, split_above = nothing, optlevel = nothing,
                                           cache = false, options...)
    S = sciml()
    @requires S !== nothing "BifurcationProblem: the extension for SciMLBase.jl is not loaded"
    @requires 2 <= C.n_in(f) <= 3 "BifurcationProblem: the function \"$(C.name(f))\" must have the arguments (u, p) or (u, p, t)"
    g = autonomous(f)
    codegen = S.codegen_options(; chunk_size, split_above, optlevel, cache)
    arguments, value, _ = S.vector_field("BifurcationProblem", g, 2)
    n, m = C.numel(arguments[1]), C.numel(arguments[2])
    nf = SciMLBase.NonlinearFunction(g; jac, codegen...)
    sys = nf.sys
    named(name) = (@requires sys !== nothing && !isempty(SII.all_symbols(sys)) "BifurcationProblem: the name $name needs distinct names for the symbols of the arguments of \"$(C.name(f))\""; true)
    state_index(name) = (named(name); @requires SII.is_variable(sys, name) "BifurcationProblem: $name is not a state"; SII.variable_index(sys, name))
    parameter_index(name) = (named(name); @requires SII.is_parameter(sys, name) "BifurcationProblem: $name is not a parameter"; SII.parameter_index(sys, name))
    x0 = values_by_name("u0", u0, n, i -> state_index(i))
    p0 = values_by_name("p", p, m, i -> parameter_index(i))
    k = parameter isa Integer ? Int(parameter) : parameter_index(parameter)
    @requires k isa Int && 1 <= k <= m "BifurcationProblem: the continued parameter must be one scalar parameter, not $parameter"
    record = record_from_solution === nothing ? (;) :
             record_from_solution isa Union{Symbol,AbstractVector{Symbol},Tuple{Vararg{Symbol}}} ?
                 (; record_from_solution = recorded(sys, Tuple(record_from_solution isa Symbol ? (record_from_solution,) : record_from_solution), named, p0, k)) :
             (; record_from_solution)
    derivatives = jet ? jet_functions(S, arguments, value, k, codegen) : (;)
    jacobian = jac ? (; J = nf.jac, J! = nf.jac) : (;)
    problem = BifurcationKit.BifurcationProblem(Residual(nf.f, n), x0, p0, k; inplace = true, jacobian..., derivatives..., record..., options...)
    @ensures BifurcationKit.getparam(problem) == p0[k] "BifurcationProblem: the continued parameter is not element $k of the parameters"
    problem
end

"The `Record` of the `names` of `sys`, after `named(name)` has checked that `sys` has names."
recorded(sys, names::Tuple, named, p, k) = Record{names}(map(name -> (named(name); reader(sys, name)), names), p, k)

"""
    jet_functions(S, arguments, value, k, codegen) -> NamedTuple

Return the generated derivatives of the residual `value` of the symbols `arguments` (state
and parameters) as keywords of `BifurcationKit.BifurcationProblem`: `jvp`, `d2F`, `d3F`, and
`R01`, `R11` by the parameter `k`.
"""
function jet_functions(S, arguments, value, k::Int, codegen)
    x, p = arguments
    n = C.numel(x)
    T = typeof(x)
    dx = [T.sym("d$i", n) for i in 1:3]
    first = C.jtimes(value, x, dx[1])
    second = C.jtimes(first, x, dx[2])
    third = C.jtimes(second, x, dx[3])
    by_parameter = CasADiCodegen.densify(CasADiCodegen.jacobian(value, p)[:, k])
    mixed = C.jtimes(by_parameter, x, dx[1])
    generated(name, directions, expression) =
        Derivative(S.inplace("bifurcation_" * name, [x, p, dx[1:directions]...], CasADiCodegen.densify(expression); codegen...), n)
    d2F, d3F = generated("d2F", 2, second), generated("d3F", 3, third)
    (; jvp = generated("jvp", 1, first), d2F, d3F,
       R01 = generated("R01", 0, by_parameter), R11 = generated("R11", 1, mixed))
end

end # module CasADiCodegenBifurcationKitExt
