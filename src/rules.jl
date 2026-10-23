# CasADi functions inside Julia code that an automatic differentiation
# differentiates. The function keeps generated code for its products with the
# Jacobian, and a package extension gives them to ChainRulesCore.jl as rules.

"""
    Differentiable

A CasADi function as a Julia function with rules for ChainRulesCore.jl; `differentiable`
makes one. A call `d(arguments...)` takes one array per input and returns one array per
output, as the out-of-place function of `build_function`.
"""
struct Differentiable{G,O}
    f::C.Function
    value::G
    options::O
    derivatives::Dict{Symbol,Any}
    lock::ReentrantLock
end
(d::Differentiable)(arguments...) = d.value(arguments...)
Base.show(io::IO, d::Differentiable) = print(io, "the differentiable function \"", C.name(d.f), "\"")

"""
    differentiable(f::CasADiCodegen.Function; options...) -> Differentiable

Make the CasADi function `f` a Julia function with forward and reverse rules for
ChainRulesCore.jl. A package for automatic differentiation that reads those rules, such as
Zygote.jl, then takes the derivatives of `f` from code that CasADi generates.

The result `d` takes one array per input of `f` and returns one array per output, as the
out-of-place function of [`build_function`](@ref). The `options` go to `build_function`.

The rules load with ChainRulesCore.jl. The code of each derivative is generated at its
first use. See [Use the generated derivatives from an AD package](@ref).

# Examples
```jldoctest
julia> using CasADiCodegen

julia> x = SX.sym("x", 2);

julia> f = CasadiFunction("f", [x], [x[1] * sin(x[2])]);

julia> d = CasADiCodegen.differentiable(f)
the differentiable function "f"

julia> d([2.0, pi / 2])
1-element Vector{Float64}:
 2.0
```
"""
differentiable(f::C.Function; options...) =
    Differentiable(f, build_function(f; expression = Val{false}, options...)[1], values(options), Dict{Symbol,Any}(), ReentrantLock())

"""
    directional(d, kind) -> Julia function

Return the generated out-of-place function of a product with the Jacobian of `d`.

- `kind = :pushforward`: the function takes the arguments and one change per argument. It
  returns the results followed by their changes.
- `kind = :pullback`: the function takes the arguments and one seed per result. It returns
  one sum per argument: the products of the transposed Jacobians with the seeds.
"""
function directional(d::Differentiable, kind::Symbol)
    @requires kind in (:pushforward, :pullback) "the kind of a derivative must be :pushforward or :pullback; got $kind"
    lock(d.lock) do
        get!(d.derivatives, kind) do
            f = d.f
            x = [MX.sym(C.name_in(f, i), C.sparsity_in(f, i)) for i in 0:C.n_in(f)-1]
            y = as_list(f(x...))
            g = if kind === :pushforward
                changes = [MX.sym("d_" * C.name_in(f, i), C.sparsity_in(f, i)) for i in 0:C.n_in(f)-1]
                Front.Function(C.name(f) * "_pushforward", MX[x; changes],
                               MX[y; [reduce(+, [Front.jtimes(r, a, da, false) for (a, da) in zip(x, changes)]) for r in y]])
            else
                seeds = [MX.sym("adj_" * C.name_out(f, o), C.sparsity_out(f, o)) for o in 0:C.n_out(f)-1]
                Front.Function(C.name(f) * "_pullback", MX[x; seeds],
                               [reduce(+, [Front.jtimes(r, a, seed, true) for (r, seed) in zip(y, seeds)]) for a in x])
            end
            build_function(g; expression = Val{false}, d.options...)[1]
        end
    end
end
