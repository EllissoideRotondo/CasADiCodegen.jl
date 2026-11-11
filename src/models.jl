"""
    CasADiCodegen.Models

Differential equations written as equations of SX expressions: `Differential(t)` gives the
time derivative of a symbol or an expression, `lhs ~ rhs` an equation, and `System`
compiles a list of equations into the semi-explicit form of the CasADi integrators. Load it
with `using CasADiCodegen.Models`.
"""
module Models

using ..CasADiCodegen: CasADiCodegen, SX, casadi, jacobian
using ..Codegen: @requires, @ensures, reduce_index

export Differential, Equation, System, unknowns, algebraic, parameters, observed, equations, invariants,
    differentiation_index

"""
    SymbolMap{V}()

A dictionary from scalar SX symbols to values of type `V`, by the hash of the node of the
symbol, so that a lookup does not compare the symbol with every key.
"""
struct SymbolMap{V}
    table::Dict{Int,Vector{Pair{SX,V}}}
end
SymbolMap{V}() where {V} = SymbolMap{V}(Dict{Int,Vector{Pair{SX,V}}}())

function Base.get(m::SymbolMap, s::SX, default)
    for (key, value) in get(m.table, casadi.element_hash(s), ())
        casadi.is_equal(key, s) && return value
    end
    default
end
Base.haskey(m::SymbolMap, s::SX) = any(e -> casadi.is_equal(first(e), s), get(m.table, casadi.element_hash(s), ()))
function Base.setindex!(m::SymbolMap{V}, value, s::SX) where {V}
    entries = get!(Vector{Pair{SX,V}}, m.table, casadi.element_hash(s))
    k = findfirst(e -> casadi.is_equal(first(e), s), entries)
    k === nothing ? push!(entries, s => value) : (entries[k] = s => value)
    value
end
function Base.get!(f::Base.Callable, m::SymbolMap, s::SX)
    haskey(m, s) && return get(m, s, nothing)
    m[s] = f()
end

# The derivative symbol of each symbol, and the symbol and the order of each derivative
# symbol. The tables keep the symbols alive, so that the hash of a node is never that of another.
const TABLES_LOCK = ReentrantLock()
const DERIVATIVES = SymbolMap{SX}()
const ORIGINS = SymbolMap{Tuple{SX,Int}}()

"Return the symbol and the order of differentiation of the scalar symbol `s`: `(s, 0)` for a symbol that is no derivative."
origin(s::SX) = @lock TABLES_LOCK get(ORIGINS, s, (s, 0))

"Return the derivative symbol of the scalar symbol `s`, named `\"d\"` followed by the name of `s`, made at the first call."
function derivative_symbol(s::SX)
    @lock TABLES_LOCK begin
        found = get(DERIVATIVES, s, nothing)
        found === nothing || return found
        base, order = get(ORIGINS, s, (s, 0))
        ds = SX.sym("d" * casadi.name(s))
        DERIVATIVES[s] = ds
        ORIGINS[ds] = (base, order + 1)
        ds
    end
end

"The scalar elements of `e` in the order of the elements, by columns."
elements(e::SX) = SX[e[i] for i in 1:casadi.numel(e)]

"The SX column of the scalars `v`, with no element when `v` is empty."
column(v::AbstractVector{SX}) = isempty(v) ? SX(0, 1) : vcat(v...)

symbols(e::SX) = SX[s for s in casadi.symvar(e)]

"""
    Differential(t::SX)

The time derivative by the scalar symbol `t`. `D = Differential(t)` applies to an SX
expression: `D(x)` of a symbol `x` is the derivative symbol of `x`, named `"dx"`, the same
symbol at each call, and `D(D(x))` is `"ddx"`. `D(e)` of an expression is the chain rule,
`D(t)` is 1 and `D` of a number is 0. A matrix gives the matrix of the derivatives of its
elements. [`System`](@ref) sets the derivative of a parameter to 0.

# Examples
```julia
t, x = SX.sym("t"), SX.sym("x")
D = Differential(t)
D(x^2 + t)    # 2 x dx + 1
```
"""
struct Differential
    t::SX
    function Differential(t::SX)
        @requires size(t) == (1, 1) && casadi.is_symbolic(t) "Differential: the time must be a scalar symbol, not $t"
        new(t)
    end
end

function (D::Differential)(e::SX)
    if casadi.is_valid_input(e)
        result = [casadi.is_equal(s, D.t) ? SX(1) : derivative_symbol(s) for s in elements(e)]
        return reshape(column(result), size(e)...)
    end
    s = filter(v -> !casadi.is_equal(v, D.t), symbols(e))
    reshape(casadi.jtimes(casadi.vec(e), column(SX[s; D.t]), column(SX[derivative_symbol.(s); SX(1)])), size(e)...)
end
(D::Differential)(::Real) = SX(0)

"""
    Equation(lhs::SX, rhs::SX)
    lhs ~ rhs

The equation `lhs = rhs` of a [`System`](@ref). `lhs` and `rhs` have one size, or one of
them is a scalar, which stands for each element of the other; a number is a scalar.
A matrix gives one scalar equation per element. Broadcasting over Julia vectors of
symbols, as in `D.(xs) .~ vs`, gives a vector of equations.
"""
struct Equation
    lhs::SX
    rhs::SX
    function Equation(lhs::SX, rhs::SX)
        @requires size(lhs) == size(rhs) || size(lhs) == (1, 1) || size(rhs) == (1, 1) "Equation: the sides have the sizes $(size(lhs)) and $(size(rhs))"
        new(lhs, rhs)
    end
end
Base.:~(lhs::SX, rhs::SX) = Equation(lhs, rhs)
Base.:~(lhs::SX, rhs::Real) = Equation(lhs, SX(rhs))
Base.:~(lhs::Real, rhs::SX) = Equation(SX(lhs), rhs)
Base.show(io::IO, e::Equation) = print(io, e.lhs, " ~ ", e.rhs)

"The scalar residuals `lhs - rhs` of `e`."
residuals(e::Equation) = elements(e.lhs - e.rhs)

"""
    System(equations, t; parameters = SX[], name = "system", baumgarte_pole = 0.0)

Compile the [`Equation`](@ref)s `equations` in the time `t` into a model in semi-explicit
form, `x' = ode(x, z, p, t)` and `0 = alg(x, z, p, t)`. The symbols of the equations other
than `t` and `parameters` are the variables of the model. `parameters` is a vector of SX
symbols, scalar or vector. The model must have one scalar equation per variable, where a
variable counts once with all its derivatives.

The compilation takes four steps:

1. An algebraic equation (one without derivatives) with a variable `y` that is not
   differentiated and that it contains linearly, with a coefficient that is a number or
   an expression of the parameters, is solved for `y`, and `y` is replaced by its solution
   in the other equations. `y` becomes an observed variable; a coefficient of parameters
   that is zero gives it the value `NaN` or `Inf`. This removes the equations of the form `a ~ b`, `a ~ -b` and
   `a ~ f(b, c)`, which connections of parts make.
2. A variable `x` whose derivatives appear up to the order `n` becomes the `n`
   differential states `x`, `dx`, ..., the derivative symbols of `D = Differential(t)`.
3. When the equations that contain the derivatives of the highest order are as many as
   these and linear in them, with a nonsingular coefficient, they are solved for the
   derivatives, which become observed variables. Otherwise each such derivative becomes
   an algebraic state, as in `exp(D(x)) ~ x`.
4. When the model has algebraic states, [`reduce_index`](@ref CasADiCodegen.reduce_index)
   reduces it to index 1, with the pole `baumgarte_pole`. The constraints that the
   reduction differentiates become [`invariants`](@ref), which the initial values must
   satisfy.

[`unknowns`](@ref) gives the states of the result, [`observed`](@ref) the eliminated
variables, `Dict(sys)` the model for `casadi.integrator`, and
`SciMLBase.ODEProblem(sys, u0, tspan, p)` the SciML problem. Only SX is supported.

Throw an `ArgumentError` when the number of scalar equations differs from that of the
variables, when the elimination leaves an equation without variables, when two symbols of
the model have one name, and when `reduce_index` refuses the model.

# Examples
```julia
using CasADiCodegen, CasADiCodegen.Models
t, x, k = SX.sym("t"), SX.sym("x"), SX.sym("k")
D = Differential(t)
sys = System([D(D(x)) ~ -k * x], t; parameters = [k])
unknowns(sys)    # [x, dx]
```
"""
struct System
    name::String
    t::SX
    x::Vector{SX}
    z::Vector{SX}
    p::Vector{SX}
    ode::Vector{SX}
    alg::Vector{SX}
    observed::Vector{Pair{SX,SX}}
    invariants::SX
    index::Int
end

function System(equations::AbstractVector, t::SX; parameters::AbstractVector = SX[], name::AbstractString = "system",
                baumgarte_pole::Real = 0.0)
    @requires all(e -> e isa Equation, equations) "System: each equation must be an Equation, made with ~"
    @requires all(p -> p isa SX && casadi.is_valid_input(p), parameters) "System: each parameter must be an SX symbol"
    time = Differential(t).t
    p = reduce(vcat, (elements(q) for q in parameters); init = SX[])
    @requires !any(q -> casadi.is_equal(q, time), p) "System: the time is not a parameter"
    numbers = Int[]
    r = SX[]
    for (i, e) in enumerate(equations)
        scalars = residuals(e)
        append!(r, scalars)
        append!(numbers, fill(i, length(scalars)))
    end
    isparameter = SymbolMap{Bool}()
    foreach(q -> isparameter[q] = true, p)

    # The derivatives of the parameters are zero; the bases of the other symbols are the unknowns.
    bases, orders = SX[], Int[]
    place = SymbolMap{Int}()
    zeroed = SX[]
    for s in symbols(column(r))
        casadi.is_equal(s, time) && continue
        base, order = origin(s)
        if haskey(isparameter, base)
            order > 0 && push!(zeroed, s)
            continue
        end
        k = get(place, base, 0)
        k == 0 ? (push!(bases, base); push!(orders, order); place[base] = length(bases)) : (orders[k] = max(orders[k], order))
    end
    if !isempty(zeroed)
        r = elements(casadi.substitute(column(r), column(zeroed), SX.zeros(length(zeroed))))
    end
    unknown_count = length(bases)
    @requires length(r) == unknown_count "System: the model has $(length(r)) equation$(length(r) == 1 ? "" : "s") and $unknown_count variable$(unknown_count == 1 ? "" : "s"); it needs one equation per variable, where a variable and its derivatives count once"
    check_names(String[casadi.name.(bases); casadi.name.(p)])

    candidates = SymbolMap{Bool}()
    foreach(b -> candidates[b] = true, bases[orders .== 0])
    r, numbers, eliminated = eliminate(r, numbers, candidates, isparameter)

    # The chains of differential states, and the derivatives of the highest order.
    x, ode, tops = SX[], SX[], SX[]
    for (b, n) in zip(bases, orders)
        n == 0 && continue
        chain = [b]
        for _ in 1:n
            push!(chain, derivative_symbol(chain[end]))
        end
        append!(x, chain[1:n])
        append!(ode, chain[2:n+1])
        push!(tops, chain[end])
    end
    gone = SymbolMap{Bool}()
    foreach(e -> gone[first(e)] = true, eliminated)
    z = SX[b for (b, n) in zip(bases, orders) if n == 0 && !haskey(gone, b)]

    observed = copy(eliminated)
    W = column(tops)
    differential = [i for i in eachindex(r) if !isempty(tops) && casadi.depends_on(r[i], W)]
    alg = SX[]
    explicit = length(differential) == length(tops) && !isempty(tops)
    if explicit
        Ed = column(r[differential])
        A = jacobian(Ed, W)
        explicit = !casadi.depends_on(A, W) && casadi.sprank(casadi.sparsity(A)) == length(tops)
    end
    if explicit
        b = casadi.substitute(Ed, W, SX.zeros(length(tops)))
        solved = elements(casadi.solve(A, -b))
        ode = elements(casadi.substitute(column(ode), W, column(solved)))
        append!(observed, [w => s for (w, s) in zip(tops, solved)])
        alg = r[setdiff(eachindex(r), differential)]
    else
        z = SX[tops; z]
        alg = r
    end

    invariants, index = SX(0, 1), 0
    if !isempty(z)
        dae = Dict("x" => column(x), "z" => column(z), "t" => time, "ode" => column(ode), "alg" => column(alg))
        isempty(p) || (dae["p"] = column(p))
        reduced = reduce_index(dae; baumgarte_pole)
        x, z = elements(reduced.dae["x"]), elements(reduced.dae["z"])
        ode, alg = elements(reduced.dae["ode"]), elements(reduced.dae["alg"])
        invariants, index = reduced.invariants, max(reduced.index, 1)
    end
    check_names(String[casadi.name.(x); casadi.name.(z); casadi.name.(p); casadi.name.(first.(observed))])
    @ensures length(ode) == length(x) && length(alg) == length(z) "System: the compiled model is not square"
    System(String(name), time, x, z, p, ode, alg, observed, invariants, index)
end

function check_names(names::Vector{String})
    seen = Set{String}()
    for n in names
        @requires n ∉ seen "System: two symbols of the model are named \"$n\"; give each symbol a name of its own"
        push!(seen, n)
    end
end

"""
    eliminate(r, numbers, candidates, isparameter) -> (r, numbers, eliminated)

Solve each algebraic residual of `r` that contains one of the unknowns `candidates`
linearly for that unknown, and substitute the solution into the other residuals, until no
residual can be solved. The coefficient of the unknown must be a nonzero number, or, when
no residual has one, an expression of the parameters, the keys of `isparameter`. Return
the remaining residuals, their equation numbers and the pairs `unknown => solution`, in
which each solution is an expression of the remaining unknowns.
"""
function eliminate(r::Vector{SX}, numbers::Vector{Int}, candidates::SymbolMap{Bool}, isparameter::SymbolMap{Bool})
    n = length(r)
    present = [symbols(e) for e in r]
    holders = SymbolMap{Vector{Int}}()     # the residuals that contain each symbol
    for i in 1:n, s in present[i]
        push!(get!(Vector{Int}, holders, s), i)
    end
    alive, queued = trues(n), trues(n)
    queue, deferred = collect(1:n), Int[]
    order = Pair{SX,SX}[]
    while true
        found = nothing
        while found === nothing && !isempty(queue)
            i = popfirst!(queue)
            queued[i] = false
            alive[i] || continue
            check_constant(r[i], numbers[i])
            kind, y, a = solvable(r[i], present[i], candidates, isparameter)
            kind === :number && (found = (i, y, a))
            kind === :parameters && push!(deferred, i)
        end
        while found === nothing && !isempty(deferred)
            i = popfirst!(deferred)
            alive[i] || continue
            kind, y, a = solvable(r[i], present[i], candidates, isparameter)
            kind === :none || (found = (i, y, a))
        end
        found === nothing && break
        i, y, a = found
        alive[i] = false
        push!(order, y => -casadi.substitute(r[i], y, SX(0)) / a)
        affected = unique!([k for k in get(holders, y, Int[]) if alive[k]])
        isempty(affected) && continue
        for (k, e) in zip(affected, elements(casadi.substitute(column(r[affected]), y, last(order[end]))))
            r[k], present[k] = e, symbols(e)
            foreach(s -> push!(get!(Vector{Int}, holders, s), k), present[k])
            queued[k] || (push!(queue, k); queued[k] = true)
        end
    end
    foreach(i -> check_constant(r[i], numbers[i]), findall(alive))
    (r[alive], numbers[alive], resolve(order))
end

"""
    resolve(order) -> Vector{Pair{SX,SX}}

Return the solutions of `order` in the remaining unknowns. Solution `k` of `order` contains
only the unknowns of solutions after it, since each solution was substituted into the
residuals that remained, so the solutions are resolved from the last one back.
"""
function resolve(order::Vector{Pair{SX,SX}})
    position = SymbolMap{Int}()
    foreach(k -> position[first(order[k])] = k, eachindex(order))
    resolved = Vector{SX}(undef, length(order))
    for k in reverse(eachindex(order))
        g = last(order[k])
        later = [s for s in symbols(g) if get(position, s, 0) > k]
        resolved[k] = isempty(later) ? g : casadi.substitute(g, column(later), column([resolved[get(position, s, 0)] for s in later]))
        @ensures all(s -> get(position, s, 0) == 0, symbols(resolved[k])) "System: an eliminated unknown remains in the solution of another"
    end
    [first(e) => v for (e, v) in zip(order, resolved)]
end

"""
    solvable(residual, present, candidates, isparameter) -> (kind, y, a)

Return an unknown `y` of `candidates` among the symbols `present` of `residual`, which has
no derivative, such that `residual` is linear in `y` with the coefficient `a`. `kind` is
`:number` for a nonzero number `a`, found first, `:parameters` for an expression of the
parameters, and `:none` when no unknown qualifies.
"""
function solvable(residual::SX, present::Vector{SX}, candidates::SymbolMap{Bool}, isparameter::SymbolMap{Bool})
    any(s -> origin(s)[2] > 0, present) && return (:none, residual, residual)
    fallback = (:none, residual, residual)
    for y in present
        haskey(candidates, y) || continue
        a = jacobian(residual, y)
        if casadi.is_constant(a)
            Float64(casadi.DM(a)) != 0 && return (:number, y, a)
        elseif fallback[1] === :none && all(s -> haskey(isparameter, s), symbols(a))
            fallback = (:parameters, y, a)
        end
    end
    fallback
end

function check_constant(residual::SX, number::Int)
    casadi.is_constant(residual) || return
    value = Float64(casadi.DM(residual))
    @requires value != 0 "System: equation $number reduces to the contradiction $value = 0 after the elimination of the other equations"
    @requires false "System: equation $number holds identically after the elimination of the other equations, so the model lacks an equation"
end

"""
    unknowns(sys::System) -> Vector{SX}

Return the states of `sys`: the differential states, then the algebraic states.
"""
unknowns(sys::System) = SX[sys.x; sys.z]

"""
    algebraic(sys::System) -> Vector{SX}

Return the algebraic states of `sys`.
"""
algebraic(sys::System) = copy(sys.z)

"""
    parameters(sys::System) -> Vector{SX}

Return the scalar parameters of `sys`, in the order of the keyword `parameters` of [`System`](@ref).
"""
parameters(sys::System) = copy(sys.p)

"""
    observed(sys::System) -> Vector{Pair{SX,SX}}

Return the observed variables of `sys`, each a symbol and its expression in the states,
the parameters and the time: the eliminated variables, and the solved derivatives of the
highest order.
"""
observed(sys::System) = copy(sys.observed)

"""
    equations(sys::System) -> Vector{Equation}

Return the equations of the compiled model: `D(x) ~ ode` for each differential state `x`,
then `0 ~ alg` for the algebraic equations.
"""
function equations(sys::System)
    D = Differential(sys.t)
    [[D(x) ~ f for (x, f) in zip(sys.x, sys.ode)]; [SX(0) ~ g for g in sys.alg]]
end

"""
    invariants(sys::System) -> SX

Return the column of the constraints that the index reduction of [`System`](@ref)
differentiates, and their derivatives below the highest, which are zero along a
solution; it has no element when the index was not reduced.
"""
invariants(sys::System) = sys.invariants

"""
    differentiation_index(sys::System) -> Int

Return the differentiation index of the model of `sys` before the index reduction: 0 for
an ordinary differential equation, 1 for an algebraic equation that determines its
algebraic states, 2 or more for one that must be differentiated.
"""
differentiation_index(sys::System) = sys.index

"""
    Dict(sys::System) -> Dict{String,SX}

Return the model of `sys` in the form of the CasADi integrators: `"x"`, `"z"`, `"p"`,
`"t"`, `"ode"` and `"alg"`, without `"z"` and `"alg"` for a model without algebraic
states and without `"p"` for one without parameters.
"""
function Base.Dict(sys::System)
    dae = Dict{String,SX}("x" => column(sys.x), "t" => sys.t, "ode" => column(sys.ode))
    isempty(sys.z) || (dae["z"] = column(sys.z); dae["alg"] = column(sys.alg))
    isempty(sys.p) || (dae["p"] = column(sys.p))
    dae
end

function Base.show(io::IO, ::MIME"text/plain", sys::System)
    count(n, word) = string(n, " ", word, n == 1 ? "" : "s")
    print(io, "System \"", sys.name, "\" in ", sys.t, " with ", count(length(sys.x), "differential state"), ", ",
          count(length(sys.z), "algebraic state"), ", ", count(length(sys.p), "parameter"), " and ",
          count(length(sys.observed), "observed variable"))
end

end # module Models
