# Structural index reduction of a semi-explicit DAE `x' = f(x, z, p, t)`, `0 = g(x, z, p, t)`.
#
# Pantelides' algorithm (1988) runs on the bipartite graph of the algebraic equations and
# the algebraic states. The differential equations are not nodes of the graph: the
# derivative of `x` is `f` itself, so a total time derivative substitutes `f` for `x'`,
# and an equation that depends on `x` gains the algebraic states of `f` when it is
# differentiated. An algebraic state that the algorithm differentiates becomes a
# differential state, and its derivative a new algebraic state, as in `dae_reduce_index`
# of CasADi. The algorithm decides the structure first; CasADi then computes the
# derivatives of the equations by `jtimes`, one call for all equations of one order.

"""
    reduce_index(dae; baumgarte_pole = 0.0) -> NamedTuple

Reduce the semi-explicit differential-algebraic equation `dae` to index 1 by
differentiating its algebraic equations, with the structural algorithm of Pantelides.
Return `(; dae, invariants, differentiations, index)`.

`dae` is a dictionary in the form of the CasADi integrators, with SX or MX values that
are dense column vectors: the differential states `"x"`, the algebraic states `"z"`,
the parameters `"p"`, the scalar time `"t"`, and the expressions `"ode"`, the
derivative of `x`, and `"alg"`, the algebraic equations `0 = alg`, of the size of `z`.
`"x"`, `"z"` and `"t"` are symbols or concatenations of symbols. `"p"` and `"t"` are
optional, and so are `"z"` and `"alg"` together.

The result `dae` has the same form, and `CasADiCodegen.C.integrator` with a DAE plugin
such as `"idas"` accepts it. Equation `alg[i]` is replaced by its time derivative of
order `differentiations[i]`, in which the derivative of `x` is replaced by `ode`. An
algebraic state `z[j]` whose derivative enters these equations becomes a differential
state, and its derivative a new symbol that takes its place in `z`. The states of the
result are `x`, then for each such `z[j]`, in the order of `z`, the state and its
derivatives below the highest. A derivative is named `"d"` followed by the name of the
symbol. For `T` differentiated twice, for example, the result has the states
`[x; T; dT]` with the derivative `[ode; dT; ddT]`, and `ddT` in `z`. The element `k` of
an MX vector symbol `z` is first replaced by the scalar symbol `z_k`, counted from 0, as
the elements of an SX vector symbol are named, so a function of the result takes
`dae["x"]` and `dae["z"]` of the result, not the original symbols.

`invariants` is the column of the replaced equations and of their derivatives of lower
order, by equation and then by order, in the symbols of the result. They are zero along
a solution that starts where they are zero, but the reduced equations do not enforce
them, so a numerical solution drifts away from them. A consistent initial point
satisfies `invariants` and `alg`. `index` is the differentiation index: 0 with no
algebraic equation, otherwise 1 plus the largest number of differentiations.

A model of index 1 is returned unchanged, with `differentiations` all 0 and no
invariant. Write a fully implicit equation `0 = F(x', x, z)` in this form with new
algebraic states `w` for `x'`, `ode = w` and `alg = F(w, x, z)`.

# Keywords
- `baumgarte_pole = 0.0`: the pole `γ ≤ 0` of Baumgarte's stabilization. An equation `g`
  differentiated `k` times becomes the sum over `j` in `0:k` of `binomial(k, j) (-γ)^j`
  times the derivative of `g` of order `k - j`, whose characteristic polynomial
  `(s - γ)^k` has the root `γ`, so that its invariants decay as `exp(γ t)` instead of
  drifting. 0 gives the derivative of order `k` alone, as
  `dae_reduce_index` of CasADi does by default. A positive `γ` throws `ArgumentError`.

Throws `ArgumentError` when the equations are structurally singular, that is, when no
number of differentiations determines the states; the message names the equations
`ode[i]` and `alg[i]` that are involved, with `i` from 1. The algorithm uses the
sparsity patterns only, and terms of a derivative can cancel. When the reduced equations
then do not determine `z`, the function throws `ArgumentError` too; it never returns
such a model.

The algorithm does not select dummy derivatives (Mattsson and Söderlind): the reduced
model keeps every state and enforces only the highest derivative of each equation, so
the invariants drift unless they are stabilized. See [Model physical systems with differential-algebraic
equations](@ref).

# Examples
```jldoctest
julia> using CasADiCodegen

julia> q, v, T = SX.sym("q", 2), SX.sym("v", 2), SX.sym("T");

julia> pendulum = Dict("x" => vertcat(q, v), "z" => T,
                       "ode" => vertcat(v, T * q - vertcat(0, 9.81)), "alg" => q[1]^2 + q[2]^2 - 1);

julia> reduced = CasADiCodegen.reduce_index(pendulum);

julia> reduced.index, reduced.differentiations
(3, [2])

julia> reduced.invariants
2×1 SX:
[((sq(q_0)+sq(q_1))-1), (((2.*q_0)*v_0)+((2.*q_1)*v_1))]
```
"""
function reduce_index(dae::AbstractDict{<:AbstractString}; baumgarte_pole::Real = 0.0)
    known = ("x", "z", "p", "t", "ode", "alg")
    @requires all(in(known), keys(dae)) "reduce_index: the keys of the DAE must be among $(join(known, ", ")); found $(join(sort!(collect(keys(dae))), ", "))"
    @requires haskey(dae, "x") && haskey(dae, "ode") "reduce_index: the DAE must have the states \"x\" and their derivative \"ode\""
    @requires haskey(dae, "z") == haskey(dae, "alg") "reduce_index: the DAE must have both \"z\" and \"alg\", or neither"
    X = typeof(dae["x"])
    @requires X in (SX, MX) && all(v -> v isa X, values(dae)) "reduce_index: the values of the DAE must all be SX or all MX"
    @requires all(v -> C.is_column(v) && C.is_dense(v), values(dae)) "reduce_index: each value of the DAE must be a dense column vector"
    @requires isfinite(baumgarte_pole) && baumgarte_pole <= 0 "reduce_index: baumgarte_pole must be finite and not positive, not $baumgarte_pole"
    x, f = dae["x"], dae["ode"]
    z, g = get(dae, "z", X(0, 1)), get(dae, "alg", X(0, 1))
    @requires C.is_valid_input(x) && C.is_valid_input(z) "reduce_index: \"x\" and \"z\" must be symbols"
    @requires size(f, 1) == size(x, 1) "reduce_index: \"ode\" has $(size(f, 1)) elements, \"x\" has $(size(x, 1))"
    @requires size(g, 1) == size(z, 1) "reduce_index: \"alg\" has $(size(g, 1)) elements, \"z\" has $(size(z, 1))"
    @requires !haskey(dae, "t") || (size(dae["t"]) == (1, 1) && C.is_symbolic(dae["t"])) "reduce_index: \"t\" must be a scalar symbol"
    nz = size(z, 1)
    nz == 0 && return (; dae = Dict{String,X}(dae), invariants = X(0, 1), differentiations = Int[], index = 0)

    rows(e, v) = dependencies(C.jacobian_sparsity(e, v))
    fx, fz, gx, gz = rows(f, x), rows(f, z), rows(g, x), rows(g, z)
    check_structure(fx, fz, gx, gz)
    graph = pantelides(fx, fz, gx, gz)
    differentiations = graph.eq_order[highest(graph.eq_der, 1:nz)]
    all(iszero, differentiations) && return (; dae = Dict{String,X}(dae), invariants = X(0, 1), differentiations, index = 1)
    reduced = reduced_dae(dae, graph, X, Float64(baumgarte_pole))
    (; reduced..., differentiations, index = 1 + maximum(differentiations))
end

"The column indices of each row of `pattern`, sorted, from 1."
function dependencies(pattern)
    deps = [Int[] for _ in 1:C.size1(pattern)]
    r, c = C.get_triplet(pattern)
    for (i, j) in zip(r, c)
        push!(deps[i+1], j + 1)
    end
    foreach(sort!, deps)
    deps
end

"The last element of each chain `i, der[i], der[der[i]], ...` that starts at `starts`."
function highest(der, starts)
    map(starts) do i
        while der[i] != 0
            i = der[i]
        end
        i
    end
end

"""
    maximum_matching(adjacency, ncols) -> Vector{Int}

Return the column matched to each row of the bipartite graph whose row `i` connects to the
columns `adjacency[i]`, 0 for an unmatched row, by augmenting paths (Kuhn's algorithm).
"""
function maximum_matching(adjacency, ncols)
    row_of = zeros(Int, ncols)
    col_of = zeros(Int, length(adjacency))
    function augment(i, seen)
        for j in adjacency[i]
            seen[j] && continue
            seen[j] = true
            if row_of[j] == 0 || augment(row_of[j], seen)
                row_of[j], col_of[i] = i, j
                return true
            end
        end
        false
    end
    for i in eachindex(adjacency)
        augment(i, falses(ncols))
    end
    @ensures all(col_of[i] == 0 || row_of[col_of[i]] == i for i in eachindex(adjacency)) "maximum_matching: the matching is not consistent"
    col_of
end

"""
    overdetermined(adjacency, col_of) -> Vector{Int}

Return the rows that an alternating path reaches from an unmatched row of the matching
`col_of`: the rows of the structurally overdetermined part (Dulmage and Mendelsohn).
"""
function overdetermined(adjacency, col_of)
    row_of = Dict(j => i for (i, j) in enumerate(col_of) if j != 0)
    reached = Set(i for i in eachindex(col_of) if col_of[i] == 0)
    queue = collect(reached)
    while !isempty(queue)
        i = popfirst!(queue)
        for j in adjacency[i]
            k = get(row_of, j, 0)
            # Loop invariant: every row in `reached` is unmatched or matched to a column next to a reached row.
            if k != 0 && !(k in reached)
                push!(reached, k)
                push!(queue, k)
            end
        end
    end
    sort!(collect(reached))
end

"""
    check_structure(fx, fz, gx, gz)

Throw `ArgumentError` naming the equations when the DAE is structurally singular: when the
equations `ode` and `alg`, with each state identified with its derivatives, have no
perfect matching to the states. Pantelides' algorithm ends exactly when they have one.
"""
function check_structure(fx, fz, gx, gz)
    nx = length(fx)
    adjacency = [[[l]; fx[l]; nx .+ fz[l]] for l in 1:nx]
    append!(adjacency, [[gx[i]; nx .+ gz[i]] for i in eachindex(gx)])
    col_of = maximum_matching(unique!.(sort!.(adjacency)), nx + length(gz))
    all(!iszero, col_of) && return nothing
    names = [i <= nx ? "ode[$i]" : "alg[$(i - nx)]" for i in overdetermined(adjacency, col_of)]
    throw(ArgumentError("reduce_index: the equations $(join(names, ", ")) are structurally singular: they cannot determine distinct states, whatever their derivatives"))
end

"""
    pantelides(fx, fz, gx, gz) -> NamedTuple

Run Pantelides' algorithm 4.1 on the algebraic equations, with the dependencies `fx`, `fz`
of `ode` and `gx`, `gz` of `alg` on `x` and `z`. Return the graph: for each variable, its
algebraic state `var_base`, its order of derivative and the index of its derivative
(`var_der`, 0 for none); the same for each equation (`eq_base`, `eq_order`, `eq_der`).
Variables and equations `1:nz` are those of the DAE.
"""
function pantelides(fx, fz, gx, gz)
    nz = length(gz)
    var_base, var_order, var_der, var_assign = collect(1:nz), zeros(Int, nz), zeros(Int, nz), zeros(Int, nz)
    eq_base, eq_order, eq_der, eq_assign = collect(1:nz), zeros(Int, nz), zeros(Int, nz), zeros(Int, nz)
    eq_vars, eq_states = [copy(v) for v in gz], [copy(v) for v in gx]
    var_colored, eq_colored = falses(nz), falses(nz)
    deleted(j) = var_der[j] != 0
    function augment(i)
        eq_colored[i] = true
        for j in eq_vars[i]
            if !deleted(j) && var_assign[j] == 0
                var_assign[j], eq_assign[i] = i, j
                return true
            end
        end
        for j in eq_vars[i]
            if !deleted(j) && !var_colored[j]
                var_colored[j] = true
                if augment(var_assign[j])
                    var_assign[j], eq_assign[i] = i, j
                    return true
                end
            end
        end
        false
    end
    # The structure is nonsingular (`check_structure`), so each equation is differentiated
    # fewer times than there are states and algebraic states.
    limit = length(fx) + nz
    for k in 1:nz
        i = k
        while true
            # Loop invariant: equation i is the newest derivative of equation k.
            @ensures eq_der[i] == 0 && eq_base[i] == k "reduce_index: Pantelides' algorithm lost the derivative of alg[$k]"
            fill!(var_colored, false)
            fill!(eq_colored, false)
            augment(i) && break
            @ensures eq_order[i] < limit "reduce_index: equation alg[$k] needs more than $limit differentiations"
            colored_vars = [j for j in eachindex(var_colored) if var_colored[j] && !deleted(j)]
            colored_eqs = findall(eq_colored)
            for j in colored_vars
                push!(var_base, var_base[j]); push!(var_order, var_order[j] + 1)
                push!(var_der, 0); push!(var_assign, 0); push!(var_colored, false)
                var_der[j] = length(var_der)
            end
            for l in colored_eqs
                @ensures all(deleted, eq_vars[l]) "reduce_index: a variable of a differentiated equation has no derivative"
                # d/dt of equation l: the variables, their derivatives, and the algebraic states of `ode` for the states in l.
                vars = union(eq_vars[l], (var_der[j] for j in eq_vars[l]), (j for s in eq_states[l] for j in fz[s]))
                push!(eq_vars, sort!(vars)); push!(eq_states, sort!(union(eq_states[l], (r for s in eq_states[l] for r in fx[s]))))
                push!(eq_base, eq_base[l]); push!(eq_order, eq_order[l] + 1)
                push!(eq_der, 0); push!(eq_assign, 0); push!(eq_colored, false)
                eq_der[l] = length(eq_der)
            end
            for j in colored_vars
                var_assign[var_der[j]] = eq_der[var_assign[j]]
                eq_assign[eq_der[var_assign[j]]] = var_der[j]
            end
            i = eq_der[i]
        end
    end
    tops = highest(eq_der, 1:nz)
    @ensures sort(eq_assign[tops]) == sort(highest(var_der, 1:nz)) "reduce_index: the highest derivatives of the equations are not matched to those of the algebraic states"
    (; var_base, var_order, var_der, eq_base, eq_order, eq_der)
end

"""
    scalar_symbols(v) -> (symbols, fresh)

Return a scalar symbol for each element of the column of symbols `v`. The elements of an MX
vector symbol are not symbols, so they get new ones, named as the elements of an SX vector
symbol (`name_0`, `name_1`, ...), and `fresh` is true.
"""
function scalar_symbols(v::C.MX)
    parts = C.vertsplit(v)
    all(C.is_symbolic, parts) && return (parts, false)
    base = C.name(v)
    ([C.MX.sym("$(base)_$(k - 1)") for k in 1:size(v, 1)], true)
end
scalar_symbols(v::C.SX) = (C.vertsplit(v), false)

"""
    reduced_dae(dae, graph, X, pole) -> NamedTuple

Return `(; dae, invariants)` of the reduced DAE of `dae` with the structure `graph` of
[`pantelides`](@ref), in the symbol type `X`, with the Baumgarte pole `pole`.
"""
function reduced_dae(dae, graph, X, pole)
    (; var_base, var_der, eq_base, eq_order, eq_der) = graph
    x, f, z, g = dae["x"], dae["ode"], dae["z"], dae["alg"]
    nz = size(z, 1)
    t = get(dae, "t", nothing)
    zs, fresh = scalar_symbols(z)
    fresh && ((f, g) = (C.substitute(f, z, Front.vertcat(zs)), C.substitute(g, z, Front.vertcat(zs))))
    syms = Vector{X}(undef, length(var_base))
    syms[1:nz] = zs
    for j in eachindex(var_der)
        var_der[j] != 0 && (syms[var_der[j]] = X.sym("d" * C.name(syms[j])))
    end
    exprs = Vector{X}(undef, length(eq_base))
    exprs[1:nz] = C.vertsplit(g)
    known = [j for j in eachindex(var_der) if var_der[j] != 0]
    time, slope = t === nothing ? (X[], X[]) : ([t], [X(1)])
    arguments = Front.vertcat(vcat([x], syms[known], time))
    seeds = Front.vertcat(vcat([f], syms[var_der[known]], slope))
    for order in 1:maximum(eq_order)
        targets = [e for e in eachindex(eq_der) if eq_der[e] != 0 && eq_order[e] == order - 1]
        derivatives = C.vertsplit(Front.jtimes(Front.vertcat(exprs[targets]), arguments, seeds))
        exprs[eq_der[targets]] = derivatives
    end
    chain(start, der) = (c = [start]; while der[c[end]] != 0; push!(c, der[c[end]]); end; c)
    var_chains, eq_chains = [chain(j, var_der) for j in 1:nz], [chain(i, eq_der) for i in 1:nz]
    states = reduce(vcat, (c[1:end-1] for c in var_chains); init = Int[])
    alg = map(eq_chains) do c
        k = length(c) - 1
        pole == 0 && return exprs[c[end]]
        reduce(+, (binomial(k, j) * (-pole)^j * exprs[c[end-j]] for j in 1:k); init = exprs[c[end]])
    end
    algebraic = Front.vertcat(syms[last.(var_chains)])
    result = Dict{String,X}("x" => Front.vertcat(vcat([x], syms[states])), "z" => algebraic,
                            "ode" => Front.vertcat(vcat([f], syms[var_der[states]])), "alg" => Front.vertcat(alg))
    haskey(dae, "p") && (result["p"] = dae["p"])
    t === nothing || (result["t"] = t)
    invariants = Front.vertcat(reduce(vcat, (exprs[c[1:end-1]] for c in eq_chains); init = X[]))
    jac = dependencies(C.jacobian_sparsity(result["alg"], algebraic))
    col_of = maximum_matching(jac, nz)
    if !all(!iszero, col_of)
        names = ["alg[$i]" for i in overdetermined(jac, col_of)]
        throw(ArgumentError("reduce_index: after differentiation, the equations $(join(names, ", ")) do not determine the algebraic states, because a dependency cancels"))
    end
    @ensures size(result["x"]) == size(result["ode"]) && size(result["z"]) == size(result["alg"]) "reduce_index: the reduced DAE is not square"
    (; dae = result, invariants = size(invariants, 1) == 0 ? X(0, 1) : invariants)
end
