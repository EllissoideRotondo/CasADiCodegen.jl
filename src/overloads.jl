# Methods of ordinary Julia functions that the official front end leaves
# without a definition for symbols.

"A CasADi symbolic matrix: scalar graph (`SX`) or matrix graph (`MX`)."
const Symbolic = Union{SX,MX}

# Functions of `Base` that the front end does not cover, elementwise as `log10` of the
# front end.
Base.log2(x::Symbolic) = log(x) / log(2.0)
Base.exp2(x::T) where {T<:Symbolic} = C.power(convert(T, 2.0), x)
Base.abs2(x::Symbolic) = C.times(x, x)

"""
    inv(A::MX, lsolver)

Return the inverse of the square matrix `A` from the linear solver `lsolver` of CasADi, for
example `"qr"`. Throws `DimensionMismatch` when `A` is not square.
"""
function Base.inv(A::MX, lsolver::AbstractString)
    size(A, 1) == size(A, 2) || throw(DimensionMismatch("the matrix is not square: its size is $(size(A))"))
    C.inv(A, lsolver)
end
SpecialFunctions.erf(x::Symbolic) = C.erf(x)
SpecialFunctions.erfinv(x::Symbolic) = C.erfinv(x)

"""
    norm(x::Union{SX,MX}, p::Real = 2)

Return the `p`-norm of the elements of `x` as a 1 x 1 matrix, as `LinearAlgebra.norm` of a
Julia matrix: `x` is a vector of all its elements, structural zeros included.
`p = 2` gives the Frobenius norm, `p = Inf` and `p = -Inf` the largest and the smallest
absolute value, and `p = 0` the number of nonzero elements. The norm of an empty `x` is 0.
`opnorm` gives the matrix norm induced by a vector norm.
"""
function LinearAlgebra.norm(x::T, p::Real = 2) where {T<:Symbolic}
    size(x, 1) * size(x, 2) == 0 && return T(0.0)
    p == 2 && return C.norm_fro(x)
    # The 1 and infinity norms of CasADi are those of the elements, not of the matrix.
    p == 1 && return C.norm_1(x)
    p == Inf && return C.norm_inf(x)
    p == -Inf && return C.mmin(C.abs(x))
    p == 0 && return C.sum(C.ne(x, T(0.0)))
    # A structural zero gives an infinite term when p < 0.
    a = C.abs(p < 0 ? C.densify(x) : x)
    # CasADi takes the exponent as a Float64.
    q = Float64(p)
    -1 <= q <= 1 && return C.power(C.sum(C.power(a, T(q))), T(inv(q)))
    # As LinearAlgebra, scale by the largest (p > 1) or smallest (p < -1) absolute value
    # against overflow; a scale that is 0 or infinite is the norm.
    scale = p > 1 ? C.norm_inf(x) : C.mmin(a)
    scaled = scale * C.power(C.sum(C.power(a / scale, T(q))), T(inv(q)))
    C.if_else(C.logic_or(C.eq(scale, T(0.0)), C.eq(scale, T(Inf))), scale, scaled)
end

"""
    opnorm(A::Union{SX,MX}, p::Real = 2)

Return the norm of the matrix `A` induced by the vector `p`-norm, as a 1 x 1 matrix, as
`LinearAlgebra.opnorm`: the largest sum of absolute values of a column for `p = 1`, of a row
for `p = Inf`. CasADi has no symbolic singular values, so `p = 2` takes a row or a column only,
where it is the 2-norm. Throws `ArgumentError` for another `p` or a matrix with `p = 2`.
"""
function LinearAlgebra.opnorm(A::T, p::Real = 2) where {T<:Symbolic}
    p in (1, 2, Inf) || throw(ArgumentError("opnorm: p must be 1, 2 or Inf, got $p"))
    size(A, 1) * size(A, 2) == 0 && return T(0.0)
    p == 1 && return C.mmax(C.sum1(C.abs(A)))
    p == Inf && return C.mmax(C.sum2(C.abs(A)))
    1 in size(A) || throw(ArgumentError("opnorm: p = 2 needs a row or a column, but the matrix has size $(size(A)); " *
                                         "use p = 1 or p = Inf, or norm(A) for the Frobenius norm"))
    C.norm_fro(A)
end

# Julia array code on symbols. A CasADi matrix is not an `AbstractArray`, so the methods
# below give it the reductions, the iteration and the mixed arithmetic of an array; a
# Julia array of symbols that meets a CasADi matrix becomes one.

"A matrix of CasADi: numeric (`DM`) or symbolic."
const CasadiMatrix = Union{SX,MX,C.DM}

"""
    casadi_matrix(T, a) -> T

Return the Julia array `a` of numbers and symbols as one CasADi matrix of type `T` with the
shape of `a`; a vector becomes a column.
"""
casadi_matrix(::Type{T}, a::AbstractVector{<:Real}) where {T} = T(collect(Float64, a))
casadi_matrix(::Type{T}, a::AbstractMatrix{<:Real}) where {T} = T(Matrix{Float64}(a))
casadi_matrix(::Type{T}, a::AbstractVector) where {T} = isempty(a) ? T(0, 1) : C._vertcat(T[graph_element(T, e) for e in a])
casadi_matrix(::Type{T}, a::AbstractMatrix) where {T} =
    C._vertcat(T[C._horzcat(T[graph_element(T, a[i, j]) for j in axes(a, 2)]) for i in axes(a, 1)])

"Return the number or CasADi matrix `e` as a matrix of type `T`; an SX in an MX graph, or the reverse, throws."
graph_element(::Type{T}, e) where {T} = e isa T ? e : e isa Symbolic ? throw(mixed_graphs(e, T)) : convert(T, e)

# CasADi has no conversion between the two graphs, so one expression holds SX or MX only.
mixed_graphs(x, T) =
    ArgumentError("cannot join an $(typeof(x)) to an $T: CasADi cannot combine SX and MX in one expression; " *
                  "build the SX part from MX symbols, or call an SX Function of it on MX arguments, " *
                  "or expand an MX Function to SX")

# The front end joins numbers, Julia arrays of numbers and CasADi matrices. Every
# concatenation of the front end checks its pieces with `_catcheck`, so a Julia array of
# symbols among them becomes one matrix there, and SX and MX together throw.
function Front._catcheck(pieces::Vector{Any}, d)
    T = graph_type(pieces)
    for x in pieces, e in (x isa AbstractArray ? x : ())
        e isa CasadiMatrix && size(e, 2) > 1 &&
            throw(ArgumentError("concatenation: an element of a Julia array is a $(size(e, 1)) x $(size(e, 2)) matrix, " *
                                "not a column; join the matrix itself"))
    end
    joined = Any[x isa AbstractArray && !(eltype(x) <: Real) ? casadi_matrix(T, x) : x for x in pieces]
    foreach(x -> x isa Symbolic && graph_element(T, x), joined)
    invoke(Front._catcheck, Tuple{Any,Any}, joined, d)
end

"Return `MX` when the pieces, or the elements of an array among them, hold an MX; `SX` when they hold an SX; `DM` otherwise."
function graph_type(pieces)
    holds(T) = any(x -> x isa T || (x isa AbstractArray && any(e -> e isa T, x)), pieces)
    holds(MX) ? MX : holds(SX) ? SX : C.DM
end

# As LinearAlgebra, an element of a Julia array that is not a scalar enters with its 2-norm.
function LinearAlgebra.norm(a::AbstractArray{T}, p::Real = 2) where {T<:Symbolic}
    elements = all(e -> size(e) == (1, 1), a) ? vec(a) : T[LinearAlgebra.norm(e) for e in vec(a)]
    LinearAlgebra.norm(casadi_matrix(T, elements), p)
end
LinearAlgebra.opnorm(A::AbstractMatrix{T}, p::Real = 2) where {T<:Symbolic} = LinearAlgebra.opnorm(casadi_matrix(T, A), p)

# The elements of a symbolic matrix are its 1 x 1 matrices; those of a `DM` are numbers.
Base.iterate(x::Symbolic, k::Int = 1) = k > length(x) ? nothing : (x[k], k + 1)
Base.iterate(d::C.DM, (m, k) = (Matrix{Float64}(d), 1)) = k > length(m) ? nothing : (m[k], (m, k + 1))
Base.IteratorSize(::Type{C.DM}) = Base.HasShape{2}()
Base.IteratorSize(::Type{<:Symbolic}) = Base.HasShape{2}()
Base.eltype(::Type{T}) where {T<:Symbolic} = T
Base.ndims(::Symbolic) = 2

"""
    sum(x::Union{SX,MX}; dims = :, init = 0)

Return the sum of the elements of the symbolic matrix `x` as a 1 x 1 matrix, or with
`dims = 1` or `dims = 2` the row of the sums of its columns or the column of the sums of its rows.
`init` is added to each sum.
"""
function Base.sum(x::Symbolic; dims::Union{Colon,Integer} = :, init::Union{Real,Symbolic} = 0)
    dims isa Colon || dims >= 1 || throw(ArgumentError("sum: dims must be at least 1, got $dims"))
    total = dims isa Colon ? C.sum(x) : dims == 1 ? C.sum1(x) : dims == 2 ? C.sum2(x) : x
    init isa Real && iszero(init) ? total : total + init
end
Base.sum(f, x::Symbolic; kwargs...) = sum(f.(x); kwargs...)
# A `DM` sums as the Julia matrix of its values.
Base.sum(d::C.DM; kwargs...) = sum(Matrix{Float64}(d); kwargs...)
Base.sum(f, d::C.DM; kwargs...) = sum(f, Matrix{Float64}(d); kwargs...)

# Numbers promote to symbols, so that `[x, 1.0]` is a vector of symbols.
Base.promote_rule(::Type{T}, ::Type{<:Union{AbstractFloat,Integer,Rational,AbstractIrrational}}) where {T<:Symbolic} = T
Base.promote_rule(::Type{T}, ::Type{C.DM}) where {T<:Symbolic} = T

# The front end takes Julia arrays of numbers; a Julia array of symbols becomes a matrix.
for op in (:+, :-, :*, :\)
    @eval begin
        Base.$op(a::T, b::AbstractVecOrMat{<:Symbolic}) where {T<:Symbolic} = $op(a, casadi_matrix(T, b))
        Base.$op(a::AbstractVecOrMat{<:Symbolic}, b::T) where {T<:Symbolic} = $op(casadi_matrix(T, a), b)
    end
end
LinearAlgebra.dot(a::T, b::AbstractVector{<:Symbolic}) where {T<:Symbolic} = LinearAlgebra.dot(a, casadi_matrix(T, b))
LinearAlgebra.dot(a::AbstractVector{<:Symbolic}, b::T) where {T<:Symbolic} = LinearAlgebra.dot(casadi_matrix(T, a), b)

# A broadcast without CasADi matrices inside one with them, such as `v .* 2` in `x .+ v .* 2`
# with a Julia vector `v`, is a Julia array, which the front end then takes as an operand.
Front._bceval(bc::Base.Broadcast.Broadcasted) = Base.Broadcast.materialize(bc)

# CasADi applies an operation to whole matrices of one shape only. When the operands
# have different shapes, such as a column and a row, the operation applies to each
# element with the broadcast rules of Julia.
function Front._bceval(bc::Base.Broadcast.Broadcasted{Front.CasadiStyle,<:Any,<:Function})
    args = map(Front._bceval, bc.args)
    bc.f === Base.literal_pow && return Front._apply(bc.f, args...)
    op = Front._bcop(bc.f)
    op === nothing || !same_shape(args) ? Front._bcelem(bc.f, args) : op(args...)
end

"Return whether the operands `args` that are matrices and not 1 x 1 all have one shape; a vector is a column."
function same_shape(args)
    shapes = [(size(a, 1), size(a, 2)) for a in args if (a isa CasadiMatrix || a isa AbstractArray) && length(a) != 1]
    all(==(first(shapes, 1)...), shapes)
end

# In-place broadcasting into a Julia array or a CasADi matrix.
function Base.copyto!(dest::AbstractArray, bc::Base.Broadcast.Broadcasted{Front.CasadiStyle})
    value = Front._bceval(bc)
    n = length(value)
    n == 1 || n == length(dest) || throw(DimensionMismatch("broadcast: a result of size $(size(value)) does not fit an array of size $(size(dest))"))
    for (k, i) in enumerate(eachindex(dest))
        dest[i] = element_value(value, n == 1 ? 1 : k)
    end
    dest
end
element_value(value::C.DM, k) = Float64(value[k])
element_value(value, k) = value[k]
function Base.copyto!(dest::CasadiMatrix, bc::Base.Broadcast.Broadcasted{Front.CasadiStyle})
    dest[:, :] = convert(typeof(dest), Front._bceval(bc))
    dest
end

"""
    vertcat(parts::AbstractVector) -> CasADi matrix
    horzcat(parts::AbstractVector) -> CasADi matrix

Join the CasADi matrices and numbers `parts` into one column of blocks or one row of
blocks, as `vertcat` and `horzcat` of CasADi; an empty `parts` gives a 0 x 0 matrix of
its element type, or a `DM`.
"""
vertcat(parts::AbstractVector) = isempty(parts) ? empty_matrix(parts) : C._vertcat(collect(parts))
horzcat(parts::AbstractVector) = isempty(parts) ? empty_matrix(parts) : C._horzcat(collect(parts))
empty_matrix(parts) = (eltype(parts) <: CasadiMatrix ? eltype(parts) : C.DM)(0, 0)
