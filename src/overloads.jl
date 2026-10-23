# Methods of ordinary Julia functions that the official front end leaves
# without a definition for symbols.

"A CasADi symbolic matrix: scalar graph (`SX`) or matrix graph (`MX`)."
const Symbolic = Union{SX,MX}

# Functions of `Base` that the front end does not cover. Each maps to the CasADi
# operation with the same meaning for ordinary numbers.
for (f, c) in ((:asinh, :asinh), (:acosh, :acosh), (:atanh, :atanh), (:log1p, :log1p),
               (:expm1, :expm1), (:log10, :log10), (:!, :not))
    @eval Base.$f(x::Symbolic) = C.$c(x)
end
Base.log2(x::Symbolic) = log(x) / log(2.0)
Base.exp2(x::T) where {T<:Symbolic} = C.power(convert(T, 2.0), x)
Base.abs2(x::Symbolic) = C.times(x, x)

"""
    inv(A)
    inv(A, lsolver)

Return the inverse of a symbolic square matrix, or the reciprocal of a scalar.

For an `MX` matrix, `inv` of the CasADi front end returns the reciprocal of
each element. This method takes the inverse from the linear solver `"qr"`, or
from `lsolver`. CasADi accepts `lsolver` for `MX` only.

Throws `DimensionMismatch` when `A` is not square.
"""
function Base.inv(A::Symbolic)
    C.is_scalar(A) && return 1 / A
    size(A, 1) == size(A, 2) || throw(DimensionMismatch("the matrix is not square: its size is $(size(A))"))
    A isa SX ? C.inv(A) : C.inv(A, "qr")
end
function Base.inv(A::MX, lsolver::AbstractString)
    size(A, 1) == size(A, 2) || throw(DimensionMismatch("the matrix is not square: its size is $(size(A))"))
    C.inv(A, lsolver)
end
SpecialFunctions.erf(x::Symbolic) = C.erf(x)
SpecialFunctions.erfinv(x::Symbolic) = C.erfinv(x)

for (f, c) in ((:min, :fmin), (:max, :fmax), (:atan, :atan2), (:hypot, :hypot), (:rem, :fmod),
               (:copysign, :copysign), (:&, :and), (:|, :or), (:!=, :ne))
    @eval begin
        Base.$f(x::T, y::T) where {T<:Symbolic} = C.$c(x, y)
        Base.$f(x::T, y::Real) where {T<:Symbolic} = C.$c(x, convert(T, y))
        Base.$f(x::Real, y::T) where {T<:Symbolic} = C.$c(convert(T, x), y)
    end
end

"""
    ifelse(condition, a, b)

Return the symbolic selection: `a` where `condition` is not zero, else `b`.
Both branches are evaluated, as in CasADi `if_else`.
"""
Base.ifelse(condition::T, a::Union{T,Real}, b::Union{T,Real}) where {T<:Symbolic} =
    C.if_else(condition, convert(T, a), convert(T, b))

# The front end exports its own `dot` and `cross`; these make the LinearAlgebra
# functions of the same name work too.
LinearAlgebra.dot(a::T, b::T) where {T<:Symbolic} = Front.dot(a, b)
LinearAlgebra.cross(a::T, b::T) where {T<:Symbolic} = Front.cross(a, b)
# As in LinearAlgebra, `norm` is the Frobenius norm: the 2-norm of a vector.
LinearAlgebra.norm(x::Symbolic) = C.norm_fro(x)

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
casadi_matrix(::Type{T}, a::AbstractVector) where {T} = isempty(a) ? T(0, 1) : Front.vertcat(T[Front._ascat(T, e) for e in a])
casadi_matrix(::Type{T}, a::AbstractMatrix) where {T} =
    Front.vertcat(T[Front.horzcat(T[Front._ascat(T, a[i, j]) for j in axes(a, 2)]) for i in axes(a, 1)])

# `zero` and `one` of a matrix, which the front end calls (see `shaped_zero_definitions!`).
"""
Return the zero of the shape of `x`, as `zero` of a Julia matrix: the structural zero, or
for a 1 x 1 `x` the dense scalar 0, which `Float64` converts as a scalar.
"""
zero_matrix(x::T) where {T<:CasadiMatrix} = size(x) == (1, 1) ? T(0.0) : T(size(x, 1), size(x, 2))
"Return the identity of the shape of the square matrix `x`, as `one` of a Julia matrix."
function identity_matrix(x::T) where {T<:CasadiMatrix}
    size(x, 1) == size(x, 2) || throw(DimensionMismatch("one: the matrix is not square: its size is $(size(x))"))
    C.eye(T, size(x, 1))
end

# Concatenation with Julia arrays. The front end concatenates CasADi matrices and numbers;
# these methods let a Julia vector or matrix of numbers and symbols join them, as an array
# joins arrays in Julia: `[x; v]` with a column `x` and a vector `v` is one column. A list of
# matrices, such as the arguments of a function, is written with commas: `[x, y]`.
"Return the CasADi matrix type of the result of concatenating `parts`."
function concatenated_type(parts)
    elements = Iterators.flatten(p isa AbstractArray ? p : (p,) for p in parts)
    any(e -> e isa MX, elements) ? MX : any(e -> e isa SX, elements) ? SX : C.DM
end

"""
    cat_part(T, a) -> T

Return the part `a` of a concatenation as a CasADi matrix of type `T`. A Julia vector
becomes a column and holds numbers or CasADi columns; a Julia matrix holds numbers or
1 x 1 CasADi matrices. Throws `ArgumentError` for another element.
"""
cat_part(::Type{T}, a::Union{CasadiMatrix,Real}) where {T} = Front._ascat(T, a)
function cat_part(::Type{T}, a::AbstractVector) where {T}
    all(e -> e isa Real || (e isa CasadiMatrix && size(e, 2) == 1), a) ||
        throw(ArgumentError("a Julia vector in a concatenation with a CasADi matrix holds numbers or columns; " *
                            "write a list of matrices with commas, [x, y], or join them with vertcat"))
    casadi_matrix(T, a)
end
function cat_part(::Type{T}, a::AbstractMatrix) where {T}
    all(e -> e isa Real || (e isa CasadiMatrix && size(e) == (1, 1)), a) ||
        throw(ArgumentError("a Julia matrix in a concatenation with a CasADi matrix holds numbers or 1 x 1 matrices; " *
                            "join larger blocks with blockcat"))
    casadi_matrix(T, a)
end
cat_parts(parts) = (T = concatenated_type(parts); [cat_part(T, p) for p in parts])

const CatPart = Union{CasadiMatrix,Real,AbstractVecOrMat}
Base.vcat(x::CasadiMatrix, xs::CatPart...) = Front.vertcat(cat_parts((x, xs...)))
Base.hcat(x::CasadiMatrix, xs::CatPart...) = Front.horzcat(cat_parts((x, xs...)))
function Base.hvcat(rows::Tuple{Vararg{Int}}, x::CasadiMatrix, xs::CatPart...)
    parts = cat_parts((x, xs...))
    sum(rows) == length(parts) || throw(ArgumentError("hvcat: the rows hold $(sum(rows)) blocks, but $(length(parts)) are given"))
    ends = cumsum(collect(rows))
    Front.vertcat([Front.horzcat(parts[e-n+1:e]) for (n, e) in zip(rows, ends)])
end

LinearAlgebra.norm(a::AbstractArray{T}) where {T<:Symbolic} = C.norm_fro(casadi_matrix(T, a))

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
Base.maximum(x::Symbolic) = C.mmax(x)
Base.minimum(x::Symbolic) = C.mmin(x)

# Numbers promote to symbols, so that `[x, 1.0]` is a vector of symbols.
Base.promote_rule(::Type{T}, ::Type{<:Union{AbstractFloat,Integer,Rational,AbstractIrrational}}) where {T<:Symbolic} = T
Base.promote_rule(::Type{T}, ::Type{C.DM}) where {T<:Symbolic} = T

for op in (:+, :-, :*, :\)
    @eval begin
        Base.$op(a::T, b::AbstractVecOrMat) where {T<:Symbolic} = $op(a, casadi_matrix(T, b))
        Base.$op(a::AbstractVecOrMat, b::T) where {T<:Symbolic} = $op(casadi_matrix(T, a), b)
    end
end
LinearAlgebra.dot(a::T, b::AbstractVector) where {T<:Symbolic} = Front.dot(a, casadi_matrix(T, b))
LinearAlgebra.dot(a::AbstractVector, b::T) where {T<:Symbolic} = Front.dot(casadi_matrix(T, a), b)

# Indexing by a vector of indices or a vector of `Bool` that selects elements; the front
# end takes integers, ranges and colons.
const Indices = Union{Integer,AbstractRange,Colon,Vector{<:Integer}}
index_list(i::Vector{Bool}) = findall(i)
index_list(i) = i
picked(x, i, d) = i isa Colon ? (1:size(x, d)) : i isa Integer ? (i:i) : index_list(i)
function Base.getindex(x::Symbolic, i::Vector{<:Integer})
    k = index_list(i)
    isempty(k) ? typeof(x)(0, 1) : Front.vertcat([x[j] for j in k])
end
Base.getindex(x::Symbolic, i::Vector{<:Integer}, j::Indices) = selected(x, i, j)
Base.getindex(x::Symbolic, i::Union{Integer,AbstractRange,Colon}, j::Vector{<:Integer}) = selected(x, i, j)
selected(x, i, j) = casadi_matrix(typeof(x), [x[r, c] for r in picked(x, i, 1), c in picked(x, j, 2)])

# Broadcasting. The front end applies a broadcast function to whole CasADi matrices, which
# is right for the functions below, elementwise in CasADi; `*`, `/` and `^` become the
# elementwise products of CasADi. Another function, such as `inv` or a closure, is applied to
# each element.
const WHOLE_MATRIX = (+, -, *, /, ^, sin, cos, tan, exp, exp2, log, sqrt, abs, tanh, sinh, cosh, asin, acos, atan,
                      floor, ceil, sign, asinh, acosh, atanh, log1p, expm1, log10, log2, abs2, min, max, hypot,
                      rem, copysign, &, |, !, !=, <, <=, >, >=, ==, ifelse, identity, SpecialFunctions.erf,
                      SpecialFunctions.erfinv)
whole_matrix(f) = any(g -> g === f, WHOLE_MATRIX)

# A Julia array of symbols or numbers in a broadcast takes the type of the CasADi matrices.
Front._castype(a::AbstractArray, rest...) =
    foldl((T, e) -> e isa CasadiMatrix && Front._rank(typeof(e)) > Front._rank(T) ? typeof(e) : T, a; init = Front._castype(rest...))
Front._coerce(::Type{T}, a::AbstractArray) where {T} = casadi_matrix(T, a)
# A nested broadcast without CasADi matrices is a Julia array.
Front._bceval(bc::Base.Broadcast.Broadcasted) = Base.Broadcast.materialize(bc)
function Front._bceval(bc::Base.Broadcast.Broadcasted{Front.CasadiStyle,<:Any,<:Function})
    args = map(Front._bceval, bc.args)
    bc.f === Base.literal_pow && return Front._apply(bc.f, args...)
    T = Front._castype(args...)
    values = map(a -> Front._coerce(T, a), args)
    whole_matrix(bc.f) && same_shape(values) ? Front._bcop(bc.f)(values...) : elementwise(bc.f, T, values)
end

# CasADi applies a function to whole matrices of one size, or to a matrix and a 1 x 1 one;
# other shapes, such as a column and a row, are broadcast element by element.
function same_shape(values)
    shapes = [size(v) for v in values if v isa CasadiMatrix && size(v) != (1, 1)]
    isempty(shapes) || all(==(first(shapes)), shapes)
end

"""
    elementwise(f, T, values) -> T

Return the CasADi matrix of type `T` whose element (i, j) is `f` of the elements (i, j) of
`values`. A CasADi matrix of one row or one column repeats it, as in broadcasting; another
value, such as a number, is passed whole.
"""
function elementwise(f, ::Type{T}, values) where {T}
    matrices = filter(v -> v isa CasadiMatrix, collect(Any, values))
    rows, cols = maximum(m -> size(m, 1), matrices), maximum(m -> size(m, 2), matrices)
    for m in matrices
        size(m, 1) in (1, rows) && size(m, 2) in (1, cols) ||
            throw(DimensionMismatch("broadcast: a matrix of size $(size(m)) does not match the size ($rows, $cols)"))
    end
    casadi_matrix(T, [f(map(v -> broadcast_element(v, i, j), values)...) for i in 1:rows, j in 1:cols])
end
broadcast_element(v::CasadiMatrix, i, j) = v[size(v, 1) == 1 ? 1 : i, size(v, 2) == 1 ? 1 : j]
broadcast_element(v::Base.RefValue, i, j) = v[]
broadcast_element(v, i, j) = v

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
    bounded(lb, x, ub)

Return the constraint `lb <= x <= ub` on the expression `x` of an `Opti` object, for
`subject_to`. The bounds are numbers, `DM` matrices or Julia arrays of numbers.

A chained comparison such as `lb <= x <= ub` cannot give it: Julia evaluates
`lb <= x && x <= ub`, and `&&` takes a `Bool` only.
"""
Front.bounded(lb::AbstractVecOrMat{<:Real}, ex::MX, ub) = Front.bounded(casadi_matrix(MX, lb), ex, ub)
Front.bounded(lb::Union{Real,C.DM,MX}, ex::MX, ub::AbstractVecOrMat{<:Real}) = Front.bounded(lb, ex, casadi_matrix(MX, ub))
