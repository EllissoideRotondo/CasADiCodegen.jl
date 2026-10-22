# A sparse LU factorization for matrices of one fixed sparsity pattern, for the Newton
# rootfinder of CasADi in generated code. It is generic in the number type, so that dual
# numbers go through it, and it needs no package beyond Julia.

"The least number of unknowns for which a Newton rootfinder with a sparse Jacobian gets a sparse factorization."
const SPARSE_NEWTON_SIZE = 16

"Whether a Newton rootfinder with `n` unknowns and `nnz` stored elements in its Jacobian factors it as a sparse matrix."
sparse_newton(n::Int, nnz::Int) = n >= SPARSE_NEWTON_SIZE && 4 * nnz <= n^2

"""
    SparseLU{T}(n, rows, columns)

The work space of the factorization `P A = L U` of an `n` by `n` matrix `A` whose stored
element `k` is at row `rows[k]` and column `columns[k]`, in column-major order (the order
of CasADi). `factor!` computes the factors from the values of the stored elements, and
`solve!` solves a linear system with them.

The method is the left-looking LU factorization of Gilbert and Peierls with partial
pivoting, as `cs_lu` of CSparse, in the natural order of the columns: the fill of `L` and
`U` follows the order of the unknowns, which suits a banded matrix. The vectors of the
factors keep their capacity, so a factorization allocates no memory after the first one.
"""
mutable struct SparseLU{T}
    const n::Int
    const colptr::Vector{Int}                   # the pattern of A: column j has its stored elements at colptr[j]:colptr[j+1]-1
    const rowval::Vector{Int}
    const Lp::Vector{Int}                       # the columns of L, unit diagonal first, with the rows of A until the factorization ends
    const Li::Vector{Int}
    const Lx::Vector{T}
    const Up::Vector{Int}                       # the columns of U, diagonal last, with the rows in the order of the pivots
    const Ui::Vector{Int}
    const Ux::Vector{T}
    const pinv::Vector{Int}                     # pinv[i] is the column whose pivot is row i of A, or 0 before that
    const x::Vector{T}                          # a dense column, zero outside a factorization step
    const reached::Vector{Int}                  # reached[top:n] are the rows that column k reaches, in topological order
    const stack::Vector{Int}
    const next::Vector{Int}
    const marked::Vector{Bool}
    singular::Bool                              # no row could be the pivot of a column
end

function SparseLU{T}(n::Int, rows::Vector{Int}, columns::Vector{Int}) where {T}
    @requires length(rows) == length(columns) && all(r -> 1 <= r <= n, rows) && issorted(columns) && all(c -> 1 <= c <= n, columns) "the stored elements of a sparse matrix must be in column-major order inside an $n by $n matrix"
    colptr = zeros(Int, n + 1)
    colptr[1] = 1
    for c in columns
        colptr[c+1] += 1
    end
    cumsum!(colptr, colptr)
    @ensures colptr[n+1] == length(rows) + 1 "every stored element must be in one column"
    vector(S) = Vector{S}(undef, n)
    SparseLU{T}(n, colptr, copy(rows), ones(Int, n + 1), Int[], T[], ones(Int, n + 1), Int[], T[], zeros(Int, n),
                zeros(T, n), vector(Int), vector(Int), vector(Int), fill(false, n), false)
end

"""
    reach!(F::SparseLU, k) -> top

Find the rows that the solution of `L x = A[:, k]` can make nonzero, by a depth-first
search in the graph of the columns of `L` done so far. They are `F.reached[top:F.n]`, in
an order in which each row comes before the rows that it changes.
"""
function reach!(F::SparseLU, k::Int)
    top = F.n + 1
    @inbounds for p in F.colptr[k]:F.colptr[k+1]-1
        start = F.rowval[p]
        F.marked[start] && continue
        head = 1
        F.stack[1] = start
        while head > 0
            # Loop invariant: F.stack[1:head] is the path of the search from `start`.
            j = F.stack[head]
            J = F.pinv[j]
            if !F.marked[j]
                F.marked[j] = true
                F.next[head] = J == 0 ? 0 : F.Lp[J] + 1        # the first element of a column of L is its diagonal
            end
            last = J == 0 ? 0 : F.Lp[J+1] - 1
            p = F.next[head]
            descended = false
            while J != 0 && p <= last
                i = F.Li[p]
                p += 1
                if !F.marked[i]
                    F.next[head] = p
                    head += 1
                    F.stack[head] = i
                    descended = true
                    break
                end
            end
            if !descended
                head -= 1
                top -= 1
                F.reached[top] = j
            end
        end
    end
    @inbounds for p in top:F.n
        F.marked[F.reached[p]] = false
    end
    top
end

"""
    factor!(F::SparseLU, values) -> F

Compute the factors of the matrix whose stored elements have the `values`. A matrix with
a column that no row can pivot sets `F.singular`; a pivot of zero gives non-finite numbers.
"""
function factor!(F::SparseLU{T}, values::AbstractVector) where {T}
    n = F.n
    @requires length(values) == length(F.rowval) "the matrix has $(length(F.rowval)) stored elements; got $(length(values)) values"
    fill!(F.pinv, 0)
    empty!(F.Li), empty!(F.Lx), empty!(F.Ui), empty!(F.Ux)
    F.singular = false
    @inbounds for k in 1:n
        # Loop invariant: columns 1:k-1 of L and U are done, and pinv names the pivot row of each of them.
        F.Lp[k], F.Up[k] = length(F.Li) + 1, length(F.Ui) + 1
        top = reach!(F, k)
        for p in F.colptr[k]:F.colptr[k+1]-1
            F.x[F.rowval[p]] = values[p]
        end
        for p in top:n
            j = F.reached[p]
            J = F.pinv[j]
            J == 0 && continue
            xj = F.x[j]
            for q in F.Lp[J]+1:F.Lp[J+1]-1
                F.x[F.Li[q]] -= F.Lx[q] * xj
            end
        end
        pivot_row, largest = 0, -one(real(T))
        for p in top:n
            i = F.reached[p]
            if F.pinv[i] == 0
                magnitude = abs(F.x[i])
                magnitude > largest && ((pivot_row, largest) = (i, magnitude))
            else
                push!(F.Ui, F.pinv[i])
                push!(F.Ux, F.x[i])
                F.x[i] = zero(T)
            end
        end
        if pivot_row == 0
            F.singular = true
            break
        end
        pivot = F.x[pivot_row]
        push!(F.Ui, k)
        push!(F.Ux, pivot)
        F.pinv[pivot_row] = k
        push!(F.Li, pivot_row)
        push!(F.Lx, one(T))
        for p in top:n
            i = F.reached[p]
            if F.pinv[i] == 0
                push!(F.Li, i)
                push!(F.Lx, F.x[i] / pivot)
            end
            F.x[i] = zero(T)
        end
    end
    F.singular && (fill!(F.x, zero(T)); return F)
    F.Lp[n+1], F.Up[n+1] = length(F.Li) + 1, length(F.Ui) + 1
    @inbounds for p in eachindex(F.Li)
        F.Li[p] = F.pinv[F.Li[p]]
    end
    F
end

"""
    solve!(F::SparseLU, b) -> b

Overwrite `b` with the solution of `A x = b` for the matrix `A` that `factor!` factored
last. A singular matrix gives NaN.
"""
function solve!(F::SparseLU{T}, b::AbstractVector) where {T}
    n = F.n
    @requires length(b) == n "the matrix has $n rows; got a vector of $(length(b))"
    F.singular && return fill!(b, T(NaN))
    y = F.x
    @inbounds for i in 1:n
        y[F.pinv[i]] = b[i]
    end
    @inbounds for j in 1:n
        for p in F.Lp[j]+1:F.Lp[j+1]-1
            y[F.Li[p]] -= F.Lx[p] * y[j]
        end
    end
    @inbounds for j in n:-1:1
        y[j] /= F.Ux[F.Up[j+1]-1]
        for p in F.Up[j]:F.Up[j+1]-2
            y[F.Ui[p]] -= F.Ux[p] * y[j]
        end
    end
    copyto!(b, y)
    fill!(y, zero(T))
    b
end
