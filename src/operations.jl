# Helper functions for the few operations that neither Base nor NaNMath covers
# with the semantics of the C library. Each is kept as source text, because a
# module written by `generate_julia` holds a copy of the ones it uses.
const HELPER_SOURCE = Dict{Symbol,String}(
    # C `fmin`/`fmax`: a NaN operand is ignored, and of two equal operands (such
    # as 0.0 and -0.0) the second is returned, as CasADi does on x86-64 Linux.
    :fmin => "fmin(x, y) = isnan(y) ? x : (x < y ? x : y)",
    :fmax => "fmax(x, y) = isnan(y) ? x : (x > y ? x : y)",
    # SpecialFunctions.erfinv throws outside [-1, 1]; the C library gives NaN.
    :erfinv => "erfinv(x) = abs(x) > one(x) ? oftype(float(x), NaN) : SpecialFunctions.erfinv(x)",
    # The structural nonzeros of an array in matrix shape, at the linear indices of a
    # `PatternCheck`, and whether an array is a sparse matrix, which holds them itself.
    # A `PatternCheck` also keeps the index vectors of the last sparse matrix whose
    # pattern was compared in full (see SPARSE_HELPER_SOURCE).
    :stored => """
mutable struct PatternCheck
    const indices::Vector{Int}
    @atomic seen::Any
end
PatternCheck(indices) = PatternCheck(indices, nothing)
stored(a, check::PatternCheck) = view(a, check.indices)
is_stored(a) = false""",
    # A vector of `n` elements of type `T` that belongs to the current task and
    # to the key type `K`. Generated code keeps its work vector there, so a call
    # allocates it once per task and not at each call; a longer vector replaces
    # one that is too short.
    # A static array from a tuple of values, as an argument of a Julia function,
    # and the call of a Julia function, which keeps its own bounds checks.
    :static => "static(::Type{S}, x::Tuple) where {S} = StaticArrays.SArray{S,promote_type(map(typeof, x)...),length(S.parameters),length(x)}(x)",
    :checked => "checked(f, arguments...) = f(arguments...)",
    # The error of a wrong number or size of arrays in a call of generated code. It is a
    # function of its own, so that the checks keep the code of a call small enough to inline.
    # Its arguments have concrete types, so that the count is not boxed and a call of
    # generated code needs no GC frame for this error path.
    :shape_error => "@noinline shape_error(message::String, count::Int) = throw(ArgumentError(string(message, count)))",
    # The place of `x` in the rising grid `w[grid+1:grid+n]`, from zero, as the CasADi routine `casadi_low`:
    # mode 0 searches from the start, mode 1 computes the place in a grid with equal steps, and mode 2
    # halves the range. A value that is no number, or too large for a whole number, has the place 0 in
    # mode 1, as on the processors where the conversion of C gives the smallest whole number.
    :low => """
function low(w, grid, n, x, mode)
    if mode == 1
        first = w[grid+1]
        v = (x - first) * (n - 1) / (w[grid+n] - first)
        return abs(v) < 9.2e18 ? clamp(trunc(Int, v)::Int, 0, n - 2) : 0
    elseif mode == 2
        (n < 2 || x < w[grid+2]) && return 0
        x > w[grid+n] && return n - 2
        start, stop = 0, n - 1
        while true
            pivot = (stop + start) ÷ 2
            if x < w[grid+pivot+1]
                pivot == stop && return pivot
                stop = pivot
            else
                pivot == start && return pivot
                start = pivot
            end
        end
    end
    i = 0
    while i < n - 2 && !(x < w[grid+i+2])
        i += 1
    end
    i
end""",
    # The whole part of a value as an index, and -1, which is no index, for a value that has none.
    :whole => "whole(x) = abs(x) < 9.0e15 ? trunc(Int, x) : -1",
    # Element `index`, from zero, of the `n` elements after `w[base]`, and NaN out of that range, as a
    # floating-point number: `w` can be an argument of integers.
    :gather => "gather(w, base, n, index) = convert(promote_type(Float64, eltype(w)), 0 <= index < n ? w[base+index+1] : NaN)",
    # The sparse QR factorization of CasADi (`casadi_qr`) of the matrix whose nonzeros follow `w[a]`.
    # The index table `t` holds after `t[base]`: the numbers of columns, of rows of V and of nonzeros of
    # A, V and R; then the column starts and the rows of A, of V and of R, the place of each row, and
    # the order of the columns, all from zero. The work vector gets after `w[s]`: the Householder
    # vectors V, the factor R, one number per column, and a column of work space.
    Symbol("qr_factor!") => """
function qr_factor!(w, t, base, a, s)
    n, m, na, nv, nr = t[base+1], t[base+2], t[base+3], t[base+4], t[base+5]
    acol = base + 5; arow = acol + n + 1; vcol = arow + na; vrow = vcol + n + 1
    rcol = vrow + nv; rrow = rcol + n + 1; prinv = rrow + nr; pc = prinv + m
    v = s; r = v + nv; beta = r + nr; x = beta + n
    @inbounds for i in 1:m
        w[x+i] = zero(eltype(w))
    end
    kr = 0
    @inbounds for c in 1:n
        c1 = t[pc+c]
        for k in t[acol+c1+1]+1:t[acol+c1+2]
            w[x+t[prinv+t[arow+k]+1]+1] = w[a+k]
        end
        # R = Q' A gives the entries of R above the diagonal.
        for k in t[rcol+c]+1:t[rcol+c+1]
            row = t[rrow+k] + 1
            row < c || break
            alpha = zero(eltype(w))
            for k1 in t[vcol+row]+1:t[vcol+row+1]
                alpha += w[v+k1] * w[x+t[vrow+k1]+1]
            end
            alpha *= w[beta+row]
            for k1 in t[vcol+row]+1:t[vcol+row+1]
                w[x+t[vrow+k1]+1] -= alpha * w[v+k1]
            end
            kr += 1
            w[r+kr] = w[x+row]
            w[x+row] = zero(eltype(w))
        end
        first, last = t[vcol+c] + 1, t[vcol+c+1]
        for k in first:last
            w[v+k] = w[x+t[vrow+k]+1]
            w[x+t[vrow+k]+1] = zero(eltype(w))
        end
        # The Householder vector of the column, as `casadi_house`.
        v0, sigma = w[v+first], zero(eltype(w))
        for k in first+1:last
            sigma += w[v+k] * w[v+k]
        end
        norm = sqrt(v0 * v0 + sigma)
        if sigma == 0
            w[v+first] = one(eltype(w))
            w[beta+c] = v0 <= 0 ? 2 * one(eltype(w)) : zero(eltype(w))
        else
            w[v+first] = v0 <= 0 ? v0 - norm : -sigma / (v0 + norm)
            w[beta+c] = -1 / (norm * w[v+first])
        end
        kr += 1
        w[r+kr] = norm
    end
    nothing
end""",
    # The solution of the system, or of the system with the transpose for `tr`, for each of the `nrhs`
    # right sides after `w[b]`, in their place, from the factors of `qr_factor!` (`casadi_qr_solve`).
    Symbol("qr_solve!") => """
function qr_solve!(w, t, base, s, b, nrhs, tr)
    n, m, na, nv, nr = t[base+1], t[base+2], t[base+3], t[base+4], t[base+5]
    vcol = base + 5 + n + 1 + na; vrow = vcol + n + 1
    rcol = vrow + nv; rrow = rcol + n + 1; prinv = rrow + nr; pc = prinv + m
    v = s; r = v + nv; beta = r + nr; x = beta + n
    @inbounds for j in 0:nrhs-1
        y = b + n * j
        if tr
            for c in 1:n
                w[x+c] = w[y+t[pc+c]+1]
            end
            for c in 1:n, k in t[rcol+c]+1:t[rcol+c+1]
                row = t[rrow+k] + 1
                row == c ? (w[x+c] /= w[r+k]) : (w[x+c] -= w[r+k] * w[x+row])
            end
        else
            for c in 1:m
                w[x+c] = zero(eltype(w))
            end
            for c in 1:n
                w[x+t[prinv+c]+1] = w[y+c]
            end
        end
        # The product with Q for the transpose, and with Q' otherwise.
        for step in 1:n
            c = tr ? n + 1 - step : step
            alpha = zero(eltype(w))
            for k in t[vcol+c]+1:t[vcol+c+1]
                alpha += w[v+k] * w[x+t[vrow+k]+1]
            end
            alpha *= w[beta+c]
            for k in t[vcol+c]+1:t[vcol+c+1]
                w[x+t[vrow+k]+1] -= alpha * w[v+k]
            end
        end
        if tr
            for c in 1:n
                w[y+c] = w[x+t[prinv+c]+1]
            end
        else
            for c in n:-1:1, k in t[rcol+c+1]:-1:t[rcol+c]+1
                row = t[rrow+k] + 1
                row == c ? (w[x+row] /= w[r+k]) : (w[x+row] -= w[r+k] * w[x+c])
            end
            for c in 1:n
                w[y+t[pc+c]+1] = w[x+c]
            end
        end
    end
    nothing
end""",
    :monitor => "monitor(text, values) = (println(text, \":\"); println(values); nothing)",
    # An assertion of the user: the condition holds only when it is 1.
    :assertion => "assertion(condition, text) = condition == 1 || throw(AssertionError(text))",
    # Print the second value, then the first one, which is the result.
    :printme => "printme(x, y) = (println(\"|> \", y, \" : \", x); x)",
    # The row of the first of the `n` nonzeros after `w[at]` that is not zero; the index table
    # has the rows after `t[base]`, and `last` is the result when every nonzero is zero.
    :find => """
function find(w, at, n, t, base, last)
    @inbounds for i in 1:n
        w[at+i] == 0 || return t[base+i]
    end
    last
end""",
    # A B-spline with a tensor product of bases, as the CasADi routine `casadi_nd_boor_eval`. The
    # index table holds after `t[tb]`: the numbers of dimensions and of results, the start of the
    # knots of each dimension and the end of the last, and the degree, the step in the coefficients
    # and the lookup mode of each dimension. The knots follow `k[kb]`, the coefficients `c[cb]`, the
    # point `w[x]`, the result `w[out]`, and the weights of the basis functions go after `w[s]`.
    Symbol("bspline!") => """
function bspline!(w, k, kb, t, tb, c, cb, x, out, s)
    n, m = t[tb+1], t[tb+2]
    off = tb + 2; deg = off + n + 1; str = deg + n; mode = str + n
    base, b = 0, s
    @inbounds for d in 1:n
        degree, knots, nk = t[deg+d], kb + t[off+d], t[off+d+1] - t[off+d]
        xd = w[x+d]
        place = low(k, knots + degree, nk - 2 * degree, xd, t[mode+d])
        start = min(place, nk - 2 * degree - 2)
        for i in 1:2*degree+1
            w[b+i] = zero(eltype(w))
        end
        if xd >= k[knots+1] && xd <= k[knots+nk]
            if xd == k[knots+2]
                for i in 1:degree+1
                    w[b+i] = one(eltype(w))
                end
            elseif xd == k[knots+nk]
                w[b+degree+1] = one(eltype(w))
            elseif k[knots+place+degree+1] == xd
                degree > 0 && (w[b+degree] = one(eltype(w)))
            else
                w[b+degree+1] = one(eltype(w))
            end
        end
        # The recurrence of de Boor on the knots of the span.
        for p in 1:degree, i in 0:2*degree-p
            v = zero(eltype(w))
            bottom = k[knots+start+i+p+1] - k[knots+start+i+1]
            bottom == 0 || (v = (xd - k[knots+start+i+1]) * w[b+i+1] / bottom)
            bottom = k[knots+start+i+p+2] - k[knots+start+i+2]
            bottom == 0 || (v += (k[knots+start+i+p+2] - xd) * w[b+i+2] / bottom)
            w[b+i+1] = v
        end
        base += start * t[str+d]
        b += degree + 1
    end
    @inbounds for j in 1:m
        w[out+j] = zero(eltype(w))
    end
    bspline_sum!(w, t, deg, str, m, c, cb + base, out, n, b - t[deg+n] - 1, one(eltype(w)), 0)
    nothing
end
function bspline_sum!(w, t, deg, str, m, c, cb, out, dim, wb, weight, offset)
    if dim == 1
        @inbounds for i in 0:t[deg+1]
            ww = weight * w[wb+i+1]
            for j in 1:m
                w[out+j] += ww * c[cb+offset+i*m+j]
            end
        end
    else
        lower = wb - t[deg+dim-1] - 1
        @inbounds for i in 0:t[deg+dim]
            bspline_sum!(w, t, deg, str, m, c, cb, out, dim - 1, lower, weight * w[wb+i+1], offset + i * t[str+dim])
        end
    end
    nothing
end""",
    # The contracted Kronecker products of CasADi (`casadi_kron_contract_inner_dense` and
    # `casadi_kron_contract_outer_dense`): `m` is a matrix of `na` by `nb` blocks, a block has
    # `ma` by `mb` elements, and the other operand `b` has the size of a block (inner) or one
    # element per block (outer).
    Symbol("kron_inner!") => """
function kron_inner!(w, m, ma, na, b, mb, nb, y)
    @inbounds for k in 1:ma*na
        w[y+k] = zero(eltype(w))
    end
    @inbounds for j in 0:na-1, s in 0:nb-1, i in 0:ma-1
        column = m + (j * nb + s) * (ma * mb) + i * mb
        acc = zero(eltype(w))
        for r in 1:mb
            acc += w[column+r] * w[b+s*mb+r]
        end
        w[y+j*ma+i+1] += acc
    end
    nothing
end""",
    Symbol("kron_outer!") => """
function kron_outer!(w, m, mb, nb, a, ma, na, y)
    @inbounds for k in 1:mb*nb
        w[y+k] = zero(eltype(w))
    end
    @inbounds for j in 0:na-1, s in 0:nb-1, i in 0:ma-1
        column = m + (j * nb + s) * (ma * mb) + i * mb
        value = w[a+j*ma+i+1]
        for r in 1:mb
            w[y+s*mb+r] += value * w[column+r]
        end
    end
    nothing
end""",
    # The node of CasADi that makes a symmetric matrix positive definite (`casadi_convexify_eval` of
    # CasADi 3.8.1, with the routines `casadi_cvx*` that it calls, statement by statement). The
    # whole numbers `t[base+1:base+10]` are the strategy (0 shift of the diagonal, 1 eigenvalues
    # below the margin are replaced by it, 2 by their size), the stored part (0 all, 1 lower, 2
    # upper triangle), the numbers of nonzeros and of columns, whether the blocks are copied to a
    # dense matrix, the number of blocks, the limit of iterations, the number of nonzeros of an
    # operand that has another pattern (or zero), the length of the list of nonzeros of the
    # blocks, and the size of the largest block. The tables follow: the start of each column,
    # the row of each nonzero, the start of each block, the nonzero of each element of the
    # blocks, and the nonzero of the operand for each nonzero of the result (zero for none).
    # Indices in C are from zero, and `w[a+i+1]` is the element `a[i]` of C.
    Symbol("convexify!") => """
struct ConvexifyTrace end
function cvx_house!(w, v, nv)
    v0 = w[v+1]
    sigma = zero(v0)
    @inbounds for i in 1:nv-1
        sigma += w[v+i+1] * w[v+i+1]
    end
    s = sqrt(v0 * v0 + sigma)
    if sigma == 0
        beta = oftype(v0, 2 * (v0 < 0))
        w[v+1] = 1
    else
        v0 = v0 <= 0 ? v0 - s : -sigma / (v0 + s)
        v02 = v0 * v0
        beta = 2 * v02 / (sigma + v02)
        w[v+1] = 1
        @inbounds for i in 1:nv-1
            w[v+i+1] /= v0
        end
    end
    (s, beta)
end
function cvx_house_apply_symm!(w, n, k, A, p, v, beta)
    stride = k + 1
    A += k + 1 + n * k
    N = n - k - 1
    @inbounds for i in 1:N
        w[p+i] = zero(eltype(w))
    end
    a = A + n
    @inbounds for i in 0:N-1
        w[p+i+1] += beta * w[a+1] * w[v+i+1]
        a += 1
        for j in i+1:N-1
            w[p+j+1] += beta * w[a+1] * w[v+i+1]
            w[p+i+1] += beta * w[a+1] * w[v+j+1]
            a += 1
        end
        a += stride + i + 1
    end
    dot = zero(eltype(w))
    @inbounds for i in 1:N
        dot += w[p+i] * w[v+i]
    end
    alpha = -beta * dot / 2
    @inbounds for i in 1:N
        w[p+i] += alpha * w[v+i]
    end
    a = A + n
    @inbounds for i in 0:N-1
        w[a+1] -= 2 * w[v+i+1] * w[p+i+1]
        a += 1
        for j in i+1:N-1
            w[a+1] -= w[v+i+1] * w[p+j+1] + w[v+j+1] * w[p+i+1]
            a += 1
        end
        a += stride + i + 1
    end
    nothing
end
function cvx_givens(a, b)
    b == 0 && return (one(a), zero(a))
    if abs(b) > abs(a)
        r = -a / b
        s = 1 / sqrt(1 + r * r)
        (s * r, s)
    else
        r = -b / a
        c = 1 / sqrt(1 + r * r)
        (c, c * r)
    end
end
function cvx_implicit_qr!(w, n, td, to, cs)
    @inbounds begin
        d = 0.5 * (w[td+n-1] - w[td+n])
        to2 = w[to+n-1] * w[to+n-1]
        sd = d < 0 ? -1 : 1
        mu = w[td+n] - to2 / (d + sd * sqrt(d * d + to2))
        x = w[td+1] - mu
        z = w[to+1]
        for i in 0:n-2
            c, s = cvx_givens(x, z)
            d0, d1, o0, o1 = w[td+i+1], w[td+i+2], w[to+i+1], w[to+i+2]
            t1 = d0 * c - o0 * s
            t2 = o0 * c - d1 * s
            w[td+i+1] = c * t1 - s * t2
            w[to+i+1] = s * t1 + c * t2
            w[td+i+2] = d0 * s * s + 2 * s * o0 * c + d1 * c * c
            w[to+i+2] *= c
            i > 0 && (w[to+i] = w[to+i] * c - z * s)
            x = w[to+i+1]
            z = -s * o1
            w[cs+1] = c
            w[cs+2] = s
            cs += 2
        end
    end
    nothing
end
function cvx_symm_schur!(w, n, td, to, tol, max_iter, iw, trace)
    meta, trace_offset, q = 1, 0, 0
    iw[1] = 0
    @inbounds while q < n
        iw[1] == max_iter && return true
        for i in 0:n-2
            abs(w[to+i+1]) <= tol * (abs(w[td+i+1]) + abs(w[td+i+2])) && (w[to+i+1] = 0)
        end
        p, q, sp, sq = 0, 0, false, false
        for i in 0:n-2
            w[to+n-i-1] == 0 && !sq ? (q += 1) : (sq = true)
            w[to+i+1] == 0 && !sp ? (p += 1) : (sp = true)
            if q == n - 1
                q, p = n, 0
            end
        end
        nn = n - q - p
        if q < n
            cvx_implicit_qr!(w, nn, td + p, to + p, trace + trace_offset)
            trace_offset += 2 * (nn - 1)
            iw[meta+1], iw[meta+2], iw[meta+3] = nn, p, trace_offset
            meta += 3
            iw[1] += 1
        end
    end
    false
end
function cvx_givens_apply!(w, n, q, c, s, p)
    m = q + p
    @inbounds for _ in 1:p
        a, b = w[m+1], w[m+2]
        w[m+1] = c * a + s * b
        w[m+2] = c * b - s * a
        m += n
    end
    @inbounds begin
        t1 = c * w[m+1] + s * w[m+2]
        t2 = c * w[m+2] + s * w[m+n+2]
        t3 = c * w[m+2] - s * w[m+1]
        t4 = c * w[m+n+2] - s * w[m+2]
        w[m+1] = c * t1 + s * t2
        w[m+2] = c * t2 - s * t1
        w[m+n+2] = c * t4 - s * t3
    end
    m = q + n * p + p + 2
    @inbounds for _ in 1:n-p-2
        a, b = w[m+1], w[m+n+1]
        w[m+1] = c * a + s * b
        w[m+n+1] = c * b - s * a
        m += 1
    end
    nothing
end
function cvx_house_apply!(w, n, m, s, A, p, v, beta)
    a = A
    @inbounds for i in 1:n
        w[p+i] = zero(eltype(w))
        w[p+i] += beta * w[a+1]
        for j in 1:m-1
            w[p+i] += beta * w[a+j+1] * w[v+j+1]
        end
        a += s
    end
    a = A
    @inbounds for i in 1:n
        w[a+1] -= w[p+i]
        for j in 1:m-1
            w[a+j+1] -= w[v+j+1] * w[p+i]
        end
        a += s
    end
    nothing
end
function cvx_scalar(epsilon, reflect, eig)
    e = reflect ? abs(eig) : eig
    e > epsilon ? e : oftype(e, epsilon)
end
function cvx!(w, n, A, epsilon, tol, reflect, max_iter, trace, iw, pp, beta)
    n == 0 && return false
    if n == 1
        w[A+1] = cvx_scalar(epsilon, reflect, w[A+1])
        return false
    end
    @inbounds for k in 0:n-3
        base = A + k + 1 + n * k
        N = n - k - 1
        v = A + N * n
        for i in 1:N
            w[v+i] = w[base+i]
        end
        w[base+1], w[beta+k+1] = cvx_house!(w, v, N)
        cvx_house_apply_symm!(w, n, k, A, pp, v, w[beta+k+1])
    end
    @inbounds for i in 0:n-1, j in 0:n-1
        i - j >= 2 && (w[A+i+j*n+1] = 0)
    end
    td, to = A, A + n
    @inbounds begin
        first_off = w[A+2]
        for i in 1:n-1
            w[td+i+1] = w[A+i+n*i+1]
        end
        w[to+1] = first_off
        for i in 1:n-2
            w[to+i+1] = w[A+i+1+n*i+1]
        end
    end
    cvx_symm_schur!(w, n, td, to, tol, max_iter, iw, trace) && return true
    @inbounds for i in 0:n-1
        w[A+i+n*i+1] = cvx_scalar(epsilon, reflect, w[td+i+1])
    end
    @inbounds for i in 0:n-1, j in i+1:n-1
        w[A+j+i*n+1] = 0
    end
    meta = 3 * (iw[1] - 1) + 1
    @inbounds for _ in 1:iw[1]
        nn, p, cs = iw[meta+1], iw[meta+2], trace + iw[meta+3]
        meta -= 3
        for j in 0:nn-2
            s, c = w[cs], w[cs-1]
            cs -= 2
            cvx_givens_apply!(w, n, A, c, s, p + nn - j - 2)
        end
    end
    @inbounds for k in n-3:-1:0
        N = n - k - 1
        v = A + N * n
        cvx_house_apply_symm!(w, n, k, A, pp, v, w[beta+k+1])
        cvx_house_apply!(w, k + 1, N, n, A + k + 1, pp, v, w[beta+k+1])
    end
    false
end
function convexify!(w, t, base, margin, x, out, scratch)
    @inbounds strategy, kind, nnz, ncol, transform, nblocks, max_iter, nin, nmap, nmax = ntuple(k -> t[base+k], 10)
    colind = base + 10
    row = colind + ncol + 1
    offsets = row + nnz
    mapping = offsets + nblocks + 1
    projection = mapping + nmap
    if nin > 0
        @inbounds for k in 1:nin
            w[scratch+k] = w[x+k]
        end
        @inbounds for k in 1:nnz
            w[out+k] = t[projection+k] == 0 ? zero(eltype(w)) : w[scratch+t[projection+k]]
        end
    elseif x != out
        @inbounds for k in 1:nnz
            w[out+k] = w[x+k]
        end
    end
    if strategy == 0
        # The least eigenvalue is not below the least value of a diagonal element less the sizes of the others of its column.
        bound = zero(eltype(w))
        @inbounds for c in 0:ncol-1
            center, radius = zero(eltype(w)), zero(eltype(w))
            for k in t[colind+c+1]:t[colind+c+2]-1
                t[row+k+1] == c ? (center = w[out+k+1]) : (radius += abs(w[out+k+1]))
            end
            value = center - radius
            bound = c == 0 || isnan(bound) || value < bound ? value : bound
        end
        reg = margin - bound
        if reg > 0
            @inbounds for c in 0:ncol-1, k in t[colind+c+1]:t[colind+c+2]-1
                t[row+k+1] == c && (w[out+k+1] += reg)
            end
        end
        return nothing
    end
    block = scratch + nin
    trace = block + nmax * nmax
    pp = trace + 2 * max(nmax - 1, 0) * max_iter
    beta = pp + nmax
    iw = workspace(ConvexifyTrace, Int, 1 + 3 * max_iter)
    offset = 0
    @inbounds for b in 0:nblocks-1
        # Loop invariant: the blocks before `b` are positive definite in the result, and `offset` elements of the list belong to them.
        n = t[offsets+b+2] - t[offsets+b+1]
        H = transform != 0 ? block : out + offset
        if transform != 0
            kk = 0
            for i in 0:n-1, j in (kind == 0 ? (0:n-1) : kind == 2 ? (0:i) : (i:n-1))
                e = w[out+t[mapping+offset+kk+1]+1]
                kk += 1
                if kind == 0
                    w[H+kk] = e
                else
                    w[H+i*n+j+1] = e
                    w[H+i+n*j+1] = e
                end
            end
        end
        cvx!(w, n, H, margin, 1e-10, strategy == 2, max_iter, trace, iw, pp, beta) &&
            error("a matrix cannot be made positive definite: the eigenvalues of a block did not converge in ", max_iter, " iterations")
        for i in 0:n-1, j in 0:i
            w[H+n*i+j+1] = w[H+n*j+i+1]
        end
        if transform != 0
            kk = 0
            for i in 0:n-1, j in (kind == 0 ? (0:n-1) : kind == 2 ? (0:i) : (i:n-1))
                w[out+t[mapping+offset+kk+1]+1] = kind == 0 ? w[H+kk+1] : w[H+n*i+j+1]
                kk += 1
            end
        end
        offset += kind == 0 ? n * n : n * (n + 1) ÷ 2
    end
    nothing
end""",
    # Write a matrix to the file `base.NNNNNN.mtx` in the Matrix Market format, as the dump node of
    # CasADi: the number counts the files of this name; `rows` and `columns` give the place of each
    # of the nonzeros `values`.
    :write_matrix => """
const DUMP_COUNT = Dict{String,Int}()
const DUMP_LOCK = ReentrantLock()
function write_matrix(base, nrow, ncol, rows, columns, values)
    count = lock(() -> (DUMP_COUNT[base] = get(DUMP_COUNT, base, -1) + 1), DUMP_LOCK)
    open(string(base, ".", lpad(count, 6, "0"), ".mtx"), "w") do io
        println(io, "%%MatrixMarket matrix coordinate real general")
        println(io, nrow, " ", ncol, " ", length(values))
        for k in eachindex(values)
            x = values[k]
            println(io, rows[k], " ", columns[k], " ", x isa Float64 ? Base.Ryu.writeexp(x, 16) : x)
        end
    end
    nothing
end""",
    :workspace => """
struct Workspace{K,T} end
function workspace(::Type{K}, ::Type{T}, n::Int) where {K,T}
    storage = task_local_storage()
    w = get(storage, Workspace{K,T}, nothing)
    w isa Vector{T} && length(w) >= n && return w
    storage[Workspace{K,T}] = Vector{T}(undef, n)
end""",
)

"""
Helpers that each helper calls. A module written by `generate_julia` must hold
them too.
"""
const HELPER_NEEDS = Dict{Symbol,Vector{Symbol}}(Symbol("bspline!") => [:low], Symbol("convexify!") => [:workspace])

# The method of `stored` for a sparse matrix, which gives its stored values after a
# check of its pattern. The full check, O(nnz), runs when the index vectors of the
# matrix (`colptr` and `rowval`) are not those of the last matrix that passed it; else
# the check compares the number of rows and of stored values only, O(1). A matrix
# whose index vectors are changed in place, and keep their length, is not checked
# again. The extension for SparseArrays.jl defines the method for this package, and a
# written module that uses `stored` holds a copy.
const SPARSE_HELPER_SOURCE = """
function stored(a::SparseArrays.SparseMatrixCSC, check::PatternCheck)
    colptr, rows, m, indices = SparseArrays.getcolptr(a), SparseArrays.rowvals(a), size(a, 1), check.indices
    length(rows) == length(indices) || pattern_error()
    seen = @atomic :acquire check.seen
    if !(seen isa Tuple{Int,WeakRef,WeakRef} && seen[1] == m && seen[2].value === colptr && seen[3].value === rows)
        all(indices[k] == (c - 1) * m + rows[k] for c in axes(a, 2) for k in SparseArrays.nzrange(a, c)) || pattern_error()
        @atomic :release check.seen = (m, WeakRef(colptr), WeakRef(rows))
    end
    SparseArrays.nonzeros(a)
end
@noinline pattern_error() = throw(ArgumentError("the pattern of a sparse matrix is not that of the generated function"))
is_stored(::SparseArrays.SparseMatrixCSC) = true"""

"The helper functions of `HELPER_SOURCE`, defined for use by this package."
module Helpers
import SpecialFunctions
import StaticArrays
import StaticArraysCore
import ..HELPER_SOURCE
for source in values(HELPER_SOURCE)
    include_string(@__MODULE__, source)
end
end

# How an operation becomes Julia code: `build(ref, a[, b])` returns the
# expression for operands `a` and `b`, where `ref(owner, name)` gives the
# expression that names a function of `owner` (:NaNMath, :SpecialFunctions or
# :Helpers). Functions of Base are written directly.
base(name) = (ref, operands...) -> Expr(:call, name, operands...)
from(owner, name) = (ref, operands...) -> Expr(:call, ref(owner, name), operands...)
logical(test) = (ref, operands...) -> :(ifelse($(test(operands...)), 1.0, 0.0))

"""
    OPERATIONS

Operation table: CasADi scalar operation name => `(arity, build)`.
`build(ref, a[, b])` returns the expression for the operands `a` and `b`.

Operations that leave their domain give NaN, as in the C library; NaNMath.jl
supplies those functions. Logical operations give 1.0 or 0.0.
"""
const OPERATIONS = Dict{Symbol,@NamedTuple{arity::Int, build::Function}}(
    op => (; arity, build) for (op, arity, build) in (
        (:OP_ASSIGN, 1, base(:identity)),
        (:OP_ADD, 2, base(:+)),
        (:OP_SUB, 2, base(:-)),
        (:OP_MUL, 2, base(:*)),
        (:OP_DIV, 2, base(:/)),
        (:OP_NEG, 1, base(:-)),
        (:OP_EXP, 1, base(:exp)),
        (:OP_LOG, 1, from(:NaNMath, :log)),
        (:OP_POW, 2, from(:NaNMath, :pow)),
        (:OP_CONSTPOW, 2, from(:NaNMath, :pow)),
        (:OP_SQRT, 1, from(:NaNMath, :sqrt)),
        (:OP_SQ, 1, (ref, a) -> :($a * $a)),
        (:OP_TWICE, 1, (ref, a) -> :(2 * $a)),
        (:OP_SIN, 1, from(:NaNMath, :sin)),
        (:OP_COS, 1, from(:NaNMath, :cos)),
        (:OP_TAN, 1, from(:NaNMath, :tan)),
        (:OP_ASIN, 1, from(:NaNMath, :asin)),
        (:OP_ACOS, 1, from(:NaNMath, :acos)),
        (:OP_ATAN, 1, base(:atan)),
        (:OP_LT, 2, logical((a, b) -> :($a < $b))),
        (:OP_LE, 2, logical((a, b) -> :($a <= $b))),
        (:OP_EQ, 2, logical((a, b) -> :($a == $b))),
        (:OP_NE, 2, logical((a, b) -> :($a != $b))),
        (:OP_NOT, 1, logical(a -> :($a == 0))),
        (:OP_AND, 2, logical((a, b) -> :(($a != 0) & ($b != 0)))),
        (:OP_OR, 2, logical((a, b) -> :(($a != 0) | ($b != 0)))),
        (:OP_FLOOR, 1, base(:floor)),
        (:OP_CEIL, 1, base(:ceil)),
        (:OP_FMOD, 2, base(:rem)),
        (:OP_FABS, 1, base(:abs)),
        (:OP_SIGN, 1, base(:sign)),
        (:OP_COPYSIGN, 2, base(:copysign)),
        (:OP_IF_ELSE_ZERO, 2, (ref, a, b) -> :(ifelse($a != 0, $b, zero($b)))),
        (:OP_ERF, 1, from(:SpecialFunctions, :erf)),
        (:OP_FMIN, 2, from(:Helpers, :fmin)),
        (:OP_FMAX, 2, from(:Helpers, :fmax)),
        (:OP_INV, 1, base(:inv)),
        (:OP_SINH, 1, base(:sinh)),
        (:OP_COSH, 1, base(:cosh)),
        (:OP_TANH, 1, base(:tanh)),
        (:OP_ASINH, 1, base(:asinh)),
        (:OP_ACOSH, 1, from(:NaNMath, :acosh)),
        (:OP_ATANH, 1, from(:NaNMath, :atanh)),
        (:OP_ATAN2, 2, base(:atan)),
        (:OP_ERFINV, 1, from(:Helpers, :erfinv)),
        (:OP_LOG1P, 1, from(:NaNMath, :log1p)),
        (:OP_EXPM1, 1, base(:expm1)),
        (:OP_HYPOT, 2, base(:hypot)),
        (:OP_REMAINDER, 2, (ref, a, b) -> :(rem($a, $b, RoundNearest))),
        (:OP_PRINTME, 2, from(:Helpers, :printme)),
    ))

const OWNERS = (; NaNMath, SpecialFunctions, LinearAlgebra, StaticArrays, StaticArraysCore, Helpers)

"Reference to a function by module object: valid in any module, needs CasADiCodegen.jl loaded."
global_ref(owner::Symbol, name::Symbol) = GlobalRef(getfield(OWNERS, owner), name)

"""
    throwing(ref) -> ref

Return the reference maker for `nanmath = false`. The functions of NaNMath
become the functions of Base with the same meaning, which throw `DomainError`
outside their domain. The guarded `erfinv` becomes `SpecialFunctions.erfinv`.
"""
throwing(ref) = function (owner::Symbol, name::Symbol)
    owner === :NaNMath && return name === :pow ? :^ : name
    (owner, name) == (:Helpers, :erfinv) && return ref(:SpecialFunctions, :erfinv)
    ref(owner, name)
end
