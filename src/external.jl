# A function of a compiled library (`external` of CasADi) is called directly
# from generated code, with the interface of the C code that CasADi generates,
# so that no call goes through CasADi.

import Libdl

"The work space of one task for an external function: the pointers to the arguments and to the results, the work vectors of the function, and the results that it returns."
struct ExternalWork{R<:Tuple}
    arg::Vector{Ptr{Float64}}
    res::Vector{Ptr{Float64}}
    iw::Vector{Clonglong}
    w::Vector{Float64}
    results::Vector{Vector{Float64}}
    returned::R
end

"""
    ExternalCall(name, library, nnz_in, nnz_out)

The function `name` of the compiled library `library`, as the Julia function
that generated code calls. The library function has the interface of the C
code that CasADi generates.

A call takes one vector of nonzeros per input. It returns a tuple with one
vector of nonzeros per result that has nonzeros. The returned vectors belong
to the task, and the next call overwrites them. A call allocates no memory
after the first one of a task.

The library gives the sizes of the work vectors with `name_work`, and a memory
object with `name_checkout` and `name_release`; each of the three can be
absent.

The arguments must be contiguous vectors of `Float64`, because the function is
compiled code. Only the element type is checked.
"""
mutable struct ExternalCall
    const name::String
    const library::String
    const nnz_in::Vector{Int}
    const nnz_out::Vector{Int}
    const entry::Ptr{Cvoid}
    const checkout::Ptr{Cvoid}
    const release::Ptr{Cvoid}
    const sizes::NTuple{4,Int}                  # of the pointers to arguments and to results, and of the two work vectors
    function ExternalCall(name::AbstractString, library::AbstractString, nnz_in, nnz_out)
        handle = Libdl.dlopen(library)
        entry = Libdl.dlsym(handle, name)
        optional(suffix) = something(Libdl.dlsym(handle, name * suffix; throw_error = false), C_NULL)
        sizes = [Ref{Clonglong}(0) for _ in 1:4]
        work = optional("_work")
        work == C_NULL || ccall(work, Cint, (Ref{Clonglong}, Ref{Clonglong}, Ref{Clonglong}, Ref{Clonglong}), sizes[1], sizes[2], sizes[3], sizes[4])
        counted = optional("_incref")
        counted == C_NULL || ccall(counted, Cvoid, ())
        new(String(name), String(library), collect(Int, nnz_in), collect(Int, nnz_out), entry, optional("_checkout"), optional("_release"),
            (max(sizes[1][], length(nnz_in)), max(sizes[2][], length(nnz_out)), sizes[3][], sizes[4][]))
    end
end

function external_work(e::ExternalCall)
    get!(task_local_storage(), e) do
        results = [Vector{Float64}(undef, n) for n in e.nnz_out]
        ExternalWork(fill(Ptr{Float64}(C_NULL), e.sizes[1]), fill(Ptr{Float64}(C_NULL), e.sizes[2]),
                     zeros(Clonglong, e.sizes[3]), zeros(e.sizes[4]), results, Tuple(r for r in results if !isempty(r)))
    end
end

function (e::ExternalCall)(arguments::Vararg{Any,N}) where {N}
    for a in arguments
        @requires eltype(a) === Float64 "\"$(e.name)\" is compiled code, which takes Float64 only; got $(eltype(a))"
    end
    s = external_work(e)::ExternalWork
    flag = GC.@preserve arguments s begin
        for (i, a) in enumerate(arguments)
            s.arg[i] = isempty(a) ? C_NULL : pointer(a)
        end
        for (o, r) in enumerate(s.results)
            s.res[o] = isempty(r) ? C_NULL : pointer(r)
        end
        memory = e.checkout == C_NULL ? Cint(0) : ccall(e.checkout, Cint, ())
        flag = ccall(e.entry, Cint, (Ptr{Ptr{Float64}}, Ptr{Ptr{Float64}}, Ptr{Clonglong}, Ptr{Float64}, Cint), s.arg, s.res, s.iw, s.w, memory)
        e.release == C_NULL || ccall(e.release, Cvoid, (Cint,), memory)
        flag
    end
    flag == 0 || error("the external function \"$(e.name)\" failed with the code $flag")
    s.returned
end

"""
    external_library(f) -> String

Return the path of the compiled library of the external function `f`. The
CasADi interface does not expose it, so it is read from the serial form of
`f`: the path is the text in it that names a file.

Throws `ArgumentError` when no such file exists. CasADi can also compile a
library in memory, which has no path.
"""
function external_library(f::casadi.Function)
    code = codeunits(casadi.serialize(f))
    for shift in 0:1
        bytes = UInt8[(code[k+1] - 0x61) << 4 | (code[k] - 0x61) & 0x0f for k in 1+shift:2:length(code)-1]
        printable = [0x20 <= b <= 0x7e for b in bytes]
        for (start, _, _, len) in runs(1:length(bytes), printable), stop in start+len-1:-1:start+1
            # Loop invariant: no longer text from this start names a file.
            path = String(bytes[start:stop])
            isfile(path) && return abspath(path)
        end
    end
    throw(ArgumentError("the library of the external function \"$(casadi.name(f))\" is not a file that this package can find"))
end

function external_call(f::casadi.Function)
    nnz_in, nnz_out = [casadi.nnz_in(f, i) for i in 0:casadi.n_in(f)-1], [casadi.nnz_out(f, o) for o in 0:casadi.n_out(f)-1]
    call = ExternalCall(casadi.name(f), external_library(f), nnz_in, nnz_out)
    JuliaCall(casadi.name(f), call, length(nnz_in), Tuple{Vararg{Int}}[(n,) for n in nnz_in], Tuple{Vararg{Int}}[(n,) for n in nnz_out], nnz_in, nnz_out)
end
