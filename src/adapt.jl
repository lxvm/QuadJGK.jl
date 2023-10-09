# Internal routine: integrate f over the union of the open intervals
# (s[1],s[2]), (s[2],s[3]), ..., (s[end-1],s[end]), using h-adaptive
# integration with the order-n Kronrod rule and weights of type Tw,
# with absolute tolerance atol and relative tolerance rtol,
# with maxevals an approximate maximum number of f evaluations.
function do_quadjgk(f::F, s::NTuple{N,T}, n, atol, rtol, maxevals, nrm, segbuf) where {T,N,F}
    @assert N ≥ 1
    segs = ntuple(Val{N}()) do i
        check_endpoint_bounds(s[i], n, throw_error=true)
        evalrule(f, s[i], n, nrm)
    end
    I = sum(s -> s.I, segs)
    E = sum(s -> s.E, segs)
    numevals = (2n+1) * N

    # logic here is mainly to handle dimensionful quantities: we
    # don't know the correct type of atol115, in particular, until
    # this point where we have the type of E from f.  Also, follow
    # Base.isapprox in that if atol≠0 is supplied by the user, rtol
    # defaults to zero.
    atol_ = something(atol, zero(E))
    rtol_ = something(rtol, iszero(atol_) ? sqrt(eps(one(atol_))) : zero(one(atol_)))

    # optimize common case of no subdivision
    if E ≤ atol_ || E ≤ rtol_ * nrm(I) || numevals ≥ maxevals
        return (I, E) # fast return when no subdivisions required
    end

    segheap = segbuf === nothing ? collect(segs) : (resize!(segbuf, N-1) .= segs)
    heapify!(segheap, Reverse)
    return resum(f, adapt(f, segheap, I, E, numevals, n, atol_, rtol_, maxevals, nrm))
end

# internal routine to perform the h-adaptive refinement of the integration segments (segs)
function adapt(f::F, segs::Vector{T}, I, E, numevals, n, atol, rtol, maxevals, nrm) where {F, T}
    # Pop the biggest-error segment and subdivide (h-adaptation)
    # until convergence is achieved or maxevals is exceeded.
    while E > atol && E > rtol * nrm(I) && numevals < maxevals
        next = refine(f, segs, I, E, numevals, n, atol, rtol, maxevals, nrm)
        next isa Vector && return next # handle type-unstable functions
        I, E, numevals = next
    end
    return segs
end

# internal routine to refine the segment with largest error
function refine(f::F, segs::Vector{T}, I, E, numevals, n, atol, rtol, maxevals, nrm) where {F, T}
    s = heappop!(segs, Reverse)

    a, b = s.space.domain.a, s.space.domain.b
    mid = (a + b) / 2
    sp1 = JacobiSpace(s.space.α, oftype(s.space.α, 0) , OpenInterval(a, mid))
    sp2 = JacobiSpace(oftype(s.space.β, 0), s.space.β , OpenInterval(mid, b))

    # early return if integrand evaluated at endpoints
    if check_endpoint_bounds(sp1, n) || check_endpoint_bounds(sp2, n)
        heappush!(segs, s, Reverse)
        return segs
    end

    s1 = evalrule(f, sp1, n, nrm)
    s2 = evalrule(f, sp2, n, nrm)

    I = (I - s.I) + s1.I + s2.I
    E = (E - s.E) + s1.E + s2.E
    numevals += 4n+2

    # handle type-unstable functions by converting to a wider type if needed
    Tj = promote_type(typeof(s1), promote_type(typeof(s2), T))
    if Tj !== T
        return adapt(f, heappush!(heappush!(Vector{Tj}(segs), s1, Reverse), s2, Reverse),
                     I, E, numevals, n, atol, rtol, maxevals, nrm)
    end

    heappush!(segs, s1, Reverse)
    heappush!(segs, s2, Reverse)

    return I, E, numevals
end

# re-sum (paranoia about accumulated roundoff)
function resum(f, segs)
    I = segs[1].I
    E = segs[1].E
    for i in 2:length(segs)
        I += segs[i].I
        E += segs[i].E
    end
    return (I, E)
end

function check_endpoint_bounds(sp, n; throw_error::Bool=false)
    x, = cachedrule(sp, n)
    a, b = sp.domain.a, sp.domain.b
    c = convert(eltype(x), 0.5) * (b-a)
    xa = a + (1+x[begin])*c
    xb = a + (1+x[end])*c
    (eval_at_a = a >= xa) && throw_error && throw_endpoint_error(xa, a, b)
    (eval_at_b = b <= xb) && throw_error && throw_endpoint_error(xb, a, b)
    eval_at_a || eval_at_b
end
function throw_endpoint_error(x, a, b)
    throw(DomainError(x, "kronrod node detected outside of the interval ($a, $b)"))
end

"""
    quadjgk(f, s::JacobiSpace...; order=7, norm=norm, maxevals=10^7, atol=nothing, rtol=nothing, segbuf=nothing)

Carefully [read](https://doi.org/10.1090/S0025-5718-1988-0942152-3) to determine what orders
produce kronrod rules with nodes in the standard interval.
"""
function quadjgk(f, s; order=7, maxevals=10^7, atol=nothing, rtol=nothing, norm=norm, segbuf=nothing)
    return do_quadjgk(f, s, order, atol, rtol, maxevals, norm, segbuf)
end
quadjgk(f, s::JacobiSpace...; kws...) = quadjgk(f, s; kws...)
