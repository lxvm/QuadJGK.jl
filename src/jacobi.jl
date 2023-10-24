abstract type AbstractDomain{T} end
Base.eltype(::Type{AbstractDomain{T}}) where {T} = T

# open intervals will not evaluate the integrand at the endpoints
struct OpenInterval{T} <: AbstractDomain{T}
    a::T
    b::T
end

# polynomials orthogonal wrt wαβ(x) = (1-x)^α * (1+x)^β
# assuming α > -1 & β > -1
struct JacobiSpace{D,T<:Real}
    α::T
    β::T
    domain::OpenInterval{D}
end

JacobiSpace(α, β) = JacobiSpace(promote(α, β)..., OpenInterval(-1, 1))

function unitintegral(sp::JacobiSpace)
    α, β = sp.α, sp.β
    return _₂F₁(1, -α, β+2, -1)/(β+1) + _₂F₁(1, -β, α+2, -1)/(α+1)
end

# infinite iteration for the standard Jacobi polynomial on [-1,1]
function abciterate(sp::JacobiSpace{D,T}) where {D,T}
    α, β = sp.α, sp.β
    n = 1
    if T <: AbstractFloat
        a₁ = (2+α+β) / 2
        b₁ = (x = (α^2 - β^2)) == zero(T) ? x : x / (2*(α+β))
        c₁ = (y = (2*α*β)) == zero(T) ? y : y / (2*(α+β))
    else
        # den = 1 // (2n*(n+α+β)*(2n+α+β-2)) # 1 // (2*(1+α+β)*(α+β))
        a₁ = (2+α+β) // 2 # (2n+α+β-1)*(2n+α+β)*(2n+α+β-2) * den # (α+β) * (1+α+β) * (2+α+β) * den
        b₁ = (α^2 - β^2) // (2*(α+β)) # (2n+α+β-1)*(α^2 - β^2) * den # (1+α+β)*(α^2 - β^2) * den
        c₁ = (2*α*β) // (2*(α+β)) # 2*(n+α-1)*(n+β-1)*(2n+α+β) * den # 2*α*β*(1+α+β) * den
    end
    return promote(a₁, b₁, c₁), n+1
end

function abciterate(sp::JacobiSpace{D,T}, n) where {D,T}
    α, β = sp.α, sp.β
    den = T <: AbstractFloat ? 1 / (2n*(n+α+β)*(2n+α+β-2)) : 1 // (2n*(n+α+β)*(2n+α+β-2))
    aₙ = (2n+α+β-1)*(2n+α+β)*(2n+α+β-2) * den
    bₙ = (2n+α+β-1)*(α^2 - β^2) * den
    cₙ = 2*(n+α-1)*(n+β-1)*(2n+α+β) * den
    return promote(aₙ, bₙ, cₙ), n+1
end

# an infinite iterator returning tuples (a, b, c) as defined by
# p_{k+1}(x) = (a_k x + b_k) p_k(x) - c_k p_{k-1}(x)
struct ABCRecurrence{S<:JacobiSpace}
    space::S
end

Base.IteratorSize(::Type{<:ABCRecurrence}) = Base.IsInfinite()
Base.eltype(::Type{ABCRecurrence{S}}) where {S} = NTuple{3}

Base.iterate(r::ABCRecurrence) = abciterate(r.space)
Base.iterate(r::ABCRecurrence, state) = abciterate(r.space, state)


# using the definition of α, β in the QuadGK documentation
struct αβRecurrence{R<:ABCRecurrence}
    r::R
end

αβRecurrence(sp::JacobiSpace) = αβRecurrence(ABCRecurrence(sp))

Base.IteratorSize(::Type{<:αβRecurrence}) = Base.IsInfinite()
Base.eltype(::Type{αβRecurrence{R}}) where {R} = NTuple{2}


function Base.iterate(r::αβRecurrence)
    (a₁, b₁, c₁), state = iterate(r.r)
    (a₂, b₂, c₂), state = iterate(r.r, state)
    ia₁ = inv(a₁)
    α₁ = -b₁*ia₁
    ia₂ = inv(a₂)
    β₁ = c₂*ia₁*ia₂
    return promote(α₁, β₁), (ia₂, b₂, state)
end
function Base.iterate(r::αβRecurrence, (iaᵢ, bᵢ, state))
    (aᵢ₊₁, bᵢ₊₁, cᵢ₊₁), state = iterate(r.r, state)
    αᵢ = -bᵢ*iaᵢ
    iaᵢ₊₁ = inv(aᵢ₊₁)
    βᵢ = cᵢ₊₁*iaᵢ*iaᵢ₊₁
    return promote(αᵢ, βᵢ), (iaᵢ₊₁, bᵢ₊₁, state)
end

mutable struct JacobiRecurrence{TF<:AbstractFloat}
    const dv::Vector{TF}
    const ev::Vector{TF}
    const itr::αβRecurrence{ABCRecurrence{JacobiSpace{TF,TF}}}
    β::TF
    state::Tuple{TF,TF,Int}
    n::Int
end

function jacobirecurrence!(r::JacobiRecurrence, n::Integer)
    (m = r.n) > n && return r
    resize!(r.dv, n)
    resize!(r.ev, n-1)
    state = r.state
    next = iterate(r.itr, state)
    β = r.β
    while n > m
        r.ev[m] = sqrt(β)
        (α, β), state = next
        r.dv[m+=1] = α
        next = iterate(r.itr, state)
    end
    r.β = β
    r.state = state
    r.n = n
    return r
end

function jacobirecurrence(r::αβRecurrence, n::Integer)
    (α, β), state = iterate(r)
    dv = Vector{typeof(α)}(undef, n)
    ev = Vector{typeof(β)}(undef, n-1)
    dv[1] = α
    rec = JacobiRecurrence(dv, ev, r, β, state, 1)
    return jacobirecurrence!(rec, n)
end

function jacobirecurrence(r::ABCRecurrence, n::Integer)
    return jacobirecurrence(αβRecurrence(r), n)
end

function jacobirecurrence(sp::JacobiSpace, n::Integer)
    return jacobirecurrence(ABCRecurrence(sp), n)
end

# cache of the recurrence in the jacobi matrices of the jacobi spaces
const jacobicache = Dict{Type,Dict}()

@generated function _cachedjacobi(::Type{TF}, sp::JacobiSpace{TF,TF}, n::Int) where {TF<:AbstractFloat}
    cache = haskey(jacobicache, TF) ? jacobicache[TF] : (jacobicache[TF] = Dict{JacobiSpace{TF,TF},JacobiRecurrence{TF}}())
    :(haskey($cache, (sp,n)) ? jacobirecurrence!($cache[sp], n) : ($cache[sp] = jacobirecurrence(sp, n)))
end

function cachedjacobi(sp::JacobiSpace{D}, n::Integer) where {D}
    x = float(real(one(D))) # domain point of (unitless) canonical type
    csp = JacobiSpace(oftype(x, sp.α), oftype(x, sp.β), OpenInterval(-x, x))  # jacobi space on canonical interval
    r = _cachedjacobi(typeof(x), csp, Int(n))
    return SymTridiagonal(r.dv, r.ev)
end

function jacobigauss(sp::JacobiSpace, n::Integer)
    J = cachedjacobi(sp, n)
    return gauss(J, unitintegral(sp))
end

function jacobikronrod(sp::JacobiSpace, n::Integer)
    J = cachedjacobi(sp, (3n+3)÷2)
    return kronrod(J, n, unitintegral(sp))
end

# cache of the kronrod rules of the jacobi spaces
const rulecache = Dict{Type,Dict}()

@generated function _cachedrule(::Type{TF}, sp::JacobiSpace, n::Int) where {TF}
    cache = haskey(rulecache, TF) ? rulecache[TF] : (rulecache[TF] = Dict{Tuple{JacobiSpace,Int},NTuple{3,Vector{TF}}}())
    :(haskey($cache, (sp,n)) ? $cache[(sp,n)] : ($cache[(sp,n)] = jacobikronrod(sp, n)))
end

function cachedrule(sp::JacobiSpace{D}, n::Integer) where {D}
    x = float(real(one(D))) # domain point of (unitless) canonical type
    csp = JacobiSpace(oftype(x, sp.α), oftype(x, sp.β), OpenInterval(-x, x))  # jacobi space on canonical interval
    return _cachedrule(typeof(x), csp, Int(n))
end
