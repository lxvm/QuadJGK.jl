# integration segment (a,b), estimated integral I, and estimated error E
struct Segment{T,TX,TI,TE}
    space::JacobiSpace{TX,T}
    I::TI
    E::TE
end
Base.@pure Base.promote_rule(::Type{Segment{T,TX,TI,TE}}, ::Type{Segment{T′,TX′,TI′,TE′}}) where {T,TX,TI,TE,T′,TX′,TI′,TE′} =
    Segment{promote_type(T,T′), promote_type(TX,TX′), promote_type(TI,TI′), promote_type(TE,TE′)}
Base.convert(::Type{T}, s::Segment) where {T<:Segment} = T(s.space,s.I,s.E)
Base.isless(i::Segment, j::Segment) = isless(i.E, j.E)

# Internal routine: approximately integrate f(x) over the interval (a,b)
# by evaluating the integration rule (x,w,gw). Return a Segment.
function evalrule(f::F, sp::JacobiSpace, n, nrm) where {F}
    x, w, gw = cachedrule(sp, n)
    α, β = sp.α, sp.β
    a, b = sp.domain.a, sp.domain.b
    # Ik and Ig are integrals via Kronrod and Gauss rules, respectively
    s = convert(eltype(x), 0.5) * (b-a)
    n1 = 1 - ((n+1) & 1) # 0 if even order, 1 if odd order
    n2 = (n+1) ÷ 2
    wαβ(x) = (1-x)^-α * (1+x)^-β

    if n1 == 0 # even: Gauss rule does not include x == 0
        xk0 = a + (1+x[n+1])*s
        Ik = f(xk0) * wαβ(x[n+1]) * w[n+1]
        Ig = zero(Ik)
    else # odd: don't count x==0 twice in Gauss rule
        xk0 = a + (1+x[n+1])*s
        f0 = f(xk0) * wαβ(x[n+1])
        Ig = f0 * gw[n2]
        xg1 = a + (1+x[n])*s
        xg2 = a + (1+x[n+2])*s
        Ik = f0 * w[n+1] + (f(xg1) * wαβ(x[n]) * w[n] + f(xg2) * wαβ(x[n+2]) * w[n+2])
    end
    for i = 1:(n2-n1)
        xg1 = a + (1+x[2i])*s
        xg2 = a + (1+x[2n-2i+2])*s
        fg1 = f(xg1) * wαβ(x[2i])
        fg2 = f(xg2) * wαβ(x[2n-2i+2])
        xk1 = a + (1+x[2i-1])*s
        xk2 = a + (1+x[2n-2i+3])*s
        fk1 = f(xk1) * wαβ(x[2i-1])
        fk2 = f(xk2) * wαβ(x[2n-2i+3])
        Ig += fg1 * gw[i] + fg2 * gw[n+1-i]
        Ik += (fg1 * w[2i] + fg2 * w[2n-2i+2]) + (fk1 * w[2i-1] + fk2 * w[2n-2i+3])
    end
    Ik_s, Ig_s = Ik * s, Ig * s # new variable since this may change the type
    E = nrm(Ik_s - Ig_s)
    if isnan(E) || isinf(E)
        throw(DomainError(a+s, "integrand produced $E in the interval ($a, $b)"))
    end
    return Segment(JacobiSpace(α, β, OpenInterval(oftype(s, a), oftype(s, b))), Ik_s, E)
end
