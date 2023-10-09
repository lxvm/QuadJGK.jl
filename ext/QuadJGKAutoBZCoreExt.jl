module QuadJGKAutoBZCoreExt

using LinearAlgebra: norm
import AutoBZCore: IntegralAlgorithm, IntegralSolution, init_cacheval, do_solve
import QuadJGK: AutoJGK, quadjgk

struct QuadJGKJL{N} <: IntegralAlgorithm
    order::Int
    norm::N
end

AutoJGK(; order = 7, norm = norm) = QuadJGKJL(order, norm)

init_cacheval(f, dom, p, alg::QuadJGKJL) = nothing

function do_solve(f::F, dom, p, alg::QuadJGKJL, cacheval; reltol=nothing, abstol=nothing, maxiters=10^7) where {F}
    val, err = quadjgk(x -> f(x, p), dom, maxevals=maxiters, rtol=reltol, atol=abstol, order=alg.order, norm=alg.norm, segbuf=cacheval)
    return IntegralSolution(val, err, true, -1)
end

end
