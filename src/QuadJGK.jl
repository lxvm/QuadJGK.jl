module QuadJGK

using QuadGK: kronrod
using LinearAlgebra: SymTridiagonal, norm
using HypergeometricFunctions: _₂F₁
using DataStructures: heapify!, heappop!, heappush!
import Base.Order.Reverse

export OpenInterval, JacobiSpace, quadjgk, quadjgk_count

export AutoJGK

function AutoJGK end

include("jacobi.jl")
include("evalrule.jl")
include("adapt.jl")

end # module QuadJGK
