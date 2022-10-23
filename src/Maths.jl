module Maths

using StatsBase
using LinearAlgebra
using NaNStatistics
using Polynomials

const SPEED_OF_LIGHT_MPS = 299792458.0
const TWO_SQRT_2LOG2 = 2 * sqrt(2 * log(2))

function nanargmaximum(x::AbstractArray)
    k = 1
    for i=1:length(x)
        if x[i] > x[k] && isfinite(x[i])
            k = i
        end
    end
    return k
end

function nanargminimum(x::AbstractArray)
    k = 1
    for i=1:length(x)
        if x[i] < x[k] && isfinite(x[i])
            k = i
        end
    end
    return k
end

function round_half_down(x)
    return ceil(x - 0.5)
end

include("chebyshev.jl")
include("stats.jl")
include("filters.jl")
include("interp.jl")

end