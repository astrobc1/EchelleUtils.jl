using DataInterpolations

"""
    cspline_interp(x, y, xnew)
Wrapper to interpolate a vector `y` sampled on `x` to `xnew` using cubic spline interpolation. Values are not extrapolated.
"""
function cspline_interp(x, y, xnew)
    good = findall(@. isfinite(x) && isfinite(y))
    if length(good) == 0
        return Float64[]
    end
    cspline = @views CubicSpline(y[good], x[good])
    bad = findall(xnew .< x[good[1]] .&& xnew .> x[good[end]])
    ynew = cspline.(xnew)
    ynew[bad] .= NaN
    return ynew
end

"""
    lin_interp(x, y, xnew)
Wrapper to interpolate a vector `y` sampled on `x` to `xnew` using linear interpolation. Values are not extrapolated.
"""
function lin_interp(x, y, xnew)
    good = findall(isfinite.(x) .&& isfinite.(y))
    _interpolator = LinearInterpolation(y[good], x[good])
    good = findall(xnew .>= x[good[1]] .&& xnew .<= x[good[end]])
    ynew = fill(NaN, length(xnew))
    for i ∈ good
        ynew[i] = _interpolator(xnew[i])
    end
    return ynew
end