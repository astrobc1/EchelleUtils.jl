using DataInterpolations

function cspline_interp(x, y, xnew)
    good = findall(isfinite.(x) .&& isfinite.(y))
    if length(good) == 0
        return Float64[]
    end
    cspline = @views CubicSpline(y[good], x[good])
    bad = findall(xnew .< x[good[1]] .&& xnew .> x[good[end]])
    ynew = cspline.(xnew)
    ynew[bad] .= NaN
    return ynew
end


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


function cross_correlate_interp(x1, y1, x2, y2, lags; interp=:linear, loss=:lsqr)
    n1 = length(x1)
    nlags = length(lags)
    ccf = fill(NaN, nlags)
    y2_shifted = fill(NaN, n1)
    weights = ones(n1)
    for i=1:nlags
        if interp == :linear
            y2_shifted .= lin_interp(x2 .+ lags[i], y2, x1)
        else
            y2_shifted .= cspline_interp(x2 .+ lags[i], y2, x1)
        end
        good = findall(isfinite.(y1) .&& isfinite.(y2_shifted))
        if length(good) < 3
            continue
        end
        weights .= 1
        bad = findall(.~isfinite.(y1) .|| .~isfinite.(y2_shifted))
        weights[bad] .= 0
        if loss == :lsqr
            ccf[i] = sqrt(nansum(weights .* (y1 .- y2_shifted).^2) / nansum(weights))
        else
            ccf[i] = nansum(y1 .* y2_shifted .* weights) / nansum(weights)
        end
    end

    return ccf

end