"""
    cross_correlate_interp(x1, y1, x2, y2, lags; interp=:linear, loss=:lsqr)
Method to perform a cross correlation by shifting the vector `y2` by each lag in `lags` (additively). The ccf curve is either the RMS if `loss=:lsqr` or a standard CCF if `loss=:xc`. The shifted data is interpolated onto `x1` using `interp` method (linear or cspline).
"""
function cross_correlate_interp(x1, y1, x2, y2, lags; interp=:linear, loss=:lsqr)
    n1 = length(x1)
    nlags = length(lags)
    ccf = fill(NaN, nlags)
    y2_shifted = fill(NaN, n1)
    weights = ones(n1)
    if interp == :linear
        _interp = lin_interp
    elseif interp == :cspline
        _interp = cspline_interp
    else
        error("interp must be either :linear or :cspline")
    end
    for i=1:nlags
        y2_shifted .= _interp(x2 .+ lags[i], y2, x1)
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