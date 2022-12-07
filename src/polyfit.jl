using Polynomials
#using EchelleUtils

function polyfit1d(x, y, w=nothing; deg, max_iterations=5, nσ=5)
    if isnothing(w)
        w = ones(size(x))
    else
        w = copy(w)
    end
    pfit = nothing
    for i=1:max_iterations

        # Fit
        good = findall(@. isfinite(x) && isfinite(y) && isfinite(w) && (w > 0))
        pfit = Polynomials.Polynomial(Polynomials.fit(ArnoldiFit, x[good], y[good], deg, weights=w[good]))

        # Flag
        res = pfit.(x) .- y
        σ = robust_σ(res; w)
        bad = findall(@. ~isfinite(x) || ~isfinite(res) || (abs(res) > nσ * σ))
        if length(bad) == 0
            break
        end
        w[bad] .= 0

    end
    return pfit
end