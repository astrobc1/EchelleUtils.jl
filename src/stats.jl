
"""
    gauss(x, a, μ, σ)
Construct a Guassian function.
"""
function gauss(x, a, μ, σ)
    return @. a * exp(-0.5 * ((x - μ) / σ)^2)
end

function gauss2d(x, y, amp, μx, μy, σx, σy, θ=0)
    y = amp * exp(-(((x - μx) * cos(θ)+ (y - μy) * sin(θ)) / σx)^2-((-(x - μx) * sin(θ) + (y - μy) * cos(θ)) / σy)^2)
    return y
end

"""
    rmsloss(residuals::AbstractArray{<:Real}, [weights::AbstractArray{<:Real}=nothing]; mask_worst::Int=0, mask_edges::Int=0)
Computes the root mean squared error (RMS) loss. Weights can also be provided, otherwise uniform weights will be used.
"""
function rmsloss(residuals, weights=nothing; mask_worst::Int=0, mask_edges::Int=0)

    # Get good data
    if isnothing(weights)
        weights = ones(length(residuals))
    end

    good = findall(@. isfinite(residuals) && isfinite(weights) && (weights > 0))
    residuals, weights = residuals[good], weights[good]
    wres2 = weights .* residuals.^2
    
    # Remove edges
    if mask_edges > 0
        wres2[1:mask_edges] .= NaN
        wres2[end-mask_edges+1:end] .= NaN
        residuals[1:mask_edges] .= NaN
        residuals[end-mask_edges+1:end] .= NaN
        weights[1:mask_edges] .= 0
        weights[end-mask_edges+1:end] .= 0
    end

    # Ignore worst N pixels
    if mask_worst > 0
        ss = sortperm(abs.(wres2))
        wres2[ss[end-mask_worst+1:end]] .= NaN
        residuals[ss[end-mask_worst+1:end]] .= NaN
        weights[ss[end-mask_worst+1:end]] .= 0
    end

        
    # Compute rms
    rms = sqrt(nansum(weights .* residuals.^2) / nansum(weights))

    # Return
    return rms
end

"""
    redχ2loss(residuals::AbstractArray{<:Real}, [errors::AbstractArray{<:Real}=nothing]; mask_worst::Int=0, mask_edges::Int=0)
Computes the reduced chi square loss.
"""
function redχ2loss(residuals, errors; mask_worst=0, mask_edges=0, n_pars_opt)

    # Compute diffs2
    good = findall(@. isfinite(residuals) && isfinite(errors))
    norm_res2 = @views (residuals[good] ./ errors[good]).^2

    # Remove edges
    if mask_edges > 0
        norm_res2[1:mask_edges-1] .= NaN
        norm_res2[end-mask_edges+1:end] .= NaN
    end
    
    # Ignore worst N pixels
    if mask_worst > 0
        ss = sortperm(norm_res2)
        norm_res2[ss[end-mask_worst+1:end]] .= NaN
    end

    # Degrees of freedom
    good = findall(isfinite.(norm_res2))
    n_good = length(good)
    ν = n_good - n_pars_opt - 2 * mask_edges - mask_worst

    @assert ν > 0

    # Compute chi2
    redχ² = nansum(norm_res2) / ν

    # Return
    return redχ²
end


"""
    robust_σ(x::AbstractArray; [w::AbstractArray] nσ::Real=4)
Computes a robust standard deviation value by flagging values through the median absolute deviation. 
"""
function robust_σ(x::AbstractArray; w::Union{Nothing, AbstractArray}=nothing, nσ::Real=4)
    if isnothing(w)
        w = get_mask(x)
    end
    med = weighted_quantile(x, w=w)
    adevs = abs.(med .- x)
    mad = weighted_quantile(adevs, w=w)
    good = findall(adevs .< 1.4826 * mad * nσ)
    if length(good) > 1
        return @views weighted_stddev(x[good], w[good])
    else
        return NaN
    end
end

"""
    robust_σ(x::AbstractArray; [w::AbstractArray] nσ::Real=4)
Computes a robust standard mean and deviation value by flagging values through the median absolute deviation. 
"""
function robust_stats(x::AbstractArray; w::Union{Nothing, AbstractArray}=nothing, nσ::Real=4)
    if isnothing(w)
        w = get_mask(x)
    end
    med = weighted_quantile(x, w=w)
    adevs = abs.(med .- x)
    mad = weighted_quantile(adevs, w=w)
    good = findall(adevs .< 1.4826 * mad * nσ)
    if length(good) > 1
        return nanmean(@view x[good]), nanstd(@view x[good])
    else
        return NaN, NaN
    end
end

"""
    robust_σ(x::AbstractArray; [w::AbstractArray] nσ::Real=4)
Computes a robust standard mean value by flagging values through the median absolute deviation. 
"""
function robust_μ(x::AbstractArray; w::Union{Nothing, AbstractArray}=nothing, nσ::Real=4)
    if isnothing(w)
        w = get_mask(x)
    end
    med = weighted_quantile(x, w=w)
    adevs = abs.(med .- x)
    mad = weighted_quantile(adevs, w=w)
    good = findall(adevs .< 1.4826 * mad * nσ)
    if length(good) > 1
        return nanmean(@view x[good])
    else
        return NaN
    end
end

"""
    weighted_mean(x::AbstractArray{<:Real}, w::AbstractArray{<:Real})
Computes the weighted mean of an array.
"""
function weighted_mean(x::AbstractArray{<:Real}, w::AbstractArray{<:Real})
    good = findall(isfinite.(x) .&& (w .> 0) .&& isfinite.(w))
    if length(good) > 0
        xx = @view x[good]
        ww = @view w[good]
        return sum(xx .* ww) / sum(ww)
    else
        return NaN
    end
end


"""
    weighted_stddev(x::AbstractArray{<:Real}, w::AbstractArray{<:Real})
Computes the unbiased weighted standard deviation of an array.
"""
function weighted_stddev(x::AbstractArray{<:Real}, w::AbstractArray{<:Real}; μ=nothing)
    good = findall(@. isfinite(x) && (w > 0) && isfinite(w))
    xx = x[good]
    ww = w[good]
    ww ./= sum(ww)
    if isnothing(μ)
        μ = weighted_mean(xx, ww)
    end
    dev = xx .- μ
    bias_estimator = 1.0 - sum(ww.^2)
    var = sum(dev .^2 .* ww) / bias_estimator
    return sqrt(var)
end

"""
    weighted_quantile(x; w=nothing, p=0.5)
Computes the weighted percentile of an array.
"""
function weighted_quantile(x; w=nothing, p=0.5)
    if isnothing(w)
        w = ones(size(x))
    end
    good = findall(@. isfinite(x) && (w > 0) && isfinite(w))
    if length(good) > 0
        xx = @view x[good]
        ww = @view w[good]
        return quantile(xx, Weights(ww), p)
    else
        return NaN
    end
end

"""
    get_mask(x)
Construct defualt mask for an array by masking nans.
"""
function get_mask(x)
    w = ones(size(x))
    bad = findall(.~isfinite.(x))
    w[bad] .= 0
    return w
end