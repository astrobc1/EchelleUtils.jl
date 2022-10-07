
"""
    gauss(x, a, μ, σ)
Construct a Guassian function.
"""
function gauss(x, a, μ, σ)
    return @. a * exp(-0.5 * ((x - μ) / σ)^2)
end

"""
    rmsloss(residuals::AbstractArray{<:Real}, [weights::AbstractArray{<:Real}=nothing]; mask_worst::Int=0, mask_edges::Int=0)
Computes the root mean squared error (RMS) loss. Weights can also be provided, otherwise uniform weights will be used.
"""
function rmsloss(residuals::AbstractArray{<:Real}, weights::Union{AbstractArray{<:Real}, Nothing}=nothing; mask_worst::Int=0, mask_edges::Int=0)

    # Get good data
    if !isnothing(weights)
        good = findall(isfinite.(residuals) .&& isfinite.(weights) .&& (weights .> 0))
        rr, ww = residuals[good], weights[good]
    else
        good = findall(isfinite.(residuals))
        rr = residuals[good]
        ww = ones(length(rr))
    end
    
    # Ignore worst N pixels
    if mask_worst > 0
        ss = sortperm(abs.(rr))
        rr[ss[end-mask_worst+1:end]] .= NaN
        if !isnothing(weights)
            ww[ss[end-mask_worst+1:end]] .= 0
        end
    end

    # Remove edges
    if mask_edges > 0
        rr[1:mask_edges] .= NaN
        rr[end-mask_edges+1:end] .= NaN
        if !isnothing(weights)
            ww[1:mask_edges] .= 0
            ww[end-mask_edges+1:end] .= 0
        end
    end
        
    # Compute rms
    rms = sqrt(nansum(ww .* rr.^2) / nansum(ww))

    # Return
    return rms
end

"""
    redχ2loss(residuals::AbstractArray{<:Real}, [errors::AbstractArray{<:Real}=nothing]; mask_worst::Int=0, mask_edges::Int=0)
Computes the reduced chi square loss.
"""
function redχ2loss(residuals::AbstractArray{<:Real}, errors::AbstractArray{<:Real}, mask::Union{AbstractArray{<:Real}, Nothing}=nothing; mask_worst=0, mask_edges=0, ν=nothing)

    # Compute diffs2
    if isnothing(mask)
        good = findall(isfinite.(residuals) .&& isfinite.(errors) .&& (mask .== 1))
        residuals, errors, mask = residuals[good], errors[good], mask[good]
    else
        good = findall(isfinite.(residuals) .&& isfinite.(errors))
        residuals, errors = residuals[good], errors[good]
        mask = ones(length(residuals))
    end

    # Remove edges
    if mask_edges > 0
        residuals[1:mask_edges-1] .= NaN
        residuals[end-mask_edges+1:end] .= NaN
        mask[1:mask_edges-1] .= 0
        mask[end-mask_edges+1:end] .= 0
    end
    
    # Ignore worst N pixels
    if mask_worst > 0
        ss = sortperm(abs.(residuals))
        residuals[ss[end-mask_worst+1:end]] .= NaN
        mask[ss[end-mask_worst+1:end]] .= 0
    end

    # Degrees of freedom
    if isnothing(ν)
        ν = sum(mm) - 2 * mask_edges - mask_worst - 1
    else
        ν = ν - 2 * mask_edges - mask_worst
    end

    @assert ν > 0

    # Compute chi2
    redχ² = nansum((residuals ./ errors).^2) / ν

    # Return
    return redχ²
end


"""
    robust_σ(x::AbstractArray; [w::AbstractArray] nσ::Real=4)
Computes a robust standard deviation value by flagging values through the median absolute deviation. 
"""
function robust_σ(x::AbstractArray; w::Union{Nothing, AbstractArray}=nothing, nσ::Real=4)
    if isnothing(w)
        w = get_weights(x)
    end
    med = weighted_median(x, w=w)
    adevs = abs.(med .- x)
    mad = weighted_median(adevs, w=w)
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
        w = get_weights(x)
    end
    med = weighted_median(x, w=w)
    adevs = abs.(med .- x)
    mad = weighted_median(adevs, w=w)
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
        w = get_weights(x)
    end
    med = weighted_median(x, w=w)
    adevs = abs.(med .- x)
    mad = weighted_median(adevs, w=w)
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
function weighted_stddev(x::AbstractArray{<:Real}, w::AbstractArray{<:Real})
    good = findall(isfinite.(x) .&& (w .> 0) .&& isfinite.(w))
    xx = x[good]
    ww = w[good]
    ww ./= sum(ww)
    μ = weighted_mean(xx, ww)
    dev = xx .- μ
    bias_estimator = 1.0 - sum(ww.^2)
    var = sum(dev .^2 .* ww) / bias_estimator
    return sqrt(var)
end

"""
    weighted_median(x; w=nothing, p=0.5)
Computes the weighted percentile of an array.
"""
function weighted_median(x; w=nothing, p=0.5)
    if isnothing(w)
        w = ones(size(x))
    end
    good = findall(isfinite.(x) .&& (w .> 0) .&& isfinite.(w))
    if length(good) > 0
        xx = @view x[good]
        ww = @view w[good]
        return quantile(xx, Weights(ww), p)
    else
        return NaN
    end
end

"""
    get_weights(x)
Construct defualt weights for an array.
"""
function get_weights(x)
    w = ones(size(x))
    bad = findall(.~isfinite.(x))
    w[bad] .= 0
    return w
end