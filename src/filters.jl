using LoopVectorization

"""
    quantile_filter1d(x::AbstractVector, width::Int, p::Real)
A filter where x_out[i] = quantile(x[i-w2:i+w2], p) where w2 = floor(width / 2).
"""
function quantile_filter1d(x::AbstractVector; width::Int, p::Real=0.5)
    @assert isodd(width)
    nx = length(x)
    x_out = fill(NaN, nx)
    w2 = Int(floor(width / 2))
    for i=1:nx
        k1 = max(i - w2, 1)
        k2 = min(i + w2, nx)
        x_out[i] = @views NaNStatistics.nanquantile(x[k1:k2], p)
    end
    return x_out
end


"""
    median_filter2d(x::AbstractVector, width::Int)
A standard median filter where x_out[i, j] = median(x[i-w2:i+w2, j-w2:j+w2]) where w2 = floor(width / 2).
"""
function quantile_filter2d(x::AbstractMatrix; width::Int, p=0.5)
    @assert isodd(width)
    ny, nx = size(x)
    x_out = fill(NaN, (ny, nx))
    w2 = Int(floor(width / 2))
    for i=1:nx
        for j=1:ny
            kx1 = max(i - w2, 1)
            kx2 = min(i + w2, nx)
            ky1 = max(j - w2, 1)
            ky2 = min(j + w2, ny)
            x_out[j, i] = NaNStatistics.nanmedian((@view x[ky1:ky2, kx1:kx2]))
        end
    end
    return x_out
end


"""
    poly_filter(x::AbstractVector{<:Real}, y::AbstractVector{<:Real}; width::Real, deg::Int)
A filter created by rolling polynomial fits of degree `deg` and window size `width`.
"""
function poly_filter(x::AbstractVector{<:Real}, y::AbstractVector{<:Real}; width::Real, deg::Int)
    nx = length(x)
    y_out = fill(NaN, nx)
    for i=1:nx
        use = findall((abs.(x .- x[i]) .<= width) .&& isfinite.(y))
        if length(use) >= deg
            try
                y_out[i] = Polynomials.fit(x[use], y[use], deg)(x[i])
            catch
                nothing
            end
        end
    end
    return y_out
end

"""
    convolve1d(x::AbstractVector{<:Real}, k::AbstractArray{<:Real})
1D direct numerical convolution.
"""
function convolve1d(x::AbstractVector{<:Real}, k::AbstractArray{<:Real})
    nx = length(x)
    nk = length(k)
    n_pad = Int(floor(nk / 2))
    out = zeros(nx)
    kf = @view k[end:-1:1]
    valleft = x[1]
    valright = x[end]
    
    # Left values
    @inbounds for i=1:n_pad
        s = 0.0
        for j=1:nk
            ii = i - n_pad + j + 1
            if ii < 1
                s += valleft * kf[j]
            else
                s += x[ii] * kf[j]
            end
        end
        out[i] = s
    end

    # Middle values
    @turbo for i=n_pad+1:nx-n_pad
        s = 0.0
        for j=1:nk
            s += x[i - n_pad + j - 1] * kf[j]
        end
        out[i] = s
    end

    # Right values
    @inbounds for i=nx-n_pad+1:nx
        s = 0.0
        for j=1:nk
            ii = i - n_pad + j + 1
            if ii > nx
                s += valright * kf[j]
            else
                s += x[ii] * kf[j]
            end
        end
        out[i] = s
    end

    # Return out
    return out

end