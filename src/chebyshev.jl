function build_cheby_flat(polys_x::Vector, polys_y::Vector, coeffs)
    n = length(polys_x)
    model = fill(NaN, n)
    nx, ny = size(coeffs)
    for i=1:n
        s = 0.0
        for j=1:nx
            for k=1:ny
                s += coeffs[j, k] * polys_x[i][j] * polys_y[i][k]
            end
        end
        model[i] = s
    end
    return model
end

function get_chebyvals(x::AbstractVector{<:Real}, y::AbstractVector{<:Real}, degx::Int, degy::Int)
    chebs_x = Vector{Float64}[]
    chebs_y = Vector{Float64}[]
    @assert length(x) == length(y)
    for i=1:length(x)
        push!(chebs_x, Maths.get_chebyvals(x[i], degx))
        push!(chebs_y, Maths.get_chebyvals(y[i], degy))
    end
    return chebs_x, chebs_y
end

function get_chebyvals(x::Real, n::Int)
    chebvals = zeros(n+1)
    for i=1:n+1
        coeffs = zeros(n+1)
        coeffs[i] = 1.0
        chebvals[i] = ChebyshevT(coeffs).(x)
    end
    return chebvals
end

function chebyval(x::Real, n::Int)
    coeffs = zeros(n+1)
    coeffs[n+1] = 1.0
    return ChebyshevT(coeffs).(x)
end