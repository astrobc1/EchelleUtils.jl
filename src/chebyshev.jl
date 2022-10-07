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