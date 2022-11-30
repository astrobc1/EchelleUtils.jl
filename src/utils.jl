export flatten_jagged_vector, get_inds1d

macro tryignore(expr)
    quote
        try
            esc(expr)
        catch e
            @error "Error reached but ignored with @tryignore!" exception=(e, catch_backtrace())
        end
    end
end

macro debugerror(expr)
    quote
        try
            esc(expr)
        catch
            @infiltrate
        end
    end
end

const COLORS_HEX_GADFLY = [
    "#00BEFF", "#D4CA3A", "#FF6DAE", "#67E1B5", "#EBACFA",
    "#9E9E9E", "#F1988E", "#5DB15A", "#E28544", "#52B8AA"
]

"""
    flatten_jagged_vector(x::Vector{Vector{T}}) where{T}
Flattens a vector of vectors, each possibly with a different length.
"""
function flatten_jagged_vector(x::Vector{Vector{T}}) where{T}
    y = T[]
    for i ∈ eachindex(x)
        y = vcat(y, x[i])
    end
    return y
end

"""
    get_inds1d(coords; dim::Int)
Gets the 1d array indices returned from the results of findall for multi-dimensional arrays.
"""
function get_inds1d(coords; dim::Int)
    good = [coord.I[dim] for coord ∈ coords]
    return good
end