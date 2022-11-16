export flatten_jagged_vector, get_inds1d

const COLORS_HEX_GADFLY = [
    "#00BEFF", "#D4CA3A", "#FF6DAE", "#67E1B5", "#EBACFA",
    "#9E9E9E", "#F1988E", "#5DB15A", "#E28544", "#52B8AA"
]

function flatten_jagged_vector(x)
    y = eltype(x[1])[]
    for i ∈ eachindex(x)
        y = vcat(y, x[i])
    end
    return y
end

function get_inds1d(coords; dim::Int)
    good = [coord.I[dim] for coord ∈ coords]
    return good
end