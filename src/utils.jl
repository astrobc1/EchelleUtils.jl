const COLORS_HEX_GADFLY = [
    "#00BEFF", "#D4CA3A", "#FF6DAE", "#67E1B5", "#EBACFA",
    "#9E9E9E", "#F1988E", "#5DB15A", "#E28544", "#52B8AA"
]

function flatten_jagged_vector(x)
    y = eltype(x[1])[]
    for i=1:length(x)
        y = vcat(y, x[i])
    end
    return y
end

function findgood(x...)
    good = trues(length(x[1]))
    for arr in x
        good .= good .&& isfinite.(arr)
    end
    good = findall(good)
    return good
end

function filtergood(x...)
    good = findgood(x...)
    xout = ()
    for arr in x
        xout = (xout..., arr[good])
    end
    return xout
end