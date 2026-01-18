using LinearAlgebra
using Plots

# Use SVector for static allocation so not on heap?
coeffs_list = [
(8.28721201814563 , -23.595886519098837 , 17.300387312530933) ,
(4.107059111542203 , -2.9478499167379106 , 0.5448431082926601) ,
(3.9486908534822946 , -2.908902115962949 , 0.5518191394370137) ,
(3.3184196573706015 , -2.488488024314874 , 0.51004894012372) ,
(2.300652019954817 , -1.6689039845747493 , 0.4188073119525673) ,
(1.891301407787398 , -1.2679958271945868 , 0.37680408948524835) ,
(1.8750014808534479 , -1.2500016453999487 , 0.3750001645474248) ,
(1.875 , -1.25 , 0.375) , # subsequent coeffs equal this numerically
]
# safety factor for numerical stability ( but exclude last polynomial )
coeffs_list = vcat(
    [
        (
            a / 1.01,
            b / (1.01^3),
            c / (1.01^5)
        ) for (a,b,c) in coeffs_list[1:end-1]
    ],
    coeffs_list[end]
)

function PolarExpress(M::AbstractArray, steps::Int)
    @assert ndims(M) >= 2
    X = convert.(Float16, M)

    # If tall matrix, take transpose for FLOP reduction
    if size(X, ndims(X)-1)>size(X, ndims(X))
        X = permutedims(X)
    end

    # Normalize X
    X = X ./ (norm(X) .* 1.01 .+ 1e-7) # Dont know what the dims argument in the python code does

    if steps <= length(coeffs_list)
        hs = coeffs_list[1:steps]
    else
        pad = ntuple(_->coeffs_list[end], steps - length(coeffs_list))
        hs = vcat(coeffs_list, pad)
    end

    for (a, b, c) in hs
        A = X * X'
        B = b .* A .+ c .* (A * A)
        X = a .* X .+ B * X
    end


    # Undo earlier transpose if applied
    if size(M, ndims(M)-1) > size(M, ndims(M))
        X = permutedims(X)
    end
    return X


end

function PolarFactor(X)
    F = svd(X)
    return F.U * F.V'
end

function singVal(X)
    F = svd(X)
    return Diagonal(F.S)
end
