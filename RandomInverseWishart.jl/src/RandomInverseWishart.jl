module RandomInverseWishart

using Distributions
using LinearAlgebra
using Random

export rwishart_chol!, rinvwishart_chol!, rinvwishart_direct!, rinvwishart_indirect!

function rwishart_chol!(rng::AbstractRNG, T::Matrix, m::Int, n::Int, UΣ::AbstractMatrix)
    Z = T
    for j in 1:m
        for i in 1:(j - 1)
            Z[i, j] = randn(rng)
        end
        Z[j, j] = rand(rng, Chi(n + 1 - j; check_args=false))
        for i in (j + 1):m
            Z[i, j] = 0
        end
    end
    T = rmul!(Z, UpperTriangular(UΣ))
    return T
end

function cholesky_upper!(
    S,
    _tmp::Union{Nothing,Matrix}=nothing,
    _tmp2::Union{Nothing,Matrix}=nothing;
    invert::Bool=false,
    ischolU::Bool=false,
)
    if !invert
        if ischolU
            return UpperTriangular(S)
        else
            tmp = _tmp === nothing ? similar(S) : _tmp
            return cholesky!(Symmetric(copyto!(tmp, S))).U
        end
    else
        tmp = _tmp === nothing ? similar(S) : _tmp
        tmp2 = _tmp2 === nothing ? similar(S) : _tmp2
        if ischolU
            copyto!(tmp2, S)
            U = UpperTriangular(tmp2)
        else
            U = cholesky!(Symmetric(copyto!(tmp2, S))).U
        end
        C = LinearAlgebra.inv!(U)
        mul!(tmp, triu!(parent(C)), C')
        return cholesky!(Symmetric(tmp)).U
    end
end

function rinvwishart_chol!(rng::AbstractRNG, R::Matrix, m::Int, n::Int, UΩ::AbstractMatrix)
    Z = R
    for j in 1:m
        for i in 1:(j - 1)
            Z[i, j] = randn(rng)
        end
        Z[j, j] = rand(rng, Chi(n - m + j; check_args=false))
    end
    C = parent(LinearAlgebra.inv!(UpperTriangular(Z)))
    triu!(C)
    R = rmul!(C, UpperTriangular(UΩ))
    return R
end

function rinvwishart_indirect!(
    rng::AbstractRNG,
    B::Matrix,
    m::Int,
    n::Int,
    S::AbstractMatrix;
    iscov::Bool=false,
    ischolU::Bool=false,
    retcholU::Bool=false,
)
    tmp = similar(B)
    UΣ = cholesky_upper!(S, B, tmp; ischolU, invert=!iscov)
    T = rwishart_chol!(rng, tmp, m, n, UΣ)
    V = parent(LinearAlgebra.inv!(UpperTriangular(T)))
    triu!(V)
    mul!(B, V, UpperTriangular(V)')
    if retcholU
        R = triu!(cholesky!(Symmetric(B)).factors)
        return R
    else
        return B
    end
end

function rinvwishart_direct!(
    rng::AbstractRNG,
    B::Matrix,
    m::Int,
    n::Int,
    S::AbstractMatrix;
    iscov::Bool=false,
    ischolU::Bool=false,
    retcholU::Bool=false,
)
    tmp1 = if ischolU && retcholU && !iscov
        nothing
    else
        retcholU ? similar(B) : B
    end
    tmp2 = retcholU ? B : similar(B)
    UΩ = cholesky_upper!(S, tmp1, tmp2; ischolU, invert=iscov)
    R = rinvwishart_chol!(rng, tmp2, m, n, UΩ)
    if retcholU
        return R
    else
        mul!(B, UpperTriangular(R)', R)
        return B
    end
end

end  # module
