using Distributions
using LinearAlgebra
using RandomInverseWishart
using Random
using Statistics
using Test

function rand_chol_pd_mat(rng, m)
    R = qr(randn(rng, m, m)).R
    R .*= sign.(diag(R))
    return R
end
rand_chol_pd_mat(m) = rand_chol_pd_mat(Random.default_rng(), m)

@testset "RandomInverseWishart.jl" begin
    @testset "cholesky_upper!" begin
        @testset for m in (5, 20)
            UΩ = rand_chol_pd_mat(m)
            Ω = UΩ' * UΩ
            Σ = Matrix(inv(Symmetric(Ω)))
            UΣ = Matrix(cholesky(Symmetric(Σ)).U)
            @test RandomInverseWishart.cholesky_upper!(UΩ; invert=false, ischolU=true) ===
                UpperTriangular(UΩ)
            @test RandomInverseWishart.cholesky_upper!(Ω; invert=false, ischolU=false) ≈
                UpperTriangular(UΩ)
            @test RandomInverseWishart.cholesky_upper!(Σ; invert=true, ischolU=false) ≈
                UpperTriangular(UΩ)
            @test RandomInverseWishart.cholesky_upper!(UΣ; invert=true, ischolU=true) ≈
                UpperTriangular(UΩ)
        end
    end

    @testset "$rinvwishart!" for rinvwishart! in
                                 (rinvwishart_indirect!, rinvwishart_direct!)
        ndraws = 10_000
        rng = Random.default_rng()
        ntests = 4 * (5 * (5 + 1) + (20 * (20 + 1)))
        α = 0.01 / ntests
        atol_mul = quantile(Normal(), 1 - α / 2)
        @testset for iscov in (true, false),
            ischolU in (true, false),
            m in (5, 20),
            n in (m + 4, m + 10)

            UΩ = rand_chol_pd_mat(m)
            Ω = UΩ' * UΩ
            if iscov
                Σ = inv(cholesky(Symmetric(Ω)))
                S = ischolU ? Matrix(cholesky(Symmetric(Σ)).U) : Matrix(Symmetric(Σ))
            else
                S = ischolU ? UΩ : Ω
            end

            B = similar(S)
            rng2 = MersenneTwister(42)
            @test @inferred(rinvwishart!(rng2, B, m, n, S; iscov, ischolU)) === B

            UB = similar(B)
            rng2 = MersenneTwister(42)
            rinvwishart!(rng2, UB, m, n, S; iscov, ischolU, retcholU=true)
            @test istriu(UB)
            @test UB' * UB ≈ B

            Bs = map(1:ndraws) do _
                rinvwishart!(rng, similar(S), m, n, S; iscov, ischolU)
            end
            dist = InverseWishart(n, Ω)
            mean_exp = mean(dist)
            mean_est = mean(Bs)
            # check marginal mean estimates consistent with known mean
            for j in 1:m, i in 1:j
                vij = var(dist, i, j)
                @test mean_est[i, j] ≈ mean_exp[i, j] atol = atol_mul * sqrt(vij / ndraws)
            end
        end

        if rinvwishart! === rinvwishart_direct!
            m = n = 100
            UΩ = rand_chol_pd_mat(m)
            UB = similar(UΩ)
            rinvwishart!(rng, UB, m, n, UΩ; iscov=false, ischolU=true, retcholU=true)
            @test @allocated(
                rinvwishart!(rng, UB, m, n, UΩ; iscov=false, ischolU=true, retcholU=true)
            ) ≤ 224
        end
    end
end
