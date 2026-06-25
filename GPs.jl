function rbf_kernel(x1, x2; ℓ=1.0, σf=1.0)
    σf^2 * exp(-(x1 - x2)^2 / (2ℓ^2))
end

function rbf_kernel_simple(x1, x2; ℓ=1.0)
    (10.0)*exp(-(x1 - x2)^2 / (2ℓ^2))
end

function kernel_matrix(X1, X2; ℓ=1.0, σf=1.0)
    [rbf_kernel(x1, x2; ℓ=ℓ, σf=σf) for x1 in X1, x2 in X2]
end

function kernel_matrix_simple(X1, X2; ℓ=1.0)
    [rbf_kernel_simple(x1, x2; ℓ=1.0) for x1 in X1, x2 in X2]
end



function log_marginal_likelihood(X, y; ℓ, σf, σn)
    K = kernel_matrix(X, X; ℓ=ℓ, σf=σf) + σn^2 * I

    L = cholesky(Hermitian(K)).L
    α = L' \ (L \ y)

    n = length(y)
    logdetK = 2sum(log.(diag(L)))

    return -0.5 * y' * α - 0.5 * logdetK - 0.5 * n * log(2π)
end

function log_marginal_extended(ts, y; dt, ℓ, σf, prior_noise = 0.0)

    K = build_observation_covariance(ts; dt=dt, ℓ=ℓ, σf=σf)

    #K[2:end,2:end] .+= 2.0 .* I
    n = size(K,1) - 1
    K[2:end,2:end] .+= prior_noise * I(n)

    L = cholesky(Hermitian(K)).L
    α = L' \ (L \ y)

    n = length(y)
    logdetK = 2sum(log.(diag(L)))

    return -0.5 * (y' * α) - 0.5 * logdetK - 0.5 * n * log(2π)
end

function log_marginal_extended2(ts, y; dt, ℓ, σf, prior_mean, prior_noise = 0.0)

    K = build_observation_covariance(ts; dt=dt, ℓ=ℓ, σf=σf)

    n = size(K,1) - 1
    K[2:end,2:end] .+= prior_noise * I(n)

    # ★ mean vectorを追加
    m = vcat([prior_mean], zeros(n))

    L = cholesky(Hermitian(K)).L

    α = L' \ (L \ (y - m))

    logdetK = 2sum(log.(diag(L)))

    return -0.5 * ((y - m)' * α) - 0.5 * logdetK - 0.5 * length(y) * log(2π)
end