

function pair_plot(samples, true_p=nothing)
    sample_len = samples.size[1]
    labels = ["dim" * string(i) for i in 1:sample_len]
    fig = plot(layout=(sample_len,sample_len), size=(1000, 1000),
                left_margin=Plots.Measures.Length(:mm, 10.0),
                right_margin=Plots.Measures.Length(:mm, 10.0),
                top_margin=Plots.Measures.Length(:mm, 3.0),
                bottom_margin=Plots.Measures.Length(:mm, 10.0))

    for i in 1:sample_len, j in 1:sample_len
        if i == j 
            if i == 1 
                histogram!(samples[i, :], bins=30, alpha=0.5, ylabel=labels[i],guidefont=font(10), label="ABC", 
                           subplot=(i-1)*sample_len + j, title="dim1", titlefont=font(10), normed=true)
                if true_p !== nothing
                    plot!([true_p[i], true_p[i]], [0, 10.0], color="red", linewidth=2, label="True")
                end
            else 
                histogram!(samples[i, :], bins=30, alpha=0.5, label="", subplot=(i-1)*sample_len + j, normed=true)
                if true_p !== nothing
                   plot!([true_p[i], true_p[i]], [0, 10.0], color="red", linewidth=2, label="", subplot=(i-1)*sample_len + j)
                end
            end
        elseif i == 1 
            scatter!(samples[j, :], samples[i, :], alpha=0.3, label="", 
                     title=labels[j], titlefont=font(10), subplot=(i-1)*sample_len + j #, xlims=(true_p[j] -0.5 , true_p[j]  + 0.5), ylims=(true_p[i] - 0.5, true_p[i]  + 0.5 )
                     )

            if true_p !== nothing
               scatter!([true_p[j]], [true_p[i]], color=:red, markersize=5, label="", subplot=(i-1)*sample_len + j,  xlims=(true_p[j] -0.5 , true_p[j]  + 0.5), ylims=(true_p[i] - 0.5, true_p[i]  + 0.5 ))
            end

        elseif j == 1
            scatter!(samples[j, :], samples[i, :], alpha=0.3, ylabel=labels[i], guidefont=font(10),  label="", subplot=(i-1)*sample_len + j #,  xlims=(true_p[j] -0.5 , true_p[j]  + 0.5), ylims=(true_p[i] - 0.5, true_p[i]  + 0.5 )
            )

            if true_p !== nothing
               scatter!([true_p[j]], [true_p[i]], color=:red, markersize=5, label="", subplot=(i-1)*sample_len + j #,  xlims=(true_p[j] -0.5 , true_p[j]  + 0.5), ylims=(true_p[i] - 0.5, true_p[i]  + 0.5 )
               )
            end

        else 
            scatter!(samples[j, :], samples[i, :], alpha=0.3, label="", subplot=(i-1)*sample_len + j#,  xlims=(true_p[j] -0.5 , true_p[j]  + 0.5), ylims=(true_p[i] - 0.5, true_p[i]  + 0.5 
            )
            if true_p !== nothing
               scatter!([true_p[j]], [true_p[i]], color=:red, markersize=5, label="", subplot=(i-1)*sample_len + j, xlims=(true_p[j] -0.5 , true_p[j]  + 0.5), ylims=(true_p[i] - 0.5, true_p[i]  + 0.5 ))
            end
        end
    end

    display(fig)
end


struct ParticleFilter
    #y::Vector{Float64}
    y::Union{Vector{Float64}, Matrix{Float64}}
    n_particle::Int
    θ_prior::Union{Distribution, Nothing}
    log_likelihood::Float64

    function ParticleFilter(
        #y::Vector{Float64}, 
        y::Union{Vector{Float64}, Matrix{Float64}},
        n_particle::Int)
        new(y, n_particle, nothing, -Inf)
    end

    function ParticleFilter(
        #y::Vector{Float64}, 
        y::Union{Vector{Float64}, Matrix{Float64}},
        n_particle::Int, 
        θ_prior::Distribution)
        new(y, n_particle, θ_prior, -Inf)
    end
end


function norm_likelihood(y, x, var)
    return (sqrt(2 * pi * var))^(-1) * exp(-(y - x)^2 / (2 * var))
end


function F_inv(w_cumsum, idx, u)
    k = findlast(w_cumsum .< u)
    # Condition ? Value if True : Value if False
    return isnothing(k) ? 1 : k + 1
end



function resampling(weights, n_particle)
    w_cumsum = cumsum(weights)
    idx = collect(1:n_particle)  # Juliaは1ベース
    k_list = zeros(Int32, n_particle)  # kを格納する配列
    
    # 一様分布に基づきリサンプリング
    for i in 1:n_particle
        # rand(): 一様乱数 
        u = rand()  # 0から1の一様乱数
        k = F_inv(w_cumsum, idx, u)
        #println("k:", k.i)
        k_list[i] = k
    end
    
    return k_list
end


function resampling2(weights, n_particle)
    idx = collect(1:n_particle)
    u0 = rand() / n_particle
    u = [1/n_particle * i + u0 for i in 0:(n_particle-1)]
    w_cumsum = cumsum(weights)
    return [F_inv(w_cumsum, idx, val) for val in u]
end



function simulate_DE(pf::ParticleFilter, obs_mean; do_resampling=true)
    #Random.seed!(seed)
    T = length(pf.y)

    x_before = zeros(T+1, pf.n_particle)
    x_before_resampled = zeros(T+1, pf.n_particle)
    
    x = zeros(T+1, pf.n_particle)
    x_resampled = zeros(T+1, pf.n_particle)
   #initial_x = rand(Normal(0, 1), pf.n_particle)
    initial_x = exp.(rand(Normal(0., 2.), pf.n_particle))
    #x_before_resampled[1, :] = initial_x
    #x_before[1, :] = initial_x
    #x_resampled[1, :] = exp.(initial_x)
    #x[1, :] = exp.(initial_x)
    x_resampled[1, :] = initial_x
    x[1, :] = initial_x

    w = zeros(T, pf.n_particle)
    w_normed = zeros(T, pf.n_particle)
    l = zeros(T)

    
    for t in 1:T
        #println("t:", t)
        if t % 20 == 0
            
        #println(t)
           println("\rCalculating... t=$t")  
        end 
        for i in 1:pf.n_particle
            v = sample_gamma(1/2, 4)
            x[t+1, i] = x_resampled[t, i] + v
            #println("x[t+1, i]", x[t+1, i])
            w[t, i] = norm_likelihood(pf.y[t], obs_mean[t, 1], x[t+1, i] + 0.04)
        end
        w_normed[t, :] = w[t, :] ./ sum(w[t, :])
        
        l[t] = log(sum(w[t, :]))
        #k = resampling2(w_normed[t, :], pf.n_particle)
        if do_resampling
            k = resampling2(w_normed[t, :], pf.n_particle)
            x_resampled[t+1, :] = x[t+1, k]
        else
            x_resampled[t+1, :] = x[t+1, :]
        end
        #x_resampled[t+1, :] = x[t+1, k]
    end
    
    log_likelihood = sum(l) - T * log(pf.n_particle)
    return x, x_resampled, w, w_normed, l, log_likelihood
end




function inverse_digamma(y; tol=1e-8)
    f(x) = digamma(x) - y
    return find_zero(f, (1e-6, 100.0), Bisection(), atol=tol)
end

function inverse_gamma_mm(x; max_iter=100, tol=1e-6)
    n = length(x)
    log_x̄ = mean(log.(x))
    inv_x_sum = sum(1 ./ x)

    # 初期値
    α = 1.0
    β = α * n / inv_x_sum

    for iter in 1:max_iter
        β_new = n * α / inv_x_sum
        target = log(β_new) - log_x̄
        α_new = inverse_digamma(target)

        # 収束判定
        if abs(α_new - α) < tol && abs(β_new - β) < tol
            break
        end

        α, β = α_new, β_new
    end

    return α, β
end



function effective_sample_size(weights::Vector{Float64})
    return 1.0 / sum(w^2 for w in weights)
end


function simulate_pf_selforg(
    ODE, 
    pf::ParticleFilter, 
    tspan,
    u0;
    start_time=0, 
    final_time=91, 
    do_resampling=true, 
    a=10.0, 
    b=0.1,
    obs_variance2 = 2.0^2
)
    T = length(pf.y)
    x = zeros(T+1, pf.n_particle)
    θ_particles = zeros(pf.n_particle)
    w = ones(T, pf.n_particle)
    w_normed = zeros(T, pf.n_particle)
    l = zeros(T)
    dt = 0.2
    block_sums = zeros(100, pf.n_particle)

    x[1, :] .= exp.(rand(Normal(0.0, 0.5), pf.n_particle)) .+ 1.0
    θ_particles .= clamp.(rand(pf.θ_prior, pf.n_particle), -1.9, 1.9)

    println("Preparation for data fitting")
    if start_time > 0
        for t in 1:start_time
            if t % 10 == 0 println("\rCalculating... t=$t") end
            j = 9 + t
            range = 1 + (j - 1) * 5 + 1 : 1 + (j - 1) * 5 + 5

            Threads.@threads for i in 1:pf.n_particle
                θ = θ_particles[i]
                p = [θ, -0.2, 1.0]

                num_euler, _ = numerical_integration(ODE, Euler, u0, dt, length(tspan), 2, p=p)
                num_runge = zeros(size(num_euler))
                num_runge[1, :] .= num_euler[1, :]

                k1 = zeros(length(u0))
                k2 = zeros(length(u0))
                sol2 = zeros(length(u0))

                for n in range
                    u = num_euler[n-1, :]
                    sol2, u_next = Runge(ODE, u, p, tspan[n-1], dt, k1, k2, sol2)
                    num_runge[n, :] = u_next
                end

                local_errors = abs.(num_euler .- num_runge)[:, 1]
                block_sums[9 + t, i] = sum(norm.(eachrow(local_errors[range, :])))
                v1 = block_sums[9 + t, i] != 0 ? sample_gamma(1, 1 / block_sums[9 + t, i]) : 0
                v3 = sample_gamma(a, 1 / b)
                proposed = v3 * x[t, i] + v1
                x[t+1, i] = max(proposed, x[t, i])
            end
        end
    end

    println("Move to data fitting")

    for t in start_time+1:final_time
        if t % 10 == 0 println("\rCalculating... t=$t") end
        j = 9 + t
        range = 1 + (j - 1) * 5 + 1 : 1 + (j - 1) * 5 + 5

        Threads.@threads for i in 1:pf.n_particle
            θ = θ_particles[i]
            p = [θ, -0.2, 1.0]

            num_euler, _ = numerical_integration(ODE, Euler, u0, dt, length(tspan), 2, p=p)
            num_runge = zeros(size(num_euler))
            num_runge[1, :] .= num_euler[1, :]

            k1 = zeros(length(u0))
            k2 = zeros(length(u0))
            sol2 = zeros(length(u0))

            for n in range
                u = num_euler[n-1, :]
                sol2, u_next = Runge(ODE, u, p, tspan[n-1], dt, k1, k2, sol2)
                num_runge[n, :] = u_next
            end

            local_errors = abs.(num_euler .- num_runge)[:, 1]
            block_sums[9 + t, i] = sum(norm.(eachrow(local_errors[range, :])))
            v1 = block_sums[9 + t, i] != 0 ? sample_gamma(1, 1 / block_sums[9 + t, i]) : 0
            v3 = sample_gamma(a, 1 / b)
            proposed = v3 * x[t, i] + v1
            x[t+1, i] = max(proposed, x[t, i])

            y_model = num_euler[obs_indices[t], 1]
            σ² = x[t+1, i]^2 + obs_variance2[1]

            if t != 1
                w[t, i] *= w[t-1, i] * norm_likelihood(pf.y[t], y_model, σ²)
            else
                w[t, i] = norm_likelihood(pf.y[t], y_model, σ²)
            end
        end

        w_normed[t, :] .= w[t, :] ./ sum(w[t, :])
        l[t] = log(sum(w[t, :]))

        if do_resampling
            k = resampling2(w_normed[t, :], pf.n_particle)
            x[t+1, :] .= x[t+1, k]
            θ_particles .= θ_particles[k]
            w[t, :] .= 1.0
        end
    end

    log_likelihood = sum(l) - T * log(pf.n_particle)
    return x, θ_particles, log_likelihood
end
