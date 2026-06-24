# compute the exact (reference) solution
function exact(f, u0, tspan, Method, rel_tol, abs_tol;p=[])
    prob = ODEProblem(f, u0, (0, tspan[end]), p)
    sol = solve(prob, Method, saveat=tspan, reltol=rel_tol, abstol=abs_tol)
    
    if length(u0) == 1
        return sol
    else
        return sol'
    end
end

# Generate obsavation data
function observation(u, tspan, Gamma;tspan_obs=(tspan[1],tspan[end]))
    Random.seed!(1234)
    n = length(u[:,1])
    m = length(u[1,:])
    d = Normal()
#     print(n)
    if m == 1
        obs = u + Gamma .* rand(d, n)
    else
        obs = u + Gamma' .* rand(d, n, m)
#         obs = u + Gamma .* rand(d, m, n)
    end

    idx = findall(x -> tspan_obs[1] <= x <= tspan_obs[2], tspan)

    if m == 1
        obs = obs[idx]
    else
#         obs = obs[:,idx]
        obs = obs[idx,:]
    end
    tspan_new = tspan[idx]
    
    return (sol = obs, time = tspan_new)
    
#     if m == 1
#         return (sol = obs, time = tspan_new)
#     else
#         return (sol = obs', time = tspan_new)
#     end
end


function observation2(u, tspan, obs_operator, Gamma;tspan_obs=(tspan[1],tspan[end]))
    Random.seed!(1234)
    n = length(u[:,1])
    m = length(u[1,:])
    d = Normal()
#     print(n)
    if m == 1
        obs = obs_operator .* u + Gamma .* rand(d, n)
    else
        obs = obs_operator .* u + Gamma' .* rand(d, n, m)
#         obs = u + Gamma .* rand(d, m, n)
    end

    idx = findall(x -> tspan_obs[1] <= x <= tspan_obs[2], tspan)

    if m == 1
        obs = obs[idx]
    else
#         obs = obs[:,idx]
        obs = obs[idx,:]
    end
    tspan_new = tspan[idx]
    
    return (sol = obs, time = tspan_new)
    
#     if m == 1
#         return (sol = obs, time = tspan_new)
#     else
#         return (sol = obs', time = tspan_new)
#     end
end



# Generate obsavation data (without seed)
function observation_without_seed(u, tspan, Gamma;tspan_obs=(tspan[1],tspan[end]))
    n = length(u[:,1])
    m = length(u[1,:])
    d = Normal()
#     print(n)
    if m == 1
        obs = u + Gamma .* rand(d, n)
    else
        obs = u + Gamma' .* rand(d, n, m)
#         obs = u + Gamma .* rand(d, m, n)
    end

    idx = findall(x -> tspan_obs[1] <= x <= tspan_obs[2], tspan)

    if m == 1
        obs = obs[idx]
    else
#         obs = obs[:,idx]
        obs = obs[idx,:]
    end
    tspan_new = tspan[idx]
    
    return (sol = obs, time = tspan_new)
    
#     if m == 1
#         return (sol = obs, time = tspan_new)
#     else
#         return (sol = obs', time = tspan_new)
#     end
end


