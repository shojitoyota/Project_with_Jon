function nls(func, params...; ini=[0.0], N_ftol=1e-8, N_xtol=0)
    if typeof(ini) <: Number
        return nlsolve((vout, vin) -> vout[1] = func(vin[1], params...), [ini], ftol=N_ftol, xtol=N_xtol).zero[1]
    else
        return nlsolve((vout, vin) -> vout .= func(vin, params...), ini, ftol=N_ftol, xtol=N_xtol).zero
    end
end

function Euler(f, u, p, t, dt)
    return u + dt * f(u, p, t)
end

function ImEuler(u_next, f, u, p, t, dt)
    return u_next - u - dt * f(u_next, p, t + dt)
end

function midpoint(u_half, f, u, p, t, dt)
    return u_half - u - dt / 2 * f(u_half, p, t + dt / 2)
end

function Runge(f, u, p, t, dt, k1, k2, sol2) #講義資料の台形即にあたる?
#function Runge(f, u, p, t, dt) #講義資料の台形即にあたる?
    k1 = f(u, p, t)
    sol2 = u + dt * k1
    k2 = f(sol2, p, t + dt)
    u = u + dt / 2 * (k1 + k2)
    return sol2, u
end

function RK4(f, u, p, t, dt, k1, k2, k3, k4, sol2, sol3, sol4)
    k1 = f(u, p, t)
    sol2 = u + dt / 2 * k1
    k2 = f(sol2, p, t + dt / 2)
    sol3 = u +  dt / 2 * k2
    k3 = f(sol3, p, t + dt / 2)
    sol4 = u +  dt * k3
    k4 = f(sol4, p, t + dt)
    u = u + dt / 6 * (k1 + 2 * k2 + 2 * k3 + k4 )
    return sol2, sol3, sol4, u
end


function RK3(f, u, p, t, dt)
    k1 = f(u, p, t)
    sol2 = u + dt/2 * k1
    k2 = f(sol2, p, t + dt/2)
    sol3 =  u - dt * k1 + 2 * dt * k2
    k3 = f(sol3, p, t + dt)
    u_new = u + dt/6 * (k1 + 4*k2 + k3)
    return u_new
end


function numerical_integration(f, method, u0, h, N, k; p=[], nls_ftol=1e-8, nls_xtol=0)
    dt = h / k
    
    if method == Euler
        if length(u0) == 1
            final = zeros(N)
            final[1] = u0
            inter_and_final = zeros((N - 1) * k + 1);
            inter_and_final[1] = u0;
            j = 1
            for n = 1:N - 1
                for m = 1:k
                    inter_and_final[j + 1] = Euler(f, inter_and_final[j], p, dt * (j - 1), dt)
                    j = j + 1
                end
                final[n + 1] = inter_and_final[j];
            end
            return final, inter_and_final
        else
            final = zeros(N, length(u0))
            final[1,:] = u0
            inter_and_final = zeros((N - 1) * k + 1, length(u0));
            inter_and_final[1,:] = u0;
            j = 1
            for n = 1:N - 1
                for m = 1:k
                    inter_and_final[j + 1,:] = Euler(f, inter_and_final[j,:], p, dt * (j - 1), dt)
                    j = j + 1
                end
                final[n + 1,:] = inter_and_final[j,:];
            end
            return final, inter_and_final
        end
    end
    
    if method == ImEuler
        if length(u0) == 1
            final = zeros(N)
            final[1] = u0
            inter_and_final = zeros((N - 1) * k + 1);
            inter_and_final[1] = u0;
            j = 1
            for n = 1:N - 1
                for m = 1:k
                    inter_and_final[j + 1] =  nls(ImEuler, f, inter_and_final[j], p, dt * (j - 1), dt, ini=inter_and_final[j], N_ftol=nls_ftol, N_xtol=nls_xtol);
                    j = j + 1
                end
                final[n + 1] = inter_and_final[j];
            end
            return final, inter_and_final
        else
            final = zeros(N, length(u0))
            final[1,:] = u0
            inter_and_final = zeros((N - 1) * k + 1, length(u0));
            inter_and_final[1,:] = u0;
            j = 1
            for n = 1:N - 1
                for m = 1:k
                    inter_and_final[j + 1,:] = nls(ImEuler, f, inter_and_final[j,:], p, dt * (j - 1), dt, ini=inter_and_final[j,:], N_ftol=nls_ftol, N_xtol=nls_xtol);
                    j = j + 1
                end
                final[n + 1,:] = inter_and_final[j,:];
            end
            return final, inter_and_final
        end
    end
    
    if method == Runge
        if length(u0) == 1
            final = zeros(N)
            final[1] = u0
            inter_and_final = zeros(2 * (N - 1) * k + 1);
            inter_and_final[1] = u0;
            k1, k2, sol2 = 0, 0, 0
            j = 1
            for n = 1:N - 1
                for m = 1:k
                    inter_and_final[2 * j], inter_and_final[2 * j + 1] = Runge(f, inter_and_final[2 * j - 1], p, dt * (j - 1), dt, k1, k2, sol2)
                    j = j + 1
                end
                final[n + 1] = inter_and_final[2 * j - 1];
            end
            return final, inter_and_final
        else
            final = zeros(N, length(u0))
            final[1,:] = u0
            inter_and_final = zeros(2 * (N - 1) * k + 1, length(u0));
            inter_and_final[1,:] = u0;
            k1, k2, sol2 = zeros(length(u0)), zeros(length(u0)), zeros(length(u0))
            j = 1
            for n = 1:N - 1
                for m = 1:k
                    inter_and_final[2 * j,:], inter_and_final[2 * j + 1,:] = Runge(f, inter_and_final[2 * j - 1,:], p, dt * (j - 1), dt, k1, k2, sol2)
                    j = j + 1
                end
                final[n + 1,:] = inter_and_final[2 * j - 1,:];
            end
            return final, inter_and_final
        end
    end
    
    if method == midpoint
        if length(u0) == 1
            final = zeros(N)
            final[1] = u0
            inter_and_final = zeros(2 * (N - 1) * k + 1);
            inter_and_final[1] = u0;
            j = 1
            for n = 1:N - 1
                for m = 1:k
                    inter_and_final[2 * j] =  nls(midpoint, f, inter_and_final[2 * j - 1], p, dt * (j - 1), dt, ini=inter_and_final[2 * j - 1], N_ftol=nls_ftol, N_xtol=nls_xtol);
                    inter_and_final[2 * j + 1] = inter_and_final[2 * j - 1] + dt * f(inter_and_final[2 * j], p, dt * (j - 1) + dt / 2)
                    j = j + 1
                end
                final[n + 1] = inter_and_final[2 * j - 1];
            end
            return final, inter_and_final
        else
            final = zeros(N, length(u0))
            final[1,:] = u0
            inter_and_final = zeros(2 * (N - 1) * k + 1, length(u0));
            inter_and_final[1,:] = u0;
            j = 1
            for n = 1:N - 1
                for m = 1:k
                    inter_and_final[2 * j,:] = nls(midpoint, f, inter_and_final[2 * j - 1,:], p, dt * (n - 1), dt, ini=inter_and_final[2 * j - 1,:], N_ftol=nls_ftol, N_xtol=nls_xtol);
                    inter_and_final[2 * j + 1,:] = inter_and_final[2 * j - 1,:] + dt * f(inter_and_final[2 * j,:], p, dt * (j - 1) + dt / 2)
                    j = j + 1
                end
                final[n + 1,:] = inter_and_final[2 * j - 1,:];
            end
            return final, inter_and_final
        end
    end
    
    if method == RK4
        if length(u0) == 1
            final = zeros(N)
            final[1] = u0
            inter_and_final = zeros(4 * (N - 1) * k + 1);
            inter_and_final[1] = u0;
            k1, k2, k3, k4, sol2, sol3, sol4 = 0, 0, 0, 0, 0, 0, 0
            j = 1
            for n = 1:N - 1
                for m = 1:k
                    inter_and_final[4 * j - 2], inter_and_final[4 * j - 1], inter_and_final[4 * j], inter_and_final[4 * j + 1] = RK4(f, inter_and_final[4 * j - 3], p, dt * (j - 1), dt, k1, k2, k3, k4, sol2, sol3, sol4)
                    j = j + 1
                end
                final[n + 1] = inter_and_final[4 * j - 3];
            end
            return final, inter_and_final
        else
            final = zeros(N, length(u0))
            final[1,:] = u0
            inter_and_final = zeros(4 * (N - 1) * k + 1, length(u0));
            inter_and_final[1,:] = u0;
            k1, k2, k3, k4, sol2, sol3, sol4 = zeros(length(u0)), zeros(length(u0)), zeros(length(u0)), zeros(length(u0)), zeros(length(u0)), zeros(length(u0)), zeros(length(u0))
            j = 1
            for n = 1:N - 1
                for m = 1:k
                    inter_and_final[4 * j - 2,:], inter_and_final[4 * j - 1,:], inter_and_final[4 * j,:], inter_and_final[4 * j + 1,:] = RK4(f, inter_and_final[4 * j - 3,:], p, dt * (j - 1), dt, k1, k2, k3, k4, sol2, sol3, sol4)
                    j = j + 1
                end
                final[n + 1,:] = inter_and_final[4 * j - 3,:];
            end
            return final, inter_and_final
        end

    end

    if method == RK3
        if length(u0) == 1
            final = zeros(N)
            final[1] = u0
            inter_and_final = zeros(3 * (N - 1) * k + 1)
            inter_and_final[1] = u0
            j = 1
            for n = 1:N - 1
                for m = 1:k
                    # RK3はu_newのみを返す単純形
                    inter_and_final[3 * j + 1] = RK3(f, inter_and_final[3 * j - 2], p, dt * (j - 1), dt)
                    j += 1
                end
                final[n + 1] = inter_and_final[3 * j - 2]
            end
            return final, inter_and_final
        else
            final = zeros(N, length(u0))
            final[1, :] = u0
            inter_and_final = zeros(3 * (N - 1) * k + 1, length(u0))
            inter_and_final[1, :] = u0
            j = 1
            for n = 1:N - 1
                for m = 1:k
                    inter_and_final[3 * j + 1, :] = RK3(f, inter_and_final[3 * j - 2, :], p, dt * (j - 1), dt)
                    j += 1
                end
                final[n + 1, :] = inter_and_final[3 * j - 2, :]
            end
            return final, inter_and_final
        end
    end
    
end