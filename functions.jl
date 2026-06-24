function sample_normal(μ, σ)
    dist = Normal(μ, σ)
    return rand(dist)
end

function sample_gamma(α, θ)
    # θ: scale parameter 
    dist = Gamma(α, 1/θ)
    return rand(dist)
end

function sample_truncated_normal(μ, σ)
    dist = Truncated(Normal(μ, σ), 0, Inf)
    return rand(dist)
end

function sample_truncated_normal(μ, σ, a)
    dist = Truncated(Normal(μ, σ), a, Inf)
    return rand(dist)
end

function sample_inv_gamma(α, θ)
    dist = InverseGamma(α, θ)
    return rand(dist)
end

function sample_exponential(λ)
    dist = Exponential(1/λ)
    return rand(dist)
end

function sample_truncated_exponential(λ,a)
    dist = Truncated(Exponential(1/λ), a, Inf)
    return rand(dist)
end

# function generate_e(y,η,D)
#     n = length(y)
#     E = zeros(n,n)
#     for i = 1:n
#         E[:,i] = y
#         for j = 1:n
#             if j != i
#                 E[:,i] -= η[j]*D[:,j]
#             end
#         end
#     end
#     return E
# end

function generate_e(y,η,D)
    n = length(y)
    E = zeros(n,n)
    for i = 1:n
        E[:,i] = y - D*η + η[i] * D[:,i]
    end
    return E
end

function generate_e(y,η,D,i)
    return y - D*η + η[i] * D[:,i]
end



#　My code for GIG

function sample_gig(λ, a, b)
    # check GIG parameters
    if a < 0 || b < 0
        throw(DomainError((a,b), "a and b must be nonnegative"))
    elseif b == 0 && λ <= 0 
        throw(DomainError((λ,b), "When b=0, λ must be positive"))
    elseif a == 0 && λ >= 0
        throw(DomainError((λ,a), "When a = 0, λ must be negative"))
    end

    # alternative parametrization
    β = sqrt(a*b)
    α = sqrt(b/a)
    abs_λ = abs(λ)

    # run generator
    ztol =  8*eps()

    if β < ztol
        if λ > 0
            return sample_gamma(λ,a/2)
        else λ < 0
            return 1/sample_gamma(-λ,b/2)
        end
    end

    if abs_λ > 2 || β > 3
        # Ratio-of-uniforms with shift by 'mode', alternative implementation
        return rgig_ROU_shift_alt(abs_λ, λ, β, α)
    end

    if abs_λ >= 1.0-2.25*β^2 || β > 0.2
        # Ratio-of-uniforms without shift 
        return rgig_ROU_noshift(abs_λ, λ, β, α)
    end

    if β > 0. # remaining case
        # New approach, constant hat in log-concave part
        return rgig_newapproach1(abs_λ, λ, β, α)
    else
        throw(DomainError((λ,β),"parameters must satisfy λ>=0 and β>0."))
    end
end

function sample_truncated_gig(λ, a, b, l)
    tmp = 0
    while true 
        tmp = sample_gig(λ, a, b)
        if tmp >= l
            return tmp
        end
    end
end

function gig_mode(λ,β)
    # Compute mode of GIG distribution.                                         
    # Parameters:                                                              
    #     λ ... parameter for distribution                                    
    #     β ... parameter for distribution                                    
    # Return: mode    
    if λ >= 1
        # mode of fgig(x)
        return (sqrt((λ-1)^2 + β^2)+(λ-1))/β
    else
        # 0 <= λ < 1: use reciprocal of mode of f(1/x)
        return β / (sqrt((1-λ)^2 + β^2)+(1-λ))    
    end
end                                                        

function rgig_ROU_shift_alt(λ, λ_old, β, α)                         
    # Ratio-of-uniforms with shift by 'mode', alternative implementation.       
    # Dagpunar (1989)                                                         
    # Lehner (1989)       
    
    # shortcuts
    t = 0.5 * (λ -1)
    s = 0.25 * β

    # mode = location of maximum of sqrt(f(x)) 
    xm = gig_mode(λ, β)

    # normalization constant: c = log(sqrt(f(xm)))
    nc = t*log(xm) - s*(xm + 1/xm)

    # location of minimum and maximum of (1/x)*sqrt(f(1/x+m)):  

    # compute coeffients of cubic equation y^3+a*y^2+b*y+c=0 
    a = -(2*(λ+1)/β + xm) # < 0 
    b = 2*(λ-1)*xm/β - 1
    c = xm

    # we need the roots in (0,xm) and (xm,inf)

    # substitute y=z-a/3 for depressed cubic equation z^3+p*z+q=0
    p = b - a^2/3
    q = (2*a^3)/27 - (a*b)/3 + c

    # use Cardano's rule
    fi = acos(-q/(2*sqrt(-(p^3)/27)))
    fak = 2*sqrt(-p/3)
    y1 = fak * cos(fi/3) - a/3
    y2 = fak * cos(fi/3 + 4/3*pi) - a/3

    # boundaries of minmal bounding rectangle:                  
    # we us the "normalized" density f(x) / f(xm). hence        
    # upper boundary: vmax = 1.                                 
    # left hand boundary: uminus = (y2-xm) * sqrt(f(y2)) / sqrt(f(xm)) 
    # right hand boundary: uplus = (y1-xm) * sqrt(f(y1)) / sqrt(f(xm)) 
    uplus  = (y1-xm) * exp(t*log(y1) - s*(y1 + 1/y1) - nc)
    uminus = (y2-xm) * exp(t*log(y2) - s*(y2 + 1/y2) - nc)

    # Generate a sample
    while true
        U = uminus + rand() * (uplus - uminus)  # U(u-,u+)
        V = rand()                              # U(0,vmax)
        X = U/V + xm
        if X > 0. && log(V) <= (t*log(X) - s*(X + 1/X) - nc)
            λ_old < 0 ? (return α / X) : (return α * X)
        end
    end
end

function rgig_ROU_noshift(λ, λ_old, β, α)
    # Ratio-of-uniforms without shift.                                          
    # Dagpunar (1988), Sect.~4.6.2                                            
    # Lehner (1989)     

    # shortcuts 
    t = 0.5 * (λ -1)
    s = 0.25 * β

    # mode = location of maximum of sqrt(f(x))
    xm = gig_mode(λ, β)

    # normalization constant: c = log(sqrt(f(xm)))
    nc = t*log(xm) - s*(xm + 1/xm)

    # location of maximum of x*sqrt(f(x))
    # we need the positive root of                 
    # omega/2*y^2 - (lambda+1)*y - omega/2 = 0 
    ym = ((λ+1) + sqrt((λ+1)^2 + β^2))/β

    # boundaries of minmal bounding rectangle:                   
    # we us the "normalized" density f(x) / f(xm). hence         
    # upper boundary: vmax = 1.                                  
    # left hand boundary: umin = 0.                              
    # right hand boundary: umax = ym * sqrt(f(ym)) / sqrt(f(xm)) 
    um = exp(0.5*(λ+1)*log(ym) - s*(ym + 1/ym) - nc)

    # Generate a sample
    while true
        U = um * rand()                    # U(0,umax)
        V = rand()                              # U(0,vmax)
        X = U/V
        if log(V) <= (t*log(X) - s*(X + 1/X) - nc)
            λ_old < 0 ? (return α / X) : (return α * X)
        end
    end
end


function rgig_newapproach1(λ, λ_old, β, α)
    # New approach, constant hat in log-concave part.                           */
    # Draw sample from GIG distribution.    
    # Hörmann and Leydold (2014)
    
    # mode = location of maximum of sqrt(f(x))
    xm = gig_mode(λ, β)

    # splitting point 
    x0 = β/(1-λ)

    # domain [0, x_0]
    k0 = exp((λ-1)*log(xm) - 0.5*β*(xm + 1/xm))     # = f(xm) 
    A0 = k0 * x0

    # domain [x_0, Infinity]
    if x0 >= 2/β
        k1 = 0
        A1 = 0
        k2 = x0^(λ-1)
        A2 = k2 * 2 * exp(-β*x0/2)/β
    else
        k1 = exp(-β)
        A1 = (λ == 0) ? k1 * log(2/(β^2)) : k1 / λ * ( (2/β)^(λ) - x0^λ)

        # domain [2/omega, Infinity]
        k2 = (2/β)^(λ-1)
        A2 = k2 * 2 * exp(-1)/β
    end


    # total area
    Atot = A0 + A1 + A2

    # Generate a sample
    while true
        # get uniform random number
        V = Atot * rand()

        # domain [0, x_0]
        if V <= A0
            X = x0 * V / A0
            hx = k0
        elseif V <= A0 + A1  # domain [x_0, 2/omega]
            V -= A0
            if λ == 0
                X = β * exp(exp(β)*V)
                hx = k1 / X
            else
                X = (x0^λ + (λ / k1 * V)) ^ (1/λ)
                hx = k1 * X^(λ-1)
            end
        else # domain [max(x0,2/omega), Infinity]
            V -= (A1+A0);
	        a = (x0 > 2/β) ? x0 : 2/β
            if exp(-β/2 * a) - β/(2*k2) * V <= 0
                println(A0)
                println(A1)
                println(A2)
                println(Atot)
                println(V)
            end

	        X = -2/β * log(exp(-β/2 * a) - β/(2*k2) * V)
	        hx = k2 * exp(-β/2 * X)
        end

        # accept or reject
        U = rand() * hx
        if log(U) <= (λ-1) * log(X) - β/2. * (X+1/X)
            (λ_old < 0.) ? (return α / X) : (return α * X)
        end

    end

end