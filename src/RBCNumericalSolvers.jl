module RBCNumericalSolvers

using LinearAlgebra
using Statistics

export solve_vfi, solve_egm, interpolate_policy, test_euler_mse

utility(c, p) = c > 0 ? (c^(1 - p.gamma) - 1) / (1 - p.gamma) : -Inf
uprime(c, p) = c^(-p.gamma)
resources(k, z, p) = exp(z) * k^p.alpha + (1 - p.delta) * k
gross_return(k, z, p) = p.alpha * exp(z) * k^(p.alpha - 1) + 1 - p.delta

function rouwenhorst(n::Int, rho::Float64, sigma_eps::Float64)
    n >= 2 || error("Rouwenhorst requires at least two states")
    q = (1 + rho) / 2
    P = [q 1-q; 1-q q]
    for m in 3:n
        Pold = P
        P = zeros(m, m)
        P[1:end-1, 1:end-1] .+= q .* Pold
        P[1:end-1, 2:end] .+= (1 - q) .* Pold
        P[2:end, 1:end-1] .+= (1 - q) .* Pold
        P[2:end, 2:end] .+= q .* Pold
        P[2:end-1, :] ./= 2
    end
    stationary_sd = sigma_eps / sqrt(1 - rho^2)
    zmax = stationary_sd * sqrt(n - 1)
    return collect(range(-zmax, zmax, length=n)), P
end

function linear_interp(x, y, xq)
    xq <= x[1] && return y[1]
    xq >= x[end] && return y[end]
    j = searchsortedlast(x, xq)
    weight = (xq - x[j]) / (x[j + 1] - x[j])
    return (1 - weight) * y[j] + weight * y[j + 1]
end

function interpolate_policy(result, k, z=0.0)
    iz = argmin(abs.(result.zgrid .- z))
    return linear_interp(result.kgrid, view(result.policy_kp, :, iz), k)
end

function grid_policy(policy_kp, kgrid, zgrid, p)
    return function (k, z)
        iz = argmin(abs.(zgrid .- z))
        kp = linear_interp(kgrid, view(policy_kp, :, iz), k)
        c = resources(k, z, p) - kp
        return c, kp
    end
end

function test_euler_mse(policy::Function, k_test, zgrid, P, p)
    squared_residuals = Float64[]
    for iz in eachindex(zgrid), k in k_test
        c, kp = policy(k, zgrid[iz])
        if !(isfinite(c) && isfinite(kp) && c > 0 && kp > 0)
            continue
        end

        conditional_residual = 0.0
        valid = true
        for izp in eachindex(zgrid)
            cp, _ = policy(kp, zgrid[izp])
            if !(isfinite(cp) && cp > 0)
                valid = false
                break
            end
            q = p.beta * uprime(cp, p) * gross_return(kp, zgrid[izp], p) / uprime(c, p) - 1
            conditional_residual += P[iz, izp] * q
        end
        valid && push!(squared_residuals, conditional_residual^2)
    end
    isempty(squared_residuals) && return Inf
    return mean(squared_residuals)
end

function test_euler_mse(policy_kp::AbstractMatrix, kgrid, zgrid, P, p; k_test=kgrid)
    return test_euler_mse(grid_policy(policy_kp, kgrid, zgrid, p), k_test, zgrid, P, p)
end

function greedy_policy(V, kgrid, zgrid, P, p)
    nk, nz = length(kgrid), length(zgrid)
    expected_value = V * transpose(P)
    policy_idx = ones(Int, nk, nz)
    Vnew = fill(-Inf, nk, nz)
    for iz in 1:nz, ik in 1:nk
        wealth = resources(kgrid[ik], zgrid[iz], p)
        best_value = -Inf
        best_idx = 1
        for ikp in 1:nk
            c = wealth - kgrid[ikp]
            c <= 0 && break
            candidate = utility(c, p) + p.beta * expected_value[ikp, iz]
            if candidate > best_value
                best_value = candidate
                best_idx = ikp
            end
        end
        Vnew[ik, iz] = best_value
        policy_idx[ik, iz] = best_idx
    end
    return Vnew, policy_idx
end

policy_levels(policy_idx, kgrid) = reshape(kgrid[vec(policy_idx)], size(policy_idx))

function solve_vfi(p, ss; nk=180, nz=5, k_min=0.40 * ss.K, k_max=1.60 * ss.K,
        tol=1e-7, max_iter=1_000, record_history=false)
    kgrid = collect(range(k_min, k_max, length=nk))
    zgrid, P = rouwenhorst(nz, p.rho_z, p.sigma_z)
    V = zeros(nk, nz)
    policy_idx = ones(Int, nk, nz)
    policy_history = Matrix{Float64}[]
    iteration_history = Int[]
    time_history = Float64[]
    elapsed_time = 0.0

    for iteration in 1:max_iter
        iteration_start = time_ns()
        Vnew, policy_idx = greedy_policy(V, kgrid, zgrid, P, p)
        value_diff = maximum(abs.(Vnew .- V))
        V = Vnew
        policy_kp = policy_levels(policy_idx, kgrid)
        elapsed_time += (time_ns() - iteration_start) / 1e9
        if record_history
            push!(policy_history, copy(policy_kp))
            push!(iteration_history, iteration)
            push!(time_history, elapsed_time)
        end
        if value_diff < tol
            return (; V, policy_kp, kgrid, zgrid, P, iteration, value_diff,
                policy_history, iteration_history, time_history)
        end
    end
    error("VFI did not converge in $(max_iter) iterations")
end

function inverse_resources(wealth, z, p, lower, upper)
    resources(lower, z, p) >= wealth && return lower
    resources(upper, z, p) <= wealth && return upper
    lo, hi = lower, upper
    for _ in 1:60
        mid = (lo + hi) / 2
        if resources(mid, z, p) < wealth
            lo = mid
        else
            hi = mid
        end
    end
    return (lo + hi) / 2
end

function initial_policy(kgrid, zgrid, p, ss)
    savings_share = ss.K / ss.W
    policy_kp = Matrix{Float64}(undef, length(kgrid), length(zgrid))
    for iz in eachindex(zgrid), ik in eachindex(kgrid)
        wealth = resources(kgrid[ik], zgrid[iz], p)
        policy_kp[ik, iz] = clamp(savings_share * wealth, first(kgrid), min(last(kgrid), wealth - 1e-10))
    end
    return policy_kp
end

function policy_consumption(policy_kp, kgrid, zgrid, p)
    c = similar(policy_kp)
    for iz in eachindex(zgrid), ik in eachindex(kgrid)
        c[ik, iz] = resources(kgrid[ik], zgrid[iz], p) - policy_kp[ik, iz]
    end
    return c
end

function egm_update(policy_kp, kgrid, zgrid, P, p)
    nk, nz = size(policy_kp)
    consumption_next = policy_consumption(policy_kp, kgrid, zgrid, p)
    policy_new = similar(policy_kp)
    kendo = Vector{Float64}(undef, nk)

    for iz in 1:nz
        for ikp in 1:nk
            kp = kgrid[ikp]
            expected_marginal_value = 0.0
            for izp in 1:nz
                cp = consumption_next[ikp, izp]
                expected_marginal_value += P[iz, izp] * uprime(cp, p) * gross_return(kp, zgrid[izp], p)
            end
            c = (p.beta * expected_marginal_value)^(-1 / p.gamma)
            kendo[ikp] = inverse_resources(c + kp, zgrid[iz], p, 1e-8, 3 * last(kgrid))
        end

        order = sortperm(kendo)
        sorted_kendo = kendo[order]
        sorted_kp = kgrid[order]
        for ik in 1:nk
            wealth = resources(kgrid[ik], zgrid[iz], p)
            kp = kgrid[ik] <= sorted_kendo[1] ? first(kgrid) :
                linear_interp(sorted_kendo, sorted_kp, kgrid[ik])
            policy_new[ik, iz] = clamp(kp, first(kgrid), min(last(kgrid), wealth - 1e-10))
        end
    end
    return policy_new
end

function solve_egm(p, ss; nk=180, nz=5, k_min=0.40 * ss.K, k_max=1.60 * ss.K,
        tol=1e-8, max_iter=1_000, damping=0.9, record_history=false)
    kgrid = collect(range(k_min, k_max, length=nk))
    zgrid, P = rouwenhorst(nz, p.rho_z, p.sigma_z)
    policy_kp = initial_policy(kgrid, zgrid, p, ss)
    policy_history = Matrix{Float64}[]
    iteration_history = Int[]
    time_history = Float64[]
    elapsed_time = 0.0
    if record_history
        push!(policy_history, copy(policy_kp))
        push!(iteration_history, 0)
        push!(time_history, 0.0)
    end

    for iteration in 1:max_iter
        iteration_start = time_ns()
        update = egm_update(policy_kp, kgrid, zgrid, P, p)
        policy_new = damping .* update .+ (1 - damping) .* policy_kp
        policy_diff = maximum(abs.(policy_new .- policy_kp))
        policy_kp = policy_new
        elapsed_time += (time_ns() - iteration_start) / 1e9
        if record_history
            push!(policy_history, copy(policy_kp))
            push!(iteration_history, iteration)
            push!(time_history, elapsed_time)
        end
        if policy_diff < tol
            return (; policy_kp, kgrid, zgrid, P, iteration, policy_diff,
                policy_history, iteration_history, time_history)
        end
    end
    error("EGM did not converge in $(max_iter) iterations")
end

end
