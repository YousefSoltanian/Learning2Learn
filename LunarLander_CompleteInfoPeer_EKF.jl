# Restart your Julia session before running this script to avoid redefinition errors.

using iLQGames
using iLQGames:
    LinearSystem,
    AffineStrategy,
    QuadraticPlayerCost,
    LTVSystem,
    LQGame,
    SystemTrajectory,
    solve_lq_game!,
    n_states,
    n_controls,
    n_players,
    uindex,
    horizon,
    samplingtime,
    dynamics,
    strategytype,
    player_costs,
    trajectory!
using Plots
using LinearAlgebra, StaticArrays, ForwardDiff
using JLD2

# =============================================================================
# Problem Setup
# =============================================================================

const nx = 6
const nu = 2
const ΔT = 0.1
const T = 50
const simulation_horizon = 50

const true_xf = 5.0
const true_yf = -5.0

# We keep alpha_lr for compatibility with your original code, 
# but it won't actually be used in the EKF update.
const alpha_lr = 0.0100

global xf_estimate = 0.001
global yf_estimate = 0.001

# EKF tracking of parameter covariance:
global xf_var = 10.0
global yf_var = 10.0

# We’ll record the covariance for plotting:
global xf_var_history = Float64[]
global yf_var_history = Float64[]

# =============================================================================
# Dynamics Definitions
# =============================================================================

struct LunarLander_iLQR <: ControlSystem{ΔT, nx, nu} end

function iLQGames.dx(cs::LunarLander_iLQR, x, u, t)
    u1, u2 = u
    return SVector(
        x[4],
        x[5],
        x[6],
        u1*sin(x[3]),
        u1*cos(x[3]),
        u2
    )
end

struct LunarLander_Linear <: ControlSystem{ΔT, nx, nu} end

function dx(cs::LunarLander_Linear, x, u, t)
    u1, u2 = u
    return SVector(
        x[4],
        x[5],
        x[6],
        u1*sin(x[3]),
        u1*cos(x[3]),
        u2
    )
end

function discrete_dynamics(x, u)
    return x + ΔT * dx(LunarLander_Linear(), x, u, 0.0)
end

# =============================================================================
# Cost Functions
# =============================================================================

function cost_function_agent1(x, u1, xf_est, yf_est)
    return 10*(x[1]-xf_est)^2 +
           10*(x[2]-yf_est)^2 +
           10*x[4]^2 + 10*x[5]^2 +
           10*(x[3]-pi/2)^2 + 10*x[6]^2 +
           1*u1^2
end

function cost_function_agent2(x, u2, xf_est, yf_est)
    return 10*(x[1]-xf_est)^2 +
           10*(x[2]-yf_est)^2 +
           10*x[4]^2 + 10*x[5]^2 +
           10*(x[3]-pi/2)^2 + 10*x[6]^2 +
           1*u2^2
end

# =============================================================================
# iLQGames Solver (Nonlinear)
# =============================================================================

function solve_ilqr_game(x0, xf_est, yf_est, current_time)
    player_inputs = (SVector(1), SVector(2))
    local_horizon = max(2, T)

    costs = (
        FunctionPlayerCost((g, x, u, t) -> cost_function_agent1(x, u[1], xf_est, yf_est)),
        FunctionPlayerCost((g, x, u, t) -> cost_function_agent2(x, u[2], xf_est, yf_est))
    )

    g_temp = GeneralGame(local_horizon, player_inputs, LunarLander_iLQR(), costs)
    solver_temp = iLQSolver(g_temp; max_n_iter=100)
    _, trajectory, _ = solve(g_temp, solver_temp, x0)
    return trajectory
end

# =============================================================================
# Linearization + Quadratic Expansion
# =============================================================================

function linearize_dynamics_full_horizon(traj)
    N = length(traj.u)
    A = Vector{SMatrix{nx, nx}}(undef, N)
    B = Vector{SMatrix{nx, nu}}(undef, N)
    for t in 1:N
        x_t, u_t = traj.x[t], traj.u[t]
        A_ = ForwardDiff.jacobian(z -> discrete_dynamics(z, u_t), x_t)
        B_ = ForwardDiff.jacobian(v -> discrete_dynamics(x_t, v), u_t)
        A[t] = SMatrix{nx, nx}(A_)
        B[t] = SMatrix{nx, nu}(B_)
    end
    return A, B
end

# =============================================================================
# Building the LQGame
# =============================================================================

function build_lq_game(A, B, cost_by_step)
    N = length(A)
    dyns = map(k -> LinearSystem{ΔT}(SMatrix{nx, nx}(A[k]), SMatrix{nx, nu}(B[k])), 1:N)
    ltv = LTVSystem(SizedVector{N}(dyns))

    c_array = map(k -> SVector(cost_by_step[k]...), 1:N)
    qtv_costs = SizedVector{N}(c_array)

    player_u = (SVector(1), SVector(2))
    g_lq = LQGame(player_u, ltv, qtv_costs)
    return g_lq
end

# =============================================================================
# The snippet solver: solve_lq_game_FBNE!
# =============================================================================

function solve_lq_game_FBNE!(strategies, g::LQGame)
    nx = n_states(g)
    nu = n_controls(g)

    Z = [pc.Q for pc in last(player_costs(g))]
    ζ = [pc.l for pc in last(player_costs(g))]

    # Let's pick up a type from the first step's data for safe typed storage:
    dyn = dynamics(g)[1]
    cost₁ = player_costs(g)[1][1]
    BᵢZᵢB_typ = typeof((dyn.B[:,1]' * Z[1] * dyn.B)[1,1])
    Bᵢζ_typ   = typeof((dyn.B[:,1]' * ζ[1]))

    S = MMatrix{nu, nu, BᵢZᵢB_typ}(undef)
    YP = MMatrix{nu, nx, BᵢZᵢB_typ}(undef)
    Yα = MVector{nu, Bᵢζ_typ}(undef)

    for kk in horizon(g):-1:1
        dyn = dynamics(g)[kk]
        cost = player_costs(g)[kk]

        A = dyn.A
        B = dyn.B

        for (ii, udxᵢ) in enumerate(uindex(g))
            BᵢZᵢ = B[:, udxᵢ]' * Z[ii]
            @inbounds S[udxᵢ, :] = cost[ii].R[udxᵢ, :] + (BᵢZᵢ * B)
            @inbounds YP[udxᵢ, :] = BᵢZᵢ * A
            @inbounds Yα[udxᵢ]   = B[:, udxᵢ]'*ζ[ii] + cost[ii].r[udxᵢ]
        end

        Sinv = inv(SMatrix(S))
        P = Sinv * SMatrix(YP)
        α = Sinv * SVector(Yα)

        F = A - B * P
        β = -B * α

        for ii in 1:n_players(g)
            cᵢ = cost[ii]
            PRᵢ = P' * cᵢ.R
            ζ[ii] = F'*(ζ[ii] + Z[ii]*β) + cᵢ.l + PRᵢ*α - P'*cᵢ.r
            Z[ii] = F'*Z[ii]*F + cᵢ.Q + PRᵢ*P
        end

        strategies[kk] = AffineStrategy(P, α)
    end
end

# =============================================================================
# Our "predicted control" using that snippet
# =============================================================================

function predicted_control_other_full_horizon(theta, agent_id, traj, xf_est, yf_est)
    local_xf = (agent_id == 1) ? theta : xf_est
    local_yf = (agent_id == 2) ? theta : yf_est
    A, B = linearize_dynamics_full_horizon(traj)
    cost_by_step = cost_expansion_full_horizon_to_QPlayerCosts(traj, A, B, local_xf, local_yf)

    g_lq = build_lq_game(A, B, cost_by_step)

    local_horizon = length(A)
    strategies = Vector{AffineStrategy}(undef, local_horizon)

    solve_lq_game_FBNE!(strategies, g_lq)

    alpha_vec = strategies[1].α

    # If agent_id=1 => "other" agent is #2 => return alpha_vec[2]
    # If agent_id=2 => "other" agent is #1 => return alpha_vec[1]
    return (agent_id == 1) ? alpha_vec[2] : alpha_vec[1]
end

function cost_expansion_full_horizon_to_QPlayerCosts(traj, A, B, xf_est, yf_est)
    N = length(A)
    costs_each_step = Vector{SVector{2,QuadraticPlayerCost}}(undef, N)
    for t in 1:N
        x_t, u_t = traj.x[t], traj.u[t]

        Q1 = @SMatrix([10.0 0.0 0.0 0.0 0.0 0.0;
                       0.0 10.0 0.0 0.0 0.0 0.0;
                       0.0 0.0 10.0 0.0 0.0 0.0;
                       0.0 0.0 0.0 10.0 0.0 0.0;
                       0.0 0.0 0.0 0.0 10.0 0.0;
                       0.0 0.0 0.0 0.0 0.0 10.0])
        R1 = @SMatrix [1.0 0.0; 0.0 0.0]
        l1_x = SVector(
            20*(x_t[1]-xf_est),
            20*(x_t[2]-yf_est),
            20*(x_t[3]-pi/2),
            20*x_t[4],
            20*x_t[5],
            20*x_t[6]
        )
        l1_u = @SVector [2.0*u_t[1], 0.0]
        pc1 = QuadraticPlayerCost(l1_x, Q1, l1_u, R1)

        Q2 = Q1
        R2 = @SMatrix [0.0 0.0; 0.0 1.0]
        l2_x = l1_x
        l2_u = @SVector [0.0, 2.0*u_t[2]]
        pc2 = QuadraticPlayerCost(l2_x, Q2, l2_u, R2)

        costs_each_step[t] = SVector(pc1, pc2)
    end

    return costs_each_step
end

# =============================================================================
# Parameter Update (EKF style, but still printing param/error/grad)
# =============================================================================

function update_estimate_full_horizon(traj, xf_est, yf_est, agent_id, observed_u_other)
    # -------------------------------------------------------------------------
    # Old definitions:
    #     param   = (agent_id == 1) ? xf_est : yf_est
    #     error   = 1*observed_u_other - (the predicted control at t=1 from other agent)
    #     grad    = derivative of predicted_control_other_full_horizon wrt param
    #     updated_param = param - eta*alpha_lr*(error*grad)
    #
    # Now we do the EKF approach, but preserve the same printing.
    # -------------------------------------------------------------------------
    param = (agent_id == 1) ? xf_est : yf_est
    param_var = (agent_id == 1) ? xf_var : yf_var

    # The "observed" control is observed_u_other
    # The "predicted" observation is what we get from param:
    predicted_u_other = predicted_control_other_full_horizon(param, agent_id, traj, xf_est, yf_est)

    # For printing consistency, we define the "error" as in your original code:
    u_pre = (agent_id == 1) ? traj.u[1][2] : traj.u[1][1]
    error = 1*observed_u_other - u_pre

    # The gradient is the derivative wrt param:
    grad = ForwardDiff.derivative(
        θ -> predicted_control_other_full_horizon(θ, agent_id, traj, xf_est, yf_est),
        param
    )

    # We'll compute the EKF update for scalar observation:
    #   y_obs = observed_u_other
    #   y_pred= predicted_u_other
    #   H = grad (a scalar)
    #   R = I => scalar 1.0
    #   K = param_var * H / (1 + H^2*param_var)
    #   param_new = param + K*(y_obs - y_pred)
    #   var_new   = param_var - K*H*param_var
    denom = 1 + (grad^2)*param_var
    K = (0.000+param_var) * grad / denom
    updated_param = param - K*(observed_u_other - u_pre)
    updated_param_var = param_var - 1*K*grad*param_var
    if updated_param_var <0.01
        updated_param_var = 0.01
    end

    # Print to match your original style, but now with more info
    println("Agent $agent_id: param=$param, param_var=$param_var, error=$error, grad=$grad, K=$K")
    println("             updated_param=$updated_param, updated_param_var=$updated_param_var")

    # Store updated covariance
    if agent_id == 1
        global xf_var = updated_param_var
    else
        global yf_var = updated_param_var
    end

    return updated_param, error, grad
end

# =============================================================================
# Controls
# =============================================================================

function agent1_control(x, xf_est, yf_est, t)
    traj = solve_ilqr_game(x, xf_est, yf_est, t)
    return traj.u[1][1], traj
end

function agent2_control(x, xf_est, yf_est, t)
    traj = solve_ilqr_game(x, xf_est, yf_est, t)
    return traj.u[1][2], traj
end

# =============================================================================
# Main Simulation
# =============================================================================

x0 = SVector(-5.0, 8.0, 0, 0.0, 0.0, 0.0)
x_history = [x0]
u_history = []
xf_history = Float64[]
yf_history = Float64[]

for k in 0:simulation_horizon-1

    push!(xf_history, xf_estimate)
    push!(yf_history, yf_estimate)
    push!(xf_var_history, xf_var)
    push!(yf_var_history, yf_var)
    u1, traj1 = agent1_control(x_history[end], xf_estimate, true_yf, k)
    u2, traj2 = agent2_control(x_history[end], true_xf, yf_estimate, k)

    updated_xf, error1, grad1 = update_estimate_full_horizon(traj1, xf_estimate, true_yf, 1, u2)
    updated_yf, error2, grad2 = update_estimate_full_horizon(traj2, true_xf, yf_estimate, 2, u1)

    global xf_estimate = updated_xf
    global yf_estimate = updated_yf

    u_now = (u1, u2)
    push!(u_history, u_now)

    x_next = x_history[end] + ΔT * dx(LunarLander_Linear(), x_history[end], u_now, 0.0)
    push!(x_history, x_next)
end

using Plots
gr()

u1_vals = [u[1] for u in u_history]
u2_vals = [u[2] for u in u_history]

layout = @layout [a; b]

param_plot = plot(
    1:simulation_horizon, xf_history,
    label="xf_est",
    xlabel="Time Step",
    ylabel="Estimate Value",
    title="Parameter Estimates (EKF-Style) Over Time",
    legend=:bottomright,
    linewidth=2,
    color=:blue
)
plot!(param_plot, 1:simulation_horizon, yf_history, label="yf_est", linewidth=2, color=:red)

# Add ±2σ shading for xf
plot!(
    param_plot,
    1:simulation_horizon,
    xf_history .+ 2 .* sqrt.(xf_var_history),
    fillrange = xf_history .- 2 .* sqrt.(xf_var_history),
    fillalpha = 0.2,
    linecolor=:blue, fillcolor=:blue,
    label=false
)
# Add ±2σ shading for yf
plot!(
    param_plot,
    1:simulation_horizon,
    yf_history .+ 2 .* sqrt.(yf_var_history),
    fillrange = yf_history .- 2 .* sqrt.(yf_var_history),
    fillalpha = 0.2,
    linecolor=:red, fillcolor=:red,
    label=false
)

# Show true parameters as dashed black lines
plot!(param_plot, 1:simulation_horizon, fill(true_xf, simulation_horizon),
    color=:black, linestyle=:dash, label="true xf"
)
plot!(param_plot, 1:simulation_horizon, fill(true_yf, simulation_horizon),
    color=:black, linestyle=:dash, label="true yf"
)

control_plot = plot(
    1:simulation_horizon, u1_vals,
    label="Control U1",
    xlabel="Time Step",
    ylabel="Control Input",
    title="Control Inputs Over Time",
    legend=:topright,
    linewidth=2,
    color=:green
)
plot!(control_plot, 1:simulation_horizon, u2_vals, label="Control U2", linewidth=2, color=:orange)

combined_plot = plot(param_plot, control_plot, layout=layout)
display(combined_plot)

anim = @animate for i in 1:simulation_horizon
    x_vals = [x_history[j][1] for j in 1:i]
    y_vals = [x_history[j][2] for j in 1:i]
    plot(x_vals, y_vals,
         xlims=(-6,8), ylims=(-6,8),
         xlabel="X Position", ylabel="Y Position",
         aspect_ratio=:equal, legend=false,
         title="Lander Trajectory")

    x_i = x_history[i][1]
    y_i = x_history[i][2]
    theta_i = x_history[i][3]
    lander_x = [x_i, x_i + 0.5*cos(theta_i)]
    lander_y = [y_i, y_i + 0.5*sin(theta_i)]
    plot!(lander_x, lander_y, color=:black, linewidth=2)
end

gif(anim, "Julia Projects/lander_trajectory.gif", fps=15)

@save "Julia Projects/LunarLander_Complete_Info_Peer.jld2" xf_history yf_history xf_var_history yf_var_history x_history u_history

println("Simulation data saved to data_PACE.jld2")