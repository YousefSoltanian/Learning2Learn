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

# The true final destinations will be set randomly for each simulation.
# We keep alpha_lr for compatibility with your original code, but it won't actually be used in the EKF update.
const alpha_lr = 0.0100

# Global variables for estimation and covariance (will be reinitialized for each simulation)
global xf_estimate = 0.001
global yf_estimate = 0.001
global xf_var = 10.0
global yf_var = 10.0

# For recording covariance (only used inside each simulation trajectory)
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
    param = (agent_id == 1) ? xf_est : yf_est
    param_var = (agent_id == 1) ? xf_var : yf_var

    predicted_u_other = predicted_control_other_full_horizon(param, agent_id, traj, xf_est, yf_est)
    u_pre = (agent_id == 1) ? traj.u[1][2] : traj.u[1][1]
    error = 1*observed_u_other - u_pre

    grad = ForwardDiff.derivative(
        θ -> predicted_control_other_full_horizon(θ, agent_id, traj, xf_est, yf_est),
        param
    )

    denom = 1 + (grad^2)*param_var
    K = (0.000+param_var) * grad / denom
    updated_param = param - K*(observed_u_other - u_pre)
    updated_param_var = param_var - 1*K*grad*param_var
    if updated_param_var < 0.01
        updated_param_var = 0.01
    end

    #println("Agent $agent_id: param=$param, param_var=$param_var, error=$error, grad=$grad, K=$K")
    #println("             updated_param=$updated_param, updated_param_var=$updated_param_var")

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
# Main Simulation Loop for 1000 Trajectories
# =============================================================================

all_simulations = []    # For full trajectory and control data
all_estimations = []    # For storing the final estimation info (4 values per simulation)

for sim in 1:10000
    println("Starting simulation $sim")
    # Set random final destinations for this simulation (uniformly in [-10, 10])
    global true_xf = rand()*20 - 10
    global true_yf = rand()*20 - 10

    # Reinitialize estimation and covariance variables
    global xf_estimate = 0.001
    global yf_estimate = 0.001
    global xf_var = 10.0
    global yf_var = 10.0
    global xf_var_history = Float64[]
    global yf_var_history = Float64[]

    # Randomize initial state:
    # x position in [-10, 10], y position in [5, 15],
    # theta in [-π/4, π/4], x velocity in [-1, 1],
    # y velocity in [-1, 1], theta velocity in [-0.5, 0.5]
    x0 = SVector(rand()*20 - 10,
                 rand()*10 + 5,
                 rand()*(pi/2) - (pi/4),
                 rand()*2 - 1,
                 rand()*2 - 1,
                 rand()*1 - 0.5)

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

    # Extract robot (torque control: u1) and human (thrust control: u2) controls
    robot_controls = [u[1] for u in u_history]
    human_controls = [u[2] for u in u_history]

    # Save this simulation's full data in a dictionary
    sim_data = Dict(
        "simulation_index" => sim,
        "true_xf" => true_xf,
        "true_yf" => true_yf,
        "x_history" => x_history,
        "u_history" => u_history,
        "robot_control" => robot_controls,
        "human_control" => human_controls,
        "xf_history" => xf_history,
        "yf_history" => yf_history,
        "xf_var_history" => xf_var_history,
        "yf_var_history" => yf_var_history
    )
    push!(all_simulations, sim_data)

    # Save final estimation info.
    # Here, agent1 updates xf_estimate (its own parameter estimate) and agent2 updates yf_estimate.
    # We record: agent1_true = true_xf, agent1_estimation = final xf_estimate,
    #             agent2_true = true_yf, agent2_estimation = final yf_estimate.
    est_data = Dict(
        "simulation_index" => sim,
        "agent1_true" => true_xf,
        "agent1_estimation" => xf_estimate,
        "agent2_true" => true_yf,
        "agent2_estimation" => yf_estimate
    )
    push!(all_estimations, est_data)
end

# =============================================================================
# Save all trajectories to one file and the estimation info to another file
# =============================================================================

@save "data/LunarLander_Complete_Info_Peer.jld2" all_simulations
@save "data/LunarLander_Estimations.jld2" all_estimations

println("Simulation data saved to:")
println("  - data/LunarLander_Complete_Info_Peer.jld2")
println("  - data/LunarLander_Estimations.jld2")
