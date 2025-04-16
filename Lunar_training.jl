###############################################################################
# TRAINING SCRIPT EXAMPLE FOR LUNAR LANDER "HUMAN" BELIEF
###############################################################################
using Flux
using Flux: gradient, Optimiser
using Zygote
using LinearAlgebra
using StaticArrays
using JLD2
using iLQGames
using iLQGames: FunctionPlayerCost, GeneralGame, iLQSolver, solve, ControlSystem
###############################################################################
# We assume you have the dynamics, cost, iLQ solver definitions 
# similar to your original code. We'll define only minimal stubs here 
# to show the full training pipeline.
###############################################################################

############################
# 1) LOAD THE 10K SIMS DATA
############################
println("Loading dataset from 'CoRL_Code/LunarLander_Complete_Info_Peer.jld2'...")
f = jldopen("CoRL_Code/LunarLander_Complete_Info_Peer.jld2", "r")
all_sims = read(f, "all_simulations")  # an Array of Dict, length ~10,000
close(f)

println("Loaded $(length(all_sims)) simulated trajectories.")
# each entry in `all_sims` has:
#   "x_history" => Array of SVector{6} (length T+1 = 51)
#   "u_history" => Array of Tuple{Float64, Float64} (length T=50)
#   plus "true_xf", "true_yf", etc.

const T = 50  # time horizon

############################
# 2) DEFINING DYNAMICS
############################
# We'll define your LunarLander control system and 
# a short-horizon iLQ solve function to "predict" controls 
# given the current state x_t, a guess for x_f, and a guess y_f
# (the latter is output by the transformer's dynamic belief).

struct LunarLanderILQ <: ControlSystem{0.1,6,2} end
function iLQGames.dx(cs::LunarLanderILQ, x::SVector{6,Float64}, u::SVector{2,Float64}, t::Float64)
    # torque = u[2], thrust = u[1]  or vice-versa, 
    # but let's match your code carefully: 
    # "the human is controlling torque which is u2"
    # We'll treat u1 as thrust, u2 as torque. 
    u1 = u[1]  # thrust
    u2 = u[2]  # torque
    return SVector(
        x[4],
        x[5],
        x[6],
        u1*sin(x[3]),
        u1*cos(x[3]),
        u2
    )
end

# We'll define a single-step "predict_control" that does 
# a short horizon iLQ solve given cost fns that depend on (x_f, y_f).

function cost_func_agent1(x, u1, x_f, y_f)
    # similar to your code:
    return 10*(x[1]-x_f)^2 + 
           10*(x[2]-y_f)^2 + 
           10*x[4]^2 + 10*x[5]^2 +
           10*(x[3]-pi/2)^2 + 10*x[6]^2 +
           1*u1^2
end

function cost_func_agent2(x, u2, x_f, y_f)
    return 10*(x[1]-x_f)^2 + 
           10*(x[2]-y_f)^2 + 
           10*x[4]^2 + 10*x[5]^2 +
           10*(x[3]-pi/2)^2 + 10*x[6]^2 +
           1*u2^2
end

function solve_ilq_singlestep(x₀::SVector{6,Float64}, x_f::Float64, y_f::Float64; 
                              horizon_len=5)
    # We'll define a small local horizon = horizon_len
    # Then define the cost for each agent:
    player_inputs = (SVector(1), SVector(2))  # agent1 controls u1, agent2 controls u2
    costs = (
        FunctionPlayerCost((g, x, u, t) -> cost_func_agent1(x, u[1], x_f, y_f)),
        FunctionPlayerCost((g, x, u, t) -> cost_func_agent2(x, u[2], x_f, y_f))
    )

    g_temp = GeneralGame(horizon_len, player_inputs, LunarLanderILQ(), costs)
    solver = iLQSolver(g_temp; max_n_iter=30)
    _, traj, _ = solve(g_temp, solver, x₀)
    # We only want the 1st action from traj.u[1], but let's just return it:
    return traj.u[1]  # a SVector{2,Float64}, i.e. (u1_pred, u2_pred)
end

############################
# 3) DEFINING THE BELIEF TRANSFORMER
############################
# We'll do a small module that:
#  - Takes partial history of size t (states x0..x_{t-1}, controls u0..u_{t-1})
#  - Outputs a single Float y_f_pred
using Flux: Chain, Dense, relu

const EMBED_DIM = 16

# We'll embed each (x_k, u_k) => R^EMBED_DIM => average => MLP => scalar
function embed_obs(x::SVector{6,Float64}, u::Tuple{Float64,Float64})
    # 6 + 2 = 8 input
    v_in = vcat(x, SVector{2}(u...))
    return v_in
end

struct BeliefTransformer
    embed_layer::Chain
    aggregator::Chain
    output_layer::Chain
end

function BeliefTransformer()
    embed = Chain(
        Dense(8, EMBED_DIM, relu),
        Dense(EMBED_DIM, EMBED_DIM, relu)
    )
    # aggregator => naive approach: sum or mean
    agg = Chain(
        Dense(EMBED_DIM, EMBED_DIM, relu),
        Dense(EMBED_DIM, EMBED_DIM)
    )
    out = Chain(
        Dense(EMBED_DIM, EMBED_DIM, relu),
        Dense(EMBED_DIM, 1)  # final => scalar
    )
    return BeliefTransformer(embed, agg, out)
end

function (bt::BeliefTransformer)(x_hist::Vector{SVector{6,Float64}}, 
                                 u_hist::Vector{Tuple{Float64,Float64}} )
    # For t steps of data, we have t states, t controls? 
    # Actually from 0..t-1 => t states, t-1 controls, depending on indexing.
    # We'll be consistent: we assume length(x_hist) == length(u_hist) + 1
    # We'll embed each step i in 0..(t-1) using (x_hist[i], u_hist[i]) 
    # except for the final state which has no matching control
    # Then sum/mean => aggregator => output
    Tt = length(u_hist)  # number of controls
    if Tt == 0
        # no data => return 0.0
        return 0.0
    end
    embs = Vector{Vector{Float64}}(undef, Tt)
    for i in 1:Tt
        v_in = embed_obs(x_hist[i], u_hist[i])    # 8-d
        embs[i] = bt.embed_layer(v_in)            # EMBED_DIM
    end
    # naive aggregator => average them => pass aggregator
    E = zero(embs[1])
    for i in 1:Tt
        E .+= embs[i]
    end
    E ./= Tt

    A = bt.aggregator(E)     # EMBED_DIM
    outvec = bt.output_layer(A)  # => scalar dimension
    return outvec[1]
end

bt_model = BeliefTransformer()
println("BeliefTransformer params count = ", length(Flux.params(bt_model)))

############################
# 4) PER-TRAJECTORY x_f
############################
# Each sim i has a static param x_f^(i). We'll store them in an array 
# and treat them as trainable. 
mutable struct XfContainer
    xf::Vector{Float64}
end

const N = length(all_sims)
xf_container = XfContainer( [0.0 for _ in 1:N] )

# We'll define a function that returns the trainable arrays combined
function all_params(bt::BeliefTransformer, xfcont::XfContainer)
    return Flux.Params([Flux.params(bt); xfcont.xf])
end

############################
# 5) LOSS FOR ONE TRAJECTORY
############################
function trajectory_loss_single(i::Int; 
                                bt::BeliefTransformer,
                                xfcont::XfContainer,
                                derivative_weight = 0.0
                               )
    sim = all_sims[i]
    x_hist = sim["x_history"]  # length T+1 => x_hist[1..51]
    u_hist = sim["u_history"]  # length T=50
    # We'll define b^h_t = bt_model( x_hist[1..t], u_hist[1..t] ), 
    # but watch indexing carefully:
    # We'll do step t in 1..T => partial history: x_hist[1..t], u_hist[1..(t-1)] 
    # THEN we solve iLQ with x_f^i and b^h_t => get predicted controls => compare to u_hist[t].
    x_fi = xfcont.xf[i]

    local_loss = 0.0
    # We'll also keep track of consecutive controls for derivative mismatch if derivative_weight>0
    for t in 1:T
        # partial history length => t-1 controls, t states
        # if t==1 => length(u_hist[1..0]) =0 => safe 
        x_part = x_hist[1:t]                # t states
        u_part = u_hist[1:t-1]             # t-1 controls
        # compute the dynamic belief:
        y_f_t = bt_model(x_part, u_part)   # or (bt_model).(some?), we do direct call

        # now we do iLQ with small horizon=5:
        x_t = x_part[end]   # the current state
        predicted_u = solve_ilq_singlestep(x_t, x_fi, y_f_t; horizon_len=5)
        # predicted_u => (u1_pred, u2_pred)

        # compare with actual:
        actual_u = u_hist[t]
        local_loss += (predicted_u[1] - actual_u[1])^2
        local_loss += (predicted_u[2] - actual_u[2])^2

        # optionally do the derivative mismatch:
        if (derivative_weight>0.0) && (t < T)
            # we'll do naive: Δu_actual = (u_hist[t+1] - u_hist[t]) 
            # the chain rule from partial derivatives would require more complex code,
            # but let's do a simpler derivative mismatch for demonstration.
            Δu_act = SVector(u_hist[t+1]...) .- SVector(u_hist[t]...)
            # We'll define a predicted derivative if we had time to do chain rule. 
            # For now, let's do predicted_u_{t+1} - predicted_u_t => we just solve again for t+1 => 
            # but that means 2 calls per step => too big. We'll do a naive approach:
            Δu_pred = zero(Δu_act) # placeholder
            # ignoring chain rule. 
            local_loss += derivative_weight * sum((Δu_act - Δu_pred).^2)
        end
    end
    return local_loss
end

function total_loss(bt::BeliefTransformer,
                    xfcont::XfContainer;
                    derivative_weight=0.0,
                    batchsize=32)
    # pick random subset
    idxs = rand(1:N, batchsize)
    sumloss = 0.0
    for i in idxs
        sumloss += trajectory_loss_single(i; 
                                          bt=bt, 
                                          xfcont=xfcont, 
                                          derivative_weight=derivative_weight)
    end
    return sumloss / batchsize
end

############################
# 6) TRAINING LOOP
############################
opt = Descent(1e-4)   # or ADAM(1e-4)
allparams = all_params(bt_model, xf_container)

n_epochs = 3
deriv_weight = 0.1  # example for derivative mismatch weighting

for epoch in 1:n_epochs
    @show epoch
    gs = gradient(allparams) do
        lval = total_loss(bt_model, xf_container; 
                          derivative_weight=deriv_weight, 
                          batchsize=8)
        return lval
    end

    # update all transformer's and x_f's parameters
    for p in allparams
        Flux.Optimise.update!(opt, p, gs[p])
    end

    # Evaluate with bigger batch or entire set
    train_loss = total_loss(bt_model, xf_container; 
                            derivative_weight=deriv_weight, 
                            batchsize=16)
    println("  epoch $epoch => train_loss=$train_loss")
end

println("Done training. x_f first few = ", xf_container.xf[1:5])
println("Inspect or save your model parameters, etc.")
