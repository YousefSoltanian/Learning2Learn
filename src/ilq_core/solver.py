# solver.py
import numpy as np
from .expansions import linearize_dynamics_full_horizon

def cost_expansion_full_horizon(traj, A, B, xf_est, yf_est):
    """
    Minimally build cost expansions for each step => (pc1, pc2).
    We'll skip HPC details. 
    """
    N = len(traj["u"])
    csteps = []
    for t in range(N):
        pc1 = {"Q": np.eye(6)*10, "R": np.eye(2), "l":np.zeros(6), "r":np.zeros(2)}
        pc2 = {"Q": np.eye(6)*10, "R": np.eye(2), "l":np.zeros(6), "r":np.zeros(2)}
        csteps.append((pc1, pc2))
    return csteps

def solve_lq_game_fbne(A, B, cost_steps):
    """
    A skeleton solver that returns feedforward alpha=0 for each step 
    for demonstration. 
    """
    N = len(A)
    strategies = []
    for k in range(N):
        alpha = np.array([0.0, 0.0], dtype=np.float64)
        strategies.append({"alpha": alpha})
    return strategies

def predicted_control_other_full_horizon(param, agent_id, traj, xf_est, yf_est):
    """
    If agent_id=1 => param => x_f, else => param => y_f
    Then expansions => feedforward => alpha => pick alpha[1 or 0].
    """
    if agent_id==1:
        local_xf = param
        local_yf = yf_est
    else:
        local_xf = xf_est
        local_yf = param

    A,B = linearize_dynamics_full_horizon(traj)
    cst = cost_expansion_full_horizon(traj, A,B, local_xf, local_yf)
    strategies = solve_lq_game_fbne(A,B,cst)
    alpha0 = strategies[0]["alpha"]
    if agent_id==1:
        return alpha0[1]  # torque feedforward
    else:
        return alpha0[0]  # thrust feedforward
