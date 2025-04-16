# expansions.py
import numpy as np

nx = 6
nu = 2
ΔT = 0.1

def discrete_dynamics(x, u):
    """
    The discrete step for your LunarLander:
    x: shape (6,), u: shape (2,) => (thrust, torque)
    """
    x_next = np.zeros(6, dtype=np.float64)
    x_next[0] = x[0] + ΔT*x[3]
    x_next[1] = x[1] + ΔT*x[4]
    x_next[2] = x[2] + ΔT*x[5]
    x_next[3] = x[3] + ΔT*(u[0]*np.sin(x[2]))
    x_next[4] = x[4] + ΔT*(u[0]*np.cos(x[2]))
    x_next[5] = x[5] + ΔT*u[1]
    return x_next

def linearize_dynamics_full_horizon(traj):
    """
    We do a finite diff approach to get A[t], B[t] for each step t in 0..N-1.
    'traj' has 'x' and 'u' lists. 
    """
    N = len(traj["u"])
    A = []
    B = []
    eps = 1e-5
    for t in range(N):
        x_t = traj["x"][t]
        u_t = traj["u"][t]
        Ax = np.zeros((nx,nx), dtype=np.float64)
        Ab = np.zeros((nx,nu), dtype=np.float64)
        for i in range(nx):
            old = x_t[i]
            x_t[i] = old+eps
            fplus = discrete_dynamics(x_t,u_t)
            x_t[i] = old-eps
            fminus= discrete_dynamics(x_t,u_t)
            x_t[i] = old
            Ax[:,i] = (fplus-fminus)/(2*eps)
        for j in range(nu):
            oldu = u_t[j]
            u_t[j] = oldu+eps
            fplus = discrete_dynamics(x_t,u_t)
            u_t[j] = oldu-eps
            fminus= discrete_dynamics(x_t,u_t)
            u_t[j] = oldu
            Ab[:,j] = (fplus-fminus)/(2*eps)
        A.append(Ax)
        B.append(Ab)
    return A,B
