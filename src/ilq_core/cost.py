# cost.py
import numpy as np

def cost_function_agent1(x, u1, xf_est, yf_est):
    return (10*(x[0]-xf_est)**2 +
            10*(x[1]-yf_est)**2 +
            10*x[3]**2 + 10*x[4]**2 +
            10*(x[2]-np.pi/2)**2 + 10*x[5]**2 +
            1.0*u1**2)

def cost_function_agent2(x, u2, xf_est, yf_est):
    return (10*(x[0]-xf_est)**2 +
            10*(x[1]-yf_est)**2 +
            10*x[3]**2 + 10*x[4]**2 +
            10*(x[2]-np.pi/2)**2 + 10*x[5]**2 +
            1.0*u2**2)
