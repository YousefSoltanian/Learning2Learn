import pickle
import random
import matplotlib.pyplot as plt
import numpy as np

# Path to the converted pickle file (adjust path if needed)
pkl_path = "data/LunarLander_Complete_Info_Peer_converted.pkl"

# Load the simulation data
with open(pkl_path, "rb") as f:
    all_simulations = pickle.load(f)

# Choose one random simulation from the list
sim = random.choice(all_simulations)

# Extract the trajectory stored under "x_history".
# Assumption: each state vector is a list (or array) where index 0 is x and index 1 is y.
x_history = sim["x_history"]

# Extract x and y coordinates
xs = [state[0] for state in x_history]
ys = [state[1] for state in x_history]

# Plot the trajectory: x vs. y
plt.figure()
plt.plot(xs, ys, marker="o")
plt.xlabel("x")
plt.ylabel("y")
plt.title(f"Trajectory for Simulation #{sim['simulation_index']}")
plt.grid(True)
plt.show()
