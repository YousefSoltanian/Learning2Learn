import os
import numpy as np
import jax.numpy as jnp
from copy import deepcopy
import matplotlib.pyplot as plt

# Import your modules (adjust as needed)
from cost import LunarLanderCost
from constraint import BoxConstraint
from multiplayer_dynamical_system import LunarLander2PlayerSystem
from ilq_solver import ILQSolver
from player_cost import PlayerCost

def main():
    # --- 1) Create the two-player Lunar Lander system ---
    # State: [px, py, angle, vx, vy, omega]
    # Controls: [thrust] for player1, [torque] for player2
    lander = LunarLander2PlayerSystem(T=0.1)  # discrete step size

    # --- 2) Set simulation parameters ---
    sim_steps = 50                  # Total simulation steps
    receding_horizon = 50           # ILQ solver receding-horizon length

    # --- 3) Initial state ---
    # For example, start at x = [-5, 8, 0, 0, 0, 0]
    x_current = jnp.array([-5.0, 8.0, 0.0, 0.0, 0.0, 0.0])

    # Containers to store simulation data:
    state_traj = []     # will store state vector at each simulation step
    control_traj_p1 = []  # player 1 control (thrust)
    control_traj_p2 = []  # player 2 control (torque)

    # --- 4) Define the cost function for each player (for receding horizon) ---
    cost_player1 = LunarLanderCost(
        u_index=0,
        xf_est=5.0,
        yf_est=-5.0,
        name="P1_lander_cost",
        horizon=receding_horizon,
        x_dim=6,
        ui_dim=1
    )
    cost_player2 = LunarLanderCost(
        u_index=0,
        xf_est=5.0,
        yf_est=-5.0,
        name="P2_lander_cost",
        horizon=receding_horizon,
        x_dim=6,
        ui_dim=1
    )

    # Wrap each cost in a PlayerCost container (if required by your solver)
    p1_cost_container = PlayerCost()
    p1_cost_container.add_cost(cost_player1, arg="x", weight=1.0)
    p2_cost_container = PlayerCost()
    p2_cost_container.add_cost(cost_player2, arg="x", weight=1.0)

    # --- 5) Define initial ILQ policies (feedback gains and feedforward terms) ---
    # For each player, shapes are:
    #   P: (control_dim, state_dim, receding_horizon)
    #   alpha: (control_dim, receding_horizon)
    P1_init = jnp.zeros((1, 6, receding_horizon))
    P2_init = jnp.zeros((1, 6, receding_horizon))
    alpha1_init = jnp.zeros((1, receding_horizon))
    alpha2_init = jnp.zeros((1, receding_horizon))
    # Example alpha scaling array for line search:
    alpha_scaling = jnp.linspace(0.1, 2.0, 4)

    # Define control input constraints per player.
    constraints_thrust = BoxConstraint(lower=-100, upper=100)
    constraints_torque = BoxConstraint(lower=-100, upper=100)
# Create an ILQSolver for the current receding horizon.
    solver = ILQSolver(
        dynamics=lander,
        player_costs=[p1_cost_container, p2_cost_container],
        Ps=[P1_init, P2_init],
        alphas=[alpha1_init, alpha2_init],
        alpha_scaling=alpha_scaling,
        max_iter=100,
        u_constraints=[constraints_thrust, constraints_torque],
        verbose=False
    )
    # --- 6) Simulation Loop: receding-horizon ILQ control ---
    for step in range(sim_steps):
        print(step)
        # Run the solver with the current state as initial state.
        #solver._horizon = receding_horizon - step
        solver.run(x_current)
        # Extract the receding-horizon trajectory and controls
        # xs_rec: shape (6, receding_horizon)
        # us_rec: list [u_player1, u_player2], each of shape (1, receding_horizon)
        xs_rec = solver._best_operating_point[0]
        us_rec = solver._best_operating_point[1]
        # Use only the first control from the receding horizon.
        u1_immediate = us_rec[0][:, 0]  # shape (1,)
        u2_immediate = us_rec[1][:, 0]  # shape (1,)

        # Save current state and controls for later plotting.
        state_traj.append(x_current)
        control_traj_p1.append(u1_immediate)
        control_traj_p2.append(u2_immediate)

        # Apply the immediate controls to the dynamics to obtain next state.
        # Each control is an array of shape (1,); form u_list accordingly.
        u_list_step = [u1_immediate, u2_immediate]
        x_next = lander.disc_time_dyn(x_current, u_list_step)
        # Update current state for next simulation step.
        x_current = x_next

    # Convert stored data to arrays for plotting.
    # state_traj: shape (6, sim_steps)
    state_traj = jnp.stack(state_traj, axis=1)
    # control trajectories: shape (1, sim_steps)
    control_traj_p1 = jnp.stack(control_traj_p1, axis=1)
    control_traj_p2 = jnp.stack(control_traj_p2, axis=1)

    # --- 7) Plot the results ---
    # Plot trajectory (x vs y).
    plt.figure()
    plt.plot(state_traj[0, :], state_traj[1, :], 'b-o')
    plt.xlabel("x position")
    plt.ylabel("y position")
    plt.title("Lunar Lander Trajectory")
    plt.grid(True)

    # Plot control signals over simulation steps.
    plt.figure()
    plt.plot(control_traj_p1[0, :], label="Thrust (Player 1)")
    plt.plot(control_traj_p2[0, :], label="Torque (Player 2)")
    plt.xlabel("Simulation Step")
    plt.ylabel("Control Signal")
    plt.title("Control Signals")
    plt.legend()
    plt.grid(True)

    plt.show()


if __name__ == "__main__":
    main()
