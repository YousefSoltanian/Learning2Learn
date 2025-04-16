##########################################################
# lunar_lander_belief_trainer.py
##########################################################

import jax
import jax.numpy as jnp
import optax
import numpy as np

# 1) Import your ILQSolver (no rewriting).
from iLQGame import *
#from iLQGame import cost, dynamics, constraint
#from iLQGame.ilq_solver import ILQSolver


# 2) Import your cost function(s) & dynamics for the Lander:
#    For example:
# from your_project.cost.lander_cost import LanderCost
# from your_project.dynamics.lander_dynamics import LunarLander2PlayerSystem
# from your_project.constraint.box_constraint import BoxConstraint

# 3) Import your BeliefUpdateTransformer (or any neural net you use for dynamic belief).
#    For example:
# from your_project.model.belief_transformer import BeliefUpdateTransformer

class LunarLanderBeliefTrainer:
    """
    A complete training class that:
      - Has 'static_params' (x_f) for each trajectory,
      - A shared BeliefUpdateTransformer (NN) for the dynamic part,
      - For each step in each trajectory, calls ILQSolver to get the final *nonlinear* solution
        => use that for mismatch,
      - Then calls the ILQSolver's internal linearization functions to get local LQ approximation
        => from that, extracts the feedforward control as a function of (x_f, NN params)
        => uses that for the delta term chain rule,
      - Does a nested approach:
         1) update x_f for each trajectory,
         2) update the NN with ADAM,
      - Incorporates all needed first/second derivatives w.r.t. x_f & NN for the delta term
        by using JAX's auto-diff on the local LQ feedforward expression.
    """

    def __init__(
        self,
        belief_model,        # e.g. a BeliefUpdateTransformer
        belief_optimizer,    # e.g. optax.adam(...) for the belief net
        static_lr,           # learning rate for x_f
        dataset,             # list of 10k trajectories: each has x_history, u_obs, etc.
        horizon=50,
        delta_weight=0.1,
        rng_seed=42
    ):
        self.belief_model = belief_model
        self.belief_optimizer = belief_optimizer
        self.static_lr = static_lr
        self.dataset = dataset
        self.num_trajectories = len(dataset)
        self.horizon = horizon
        self.delta_weight = delta_weight

        # --- Initialize the belief net parameters
        rng = jax.random.PRNGKey(rng_seed)
        dummy_input = jnp.ones((1, horizon, self.belief_model.state_dim))
        rng, init_rng, sample_rng, dropout_rng = jax.random.split(rng, 4)
        variables = self.belief_model.init(
            {'params': init_rng, 'sample': sample_rng, 'dropout': dropout_rng},
            dummy_input,
            train=True
        )
        self.belief_params = variables['params']
        self.opt_state = self.belief_optimizer.init(self.belief_params)

        # --- For each trajectory, store x_f in an array (num_trajs, 1) or (num_trajs, something)
        self.static_params = jnp.zeros((self.num_trajectories, 1))

        # We'll keep an ILQ policy guess (Ps, alphas) if you want a warmstart
        # For simplicity, we do None. We'll create them on the fly.
        self.rng = rng

    # ------------------------------------------------
    #  HELPER: run ILQ to get final control for mismatch
    # ------------------------------------------------
    def _run_ilq_at_step(
        self,
        x_state,
        xf_value,
        belief_params,
        step_idx,
        horizon_ilq=50
    ):
        """
        For state x_state (shape ~ (6,)), we form the ILQ game at this step:
          1) define cost that references xf_value & the belief's y_f (if relevant)
          2) define constraints, initial Ps, alphas
          3) build ILQSolver => solver.run(x_state)
          4) extract the final best controls => mismatch control
        Return (u_pred, solver, xs_nonlinear, us_nonlinear)
        so we can also do linearization afterward
        """
        # e.g. build cost:
        # cost_p1 = LanderCost(xf_value, some_yf, ...)
        # cost_p2 = ...
        # or if you have a single-player setting for the human, do that.

        # create ILQSolver
        # For demonstration, let's do a minimal example:
        # Ps_init, alphas_init => shape ...
        # constraints => ...
        # solver = ILQSolver(dynamics=..., player_costs=[cost_p1, cost_p2],
        #                    Ps=[Ps_init1, Ps_init2], alphas=[alpha_init1, alpha_init2],
        #                    alpha_scaling=[0.5,1.0,1.5],
        #                    max_iter=50,
        #                    u_constraints=[box1, box2],
        #                    verbose=False)
        # solver.run(x_state)
        # xs_final, us_final = solver._best_operating_point[0], solver._best_operating_point[1]
        # immediate_control = ...
        # return immediate_control, solver, xs_final, us_final

        # We'll do placeholders:
        solver = None  # you'd actually build it
        xs_final = jnp.zeros((6, horizon_ilq))
        us_final = [jnp.zeros((1, horizon_ilq)), jnp.zeros((1, horizon_ilq))]
        # immediate control:
        u_pred = us_final[0][:,0]  # if we consider player0 is the "human"
        return u_pred, solver, xs_final, us_final

    # ------------------------------------------------
    # HELPER: linearize around final solution => get feedforward
    # ------------------------------------------------
    def _get_feedforward_lq(
        self,
        solver,
        xs_nonlinear,
        us_nonlinear
    ):
        """
        1) call solver._linearize_dynamics(xs_nonlinear, us_nonlinear)
        2) call solver._quadraticize_costs(xs_nonlinear, us_nonlinear)
        3) call solver._solve_lq_game(...)
        4) extract the immediate feedforward alpha from the first time step => alpha_approx
        Return alpha_approx => shape (control_dim,).
        That alpha_approx is a function of x_f & belief net if the cost depends on them,
        so we use it for autodiff.
        """
        As, Bs_list = solver._linearize_dynamics(xs_nonlinear, us_nonlinear)
        costs, lxs, Hxxs, Huus = solver._quadraticize_costs(xs_nonlinear, us_nonlinear)

        As_np = np.asarray(As)
        Bs_np_list = [np.asarray(Bs) for Bs in Bs_list]
        lxs_np = [np.asarray(l) for l in lxs]
        Hxxs_np = [np.asarray(Hxx) for Hxx in Hxxs]
        Huus_np = [np.asarray(Huu) for Huu in Huus]

        Ps, alphas, Zs, zetas = solver._solve_lq_game(As_np, Bs_np_list, Hxxs_np, lxs_np, Huus_np)
        # alpha_approx for the first time step => alphas[0][:,0] if the "human" is player 0
        alpha_approx = alphas[0][:,0]  # shape (1,) if single-dim control
        return alpha_approx

    # ------------------------------------------------
    # compute the entire trajectory's loss
    # mismatch + delta
    # ------------------------------------------------
    def compute_trajectory_loss(self, traj_idx, xf_value, belief_params):
        """
        For each time step in [0..T-1]:
          1) get current state x_t, observed control u_obs[t]
          2) run ILQ => final control => mismatch
          3) linearize => local feedforward => delta
        sum them up => return scalar
        We'll do manual partial derivatives if needed, or rely on jax to do so if we define
        the feedforward as a function. 
        """
        data_i = self.dataset[traj_idx]
        x_history = data_i['x_history']  # shape (T, x_dim)
        u_obs = data_i['u_obs']         # shape (T,) or (T,1)
        mismatch_loss = 0.0
        delta_loss = 0.0

        prev_control_pred = None
        prev_control_obs = None
        for t in range(self.horizon):
            # 1) run ILQ
            x_t = x_history[t]
            # we might call the belief net with the entire history up to t to get yf
            # e.g. => (sample_y, mean_y, var_y) = self.belief_model(...)
            # for demonstration, skip or do a placeholder
            y_f_est = 0.0

            # run ILQ to get final control
            u_pred_t, solver_t, xs_final, us_final = self._run_ilq_at_step(
                x_t, xf_value, belief_params, t, horizon_ilq=self.horizon
            )

            # mismatch
            mismatch_loss += jnp.sum((u_obs[t] - u_pred_t)**2)

            # 2) delta => if t>0
            if t > 0:
                delta_obs = (u_obs[t] - prev_control_obs)
                # approximate local feedforward from the linear game
                alpha_approx_t = self._get_feedforward_lq(solver_t, xs_final, us_final)
                # predicted delta = alpha_approx_t - prev_control_pred
                # This is a simplified discrete difference approach
                delta_pred = alpha_approx_t - prev_control_pred
                delta_loss += jnp.sum((delta_obs - delta_pred)**2)

            prev_control_pred = u_pred_t
            prev_control_obs = u_obs[t]

        total_loss = mismatch_loss + self.delta_weight * delta_loss
        return total_loss

    # ------------------------------------------------
    # partial derivative wrt x_f
    # we do a standard jax.grad
    # ------------------------------------------------
    def grad_static_param(self, traj_idx, xf_value, belief_params):
        """
        We'll define local_loss(sp) => compute_trajectory_loss(...),
        then do g = jax.grad(local_loss)(sp).
        That is enough for a first-order approach, but we have the feedforward
        as a function => jax can do second derivatives if we do jax.hessian if you want.

        But let's keep it first-order:
        """
        def local_loss(sp):
            return self.compute_trajectory_loss(traj_idx, sp, belief_params)
        g = jax.grad(local_loss)(xf_value)
        return g

    # ------------------------------------------------
    # summation over a batch for the belief net
    # ------------------------------------------------
    def loss_batch_belief(self, belief_params, batch_indices, static_batch):
        total = 0.0
        for i, idx in enumerate(batch_indices):
            sp_i = static_batch[i]
            total += self.compute_trajectory_loss(idx, sp_i, belief_params)
        return total / len(batch_indices)

    # ------------------------------------------------
    # 1) inner update for x_f
    # ------------------------------------------------
    def update_static_params_inner(self, batch_indices, static_batch, belief_params):
        updated = []
        for i, idx in enumerate(batch_indices):
            g = self.grad_static_param(idx, static_batch[i], belief_params)
            new_sp = static_batch[i] - self.static_lr * g
            updated.append(new_sp)
        return jnp.stack(updated, axis=0)

    # ------------------------------------------------
    # 2) outer update for belief net
    # ------------------------------------------------
    @jax.jit
    def update_belief_params_outer(self, belief_params, opt_state, batch_indices, static_batch):
        def loss_fn(p):
            return self.loss_batch_belief(p, batch_indices, static_batch)
        loss_val, grads = jax.value_and_grad(loss_fn)(belief_params)
        updates, new_opt_state = self.belief_optimizer.update(grads, opt_state)
        new_belief_params = optax.apply_updates(belief_params, updates)
        return new_belief_params, new_opt_state, loss_val

    # ------------------------------------------------
    # combine them => nested step
    # ------------------------------------------------
    def train_step(self, batch_indices):
        static_batch = self.static_params[batch_indices]
        # 1) update x_f
        updated_sp = self.update_static_params_inner(batch_indices, static_batch, self.belief_params)
        # 2) update belief
        new_belief, new_opt_state, loss_val = self.update_belief_params_outer(
            self.belief_params, self.opt_state, batch_indices, updated_sp
        )
        self.belief_params = new_belief
        self.opt_state = new_opt_state
        self.static_params = self.static_params.at[batch_indices].set(updated_sp)
        return loss_val

    # ------------------------------------------------
    # full training
    # ------------------------------------------------
    def train(self, num_epochs=10, batch_size=32):
        num_trajs = self.num_trajectories
        for epoch in range(num_epochs):
            perm = np.random.permutation(num_trajs)
            for start in range(0, num_trajs, batch_size):
                batch_idx = perm[start:start+batch_size]
                loss_val = self.train_step(batch_idx)
            print(f"Epoch {epoch+1}: last batch loss = {loss_val}")
        print("Training complete.")
