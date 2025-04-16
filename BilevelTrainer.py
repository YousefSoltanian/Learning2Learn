import jax
import jax.numpy as jnp
import optax
import numpy as np
import sys
import os
sys.path.append("src")

# --- ILQ solver + cost classes (not purely jax) ---
from functools import partial
from iLQGame.ilq_solver import ILQSolver
from iLQGame.cost import LunarLanderCost
from iLQGame.constraint import BoxConstraint
from iLQGame.multiplayer_dynamical_system import LunarLander2PlayerSystem
from iLQGame.player_cost import PlayerCost

# --- Belief & static modules (Flax-based) ---
from model.transformer_belief import BeliefUpdateTransformer
from model.intent_model import StaticIntentModule


###############################################################################
# 1) A custom mismatch: numeric => (u_obs - u_ilq)**2, but partial derivative
#    uses alpha_0 => see your earlier code
###############################################################################
@jax.custom_vjp
def mismatch_custom(u_obs, u_ilq, alpha_0_val):
    return (u_obs - u_ilq)**2

def _mismatch_fwd(u_obs, u_ilq, alpha_0_val):
    y = mismatch_custom(u_obs, u_ilq, alpha_0_val)
    return y, (u_obs, u_ilq, alpha_0_val)

def _mismatch_bwd(res, g):
    (u_obs, u_ilq, alpha_0_val) = res
    grad_uobs = 0.0
    grad_uilq = 0.0
    # partial wrt alpha_0 = -2*(u_obs - u_ilq)*g
    grad_alpha = -2.0*(u_obs - u_ilq)*g
    return (grad_uobs, grad_uilq, grad_alpha)

mismatch_custom.defvjp(_mismatch_fwd, _mismatch_bwd)


###############################################################################
# 2) The main nested trainer, but adds a pure-JAX LQ solver to get alpha0
###############################################################################
class NestedADAMLanderTrainer:
    def __init__(
        self,
        dataset: list,
        horizon: int,
        belief_model: BeliefUpdateTransformer,
        delta_weight=0.1,
        xf_lr=1e-3,
        net_lr=1e-3,
        rng_seed=42
    ):
        """
        :param dataset: e.g. 1000 or 10000 items, each: 
                        {'x_history':(horizon,6), 'u_obs':(horizon,)}
        :param horizon: time steps
        :param belief_model: your BeliefUpdateTransformer
        :param delta_weight: weighting for the delta term
        :param xf_lr: ADAM LR for each x_f
        :param net_lr: ADAM LR for net
        :param rng_seed: random seed
        """
        self.intent_model = StaticIntentModule(output_dim=1)
        self.dataset = dataset
        self.num_trajs = len(dataset)
        self.horizon = horizon
        self.belief_model = belief_model
        self.delta_weight = delta_weight

        rng = jax.random.PRNGKey(rng_seed)

        # 1) Initialize the net
        dummy_input = jnp.ones((1, horizon, belief_model.state_dim))
        rng, init_rng, sample_rng, dropout_rng = jax.random.split(rng, 4)
        net_vars = belief_model.init(
            {"params": init_rng, "sample": sample_rng, "dropout": dropout_rng},
            dummy_input,
            train=True
        )
        self.net_params = net_vars["params"]
        self.net_opt = optax.adam(net_lr)
        self.net_opt_state = self.net_opt.init(self.net_params)

        # 2) For each trajectory, create a StaticIntentModule => store param dict
        #    plus an ADAM state
        self.xf_modules = []
        self.xf_optimizers = []
        self.xf_opt_states = []

        for _ in range(self.num_trajs):
            mod = StaticIntentModule(output_dim=1)
            rng, rng_init = jax.random.split(rng)
            mod_vars = mod.init(
                {'params': rng_init},  # params
                rng_init, sample=True
            )
            self.xf_modules.append(mod_vars)
            xf_opt = optax.adam(xf_lr)
            xf_opt_state = xf_opt.init(mod_vars["params"])
            self.xf_optimizers.append(xf_opt)
            self.xf_opt_states.append(xf_opt_state)


    ################################################################
    # A pure-JAX LQ solver => alpha0 is fully differentiable
    # We'll call this after we do a numeric ILQ solve 
    # to build A, B, Q, l, R as jax arrays
    ################################################################
    def _solve_lq_game_jax(self, A_jax, B_jax_list, Q_jax_list, l_jax_list, R_jax_list):
        """
        Minimal JAX-based time-varying LQ game solver => returns (Ps, alphas)
        plus Zs, zetas if you want. 
        shape details:
         - A_jax: (x_dim, x_dim, horizon)
         - B_jax_list[i]: (x_dim, ui_dim, horizon)
         - Q_jax_list[i]: (x_dim, x_dim, horizon)
         - l_jax_list[i]: (x_dim, horizon)
         - R_jax_list[i]: (ui_dim, ui_dim, horizon)
        We'll do 2 players for example. Adjust for your case.
        """
        horizon = A_jax.shape[2]
        num_players = len(B_jax_list)
        x_dim = A_jax.shape[0]

        # For demonstration, assume each player's control dim is known
        u_dims = [B_jax_list[i].shape[1] for i in range(num_players)]

        # allocate
        Zs = []
        zetas = []
        for i in range(num_players):
            Zs.append(jnp.zeros((x_dim, x_dim, horizon+1)))
            zetas.append(jnp.zeros((x_dim, horizon+1)))

        # final boundary
        for i in range(num_players):
            Zs[i] = Zs[i].at[:, :, -1].set(Q_jax_list[i][:, :, -1])
            zetas[i] = zetas[i].at[:, -1].set(l_jax_list[i][:, -1])

        Ps_list = []
        alphas_list = []
        for i in range(num_players):
            Ps_list.append(jnp.zeros((u_dims[i], x_dim, horizon)))
            alphas_list.append(jnp.zeros((u_dims[i], horizon)))

        # backward pass
        for k in reversed(range(horizon)):
            A_k = A_jax[:, :, k]
            B_k = [B_jax_list[i][:, :, k] for i in range(num_players)]
            Q_k = [Q_jax_list[i][:, :, k] for i in range(num_players)]
            l_k = [l_jax_list[i][:, k]    for i in range(num_players)]
            R_k = [R_jax_list[i][:, :, k] for i in range(num_players)]
            Z_next = [Zs[i][:, :, k+1]  for i in range(num_players)]
            zeta_next = [zetas[i][:, k+1] for i in range(num_players)]

            # build S
            row_blocks = []
            for i2 in range(num_players):
                row_i = []
                for j2 in range(num_players):
                    block_ij = B_k[i2].T @ Z_next[i2] @ B_k[j2]
                    if i2 == j2:
                        block_ij = block_ij + R_k[i2]
                    row_i.append(block_ij)
                row_blocks.append(jnp.concatenate(row_i, axis=1))
            S = jnp.concatenate(row_blocks, axis=0)  # shape(sum_u, sum_u)

            # build Y => shape (sum_u, x_dim)
            Y_list = []
            for i2 in range(num_players):
                tmp = B_k[i2].T @ Z_next[i2] @ A_k
                Y_list.append(tmp)
            Y = jnp.concatenate(Y_list, axis=0)

            # solve P_big = pinv(S) * Y
            S_inv = jnp.linalg.pinv(S)
            P_big = S_inv @ Y

            # split P_big
            offset = 0
            P_split = []
            for i2 in range(num_players):
                du = u_dims[i2]
                sub = P_big[offset:offset+du, :]
                offset += du
                Ps_list[i2] = Ps_list[i2].at[:, :, k].set(sub)
                P_split.append(sub)

            # F = A_k - sum B_k[i2] @ P_split[i2]
            sumBP = jnp.zeros((x_dim, x_dim))
            for i2 in range(num_players):
                sumBP = sumBP + B_k[i2] @ P_split[i2]
            F_k = A_k - sumBP

            # update Z
            for i2 in range(num_players):
                newZ = F_k.T @ Z_next[i2] @ F_k + Q_k[i2] + P_split[i2].T @ R_k[i2] @ P_split[i2]
                Zs[i2] = Zs[i2].at[:, :, k].set(newZ)

            # build Y2 => shape (sum_u,)
            Y2_list = []
            for i2 in range(num_players):
                tmp2 = B_k[i2].T @ zeta_next[i2]
                Y2_list.append(tmp2)
            Y2 = jnp.concatenate(Y2_list, axis=0)

            alpha_big = S_inv @ Y2

            # split alpha_big
            offset = 0
            alpha_subs = []
            for i2 in range(num_players):
                du = u_dims[i2]
                a_sub = alpha_big[offset:offset+du]
                offset += du
                alphas_list[i2] = alphas_list[i2].at[:, k].set(a_sub)
                alpha_subs.append(a_sub)

            # beta
            beta = jnp.zeros((x_dim,))
            for i2 in range(num_players):
                beta = beta - B_k[i2] @ alpha_subs[i2]

            # update zeta
            for i2 in range(num_players):
                zeta_new = (F_k.T @ (zeta_next[i2] + Z_next[i2] @ beta)
                            + l_k[i2]
                            + P_split[i2].T @ R_k[i2] @ alpha_subs[i2])
                zetas[i2] = zetas[i2].at[:, k].set(zeta_new)

        return Ps_list, alphas_list, Zs, zetas


    ################################################################
    # 3) ILQ => numeric => u_ilq, but then we do a JAX-based LQ solve
    #    to get alpha_0 for chain rule
    ################################################################
    def _ilq_and_alpha0(self, x_state, x_f_val, b_val):
        """
        1) numeric ILQ => u_ilq
        2) build A_jax, B_jax, Q_jax, l_jax, R_jax => call _solve_lq_game_jax => alpha0
        """
        # A) numeric ILQ part
        #x_f_val_ilq=np.array(jax.lax.stop_gradient(x_f_val))
        #b_val_ilq=np.array(jax.lax.stop_gradient(b_val))
        x_f_val_ilq = x_f_val
        b_val_ilq = b_val
        cost1 = LunarLanderCost(
            u_index=0, xf_est=x_f_val_ilq, yf_est=b_val_ilq,
            name="Cost1", horizon=self.horizon, x_dim=6, ui_dim=1
        )
        cost2 = LunarLanderCost(
            u_index=0, xf_est=x_f_val_ilq, yf_est=b_val_ilq,
            name="Cost2", horizon=self.horizon, x_dim=6, ui_dim=1
        )
        pc1 = PlayerCost()
        pc1.add_cost(cost1, arg="x", weight=1.0)
        pc2 = PlayerCost()
        pc2.add_cost(cost2, arg="x", weight=1.0)

        lander = LunarLander2PlayerSystem(T=0.1)
        box_thrust = BoxConstraint(-100, 100)
        box_torque = BoxConstraint(-100, 100)

        import jax.numpy as jnp
        P1_init = jnp.zeros((1, 6, self.horizon))
        P2_init = jnp.zeros((1, 6, self.horizon))
        alpha1_init = jnp.zeros((1, self.horizon))
        alpha2_init = jnp.zeros((1, self.horizon))
        alpha_scaling = jnp.linspace(0.1, 2.0, 4)

        solver = ILQSolver(
            dynamics=lander,
            player_costs=[pc1, pc2],
            Ps=[P1_init, P2_init],
            alphas=[alpha1_init, alpha2_init],
            alpha_scaling=alpha_scaling,
            max_iter=50,
            u_constraints=[box_thrust, box_torque],
            verbose=False
        )
        solver._horizon = self.horizon
        solver.run(x_state)

        # numeric => from solver best
        xs_fin = solver._best_operating_point[0]  # shape(6,horizon)
        us_fin = solver._best_operating_point[1]  # list(2), each shape(1,horizon)
        # immediate
        u_ilq_val = us_fin[0][0, 0]  # numeric

        # B) get alpha0 from a pure JAX LQ solve
        #    gather the A, B, Q, l, R from solver => then convert to jax arrays
        As, Bs_list = solver._linearize_dynamics(xs_fin, us_fin)
        cost_list, lxs_list, Hxxs_list, Huus_list = solver._quadraticize_costs(xs_fin, us_fin)

        # They might be partial lists => cost_list is a python list of shape(2) for 2 players
        # We'll build jax arrays for each
        # cost_list[i] => shape(horizon,)?
        # lxs_list[i] => shape(6,horizon)
        # Hxxs_list[i] => shape(6,6,horizon)
        # Huus_list[i] => shape(1,1,horizon)

        # We'll define Q_jax_list = Hxxs_list, l_jax_list = lxs_list, R_jax_list = Huus_list
        # Then call _solve_lq_game_jax
        # We'll rely on jnp.asarray(...) for each

        A_jax = jnp.asarray(As)  # shape(6,6,horizon)
        B_jax_list = []
        Q_jax_list = []
        l_jax_list = []
        R_jax_list = []
        for i in range(2):
            B_jax_list.append(jnp.asarray(Bs_list[i]))
            Q_jax_list.append(jnp.asarray(Hxxs_list[i]))
            l_jax_list.append(jnp.asarray(lxs_list[i]))
            R_jax_list.append(jnp.asarray(Huus_list[i]))

        # Solve
        Ps_list, alphas_list, _, _ = self._solve_lq_game_jax(
            A_jax, B_jax_list, Q_jax_list, l_jax_list, R_jax_list
        )
        # alpha0 is alphas_list[0], shape(1,horizon)
        alpha0_val = alphas_list[0][0, 0]  # immediate

        return u_ilq_val, alpha0_val

    ################################################################
    # compute belief => just call net
    ################################################################
    def _compute_belief(self, net_params, x_history_up_to_t):
        s, m, v = self.belief_model.apply(
            {"params": net_params},
            x_history_up_to_t,
            train=True,
            rngs={"sample": jax.random.PRNGKey(999), "dropout": jax.random.PRNGKey(1000)}
        )
        # for demonstration => pick final mean
        return s[0,0]

    ################################################################
    # single trajectory => mismatch + delta
    ################################################################
    def single_traj_loss(self, xf_mod_vars, net_params, i):
        # parse x_f
        rng = jax.random.PRNGKey(42)

        # For a forward pass (e.g., computing the static intent sample)
        rng, subkey = jax.random.split(rng)
        StaticIntentModule(output_dim=1)
        sample_val, mean, var = self.intent_model.apply(xf_mod_vars, subkey, sample=True)

        data = self.dataset[i]
        x_hist = data["x_history"]  # shape (horizon,6)
        u_obs = data["u_obs"]       # shape (horizon,)

        mismatch_sum = 0.0
        delta_sum = 0.0
        prev_x = None
        prev_b = None
        prev_u_obs = None

        for t in range(self.horizon):
            x_t = x_hist[t]
            x_up = jnp.expand_dims(x_hist[: t + 1], axis=0)  # shape (1, t+1, 6)
            b_t = self._compute_belief(net_params, x_up)

            # numeric ILQ => u_ilq, + jax-based alpha => alpha_0
            u_ilq_val, alpha_0_val = self._ilq_and_alpha0(x_t, sample_val, b_t)

            # mismatch => custom
            mm = mismatch_custom(u_obs[t], u_ilq_val, alpha_0_val)
            mismatch_sum += mm

            # delta => if t>0
            if t > 0:
                delta_obs = (u_obs[t] - prev_u_obs)
                dx = (x_t - prev_x)
                db = (b_t - prev_b)

                # define alpha_sub
                def alpha_sub(xt_in, bt_in):
                    _, a_val = self._ilq_and_alpha0(xt_in, sample_val, bt_in)
                    return a_val

                jacfun = jax.jacobian(alpha_sub, argnums=(0, 1))
                grad_x_alpha, grad_b_alpha = jacfun(x_t, b_t)

                # shape => grad_x_alpha is scalar(6?), grad_b_alpha is scalar => assume
                # for short => do dot => ignoring dimension checks
                delta_model = jnp.dot(grad_x_alpha, dx) + grad_b_alpha * db
                delta_expr = (delta_obs - delta_model)
                delta_sum += delta_expr ** 2

            prev_x = x_t
            prev_b = b_t
            prev_u_obs = u_obs[t]

        return mismatch_sum + self.delta_weight * delta_sum

    ################################################################
    # partial derivative wrt xf => single trajectory
    ################################################################
    def grad_xf_traj(self, xf_mod_vars, net_params, i):
        def local_loss(mv):
            return self.single_traj_loss(mv, net_params, i)
        return jax.grad(local_loss)(xf_mod_vars)

    ################################################################
    # single ADAM step => x_f
    ################################################################
    #@partial(jax.jit, static_argnums=(2,))
    def update_xf_single(self, xf_mod_vars, xf_opt_state, i, net_params):
        def loss_fn(modvars):
            return self.single_traj_loss(modvars, net_params, i)
        lv, grads = jax.value_and_grad(loss_fn)(xf_mod_vars)
        updates, new_opt_state = self.xf_optimizers[i].update(grads["params"], xf_opt_state)
        new_vars = {"params": optax.apply_updates(xf_mod_vars["params"], updates)}  
        return new_vars, new_opt_state, lv

    ################################################################
    # partial derivative wrt net => sum across batch
    ################################################################
    def loss_batch_net(self, net_params, batch_indices):
        total = 0.0
        for i in batch_indices:
            total += self.single_traj_loss(self.xf_modules[i], net_params, i)
        return total / len(batch_indices)

    #@jax.jit
    #@partial(jax.jit, static_argnums=0)
    def update_net_once(self, net_params, net_opt_state, batch_indices):
        def loss_fn(pp):
            return self.loss_batch_net(pp, batch_indices)
        lv, grads = jax.value_and_grad(loss_fn)(net_params)
        updates, new_opt_state = self.net_opt.update(grads, net_opt_state)
        new_pp = optax.apply_updates(net_params, updates)
        return new_pp, new_opt_state, lv

    ################################################################
    # the nested approach
    ################################################################
    def train(self, epochs=10, xf_steps=1, net_steps=1, batch_size=32):
        n = self.num_trajs
        for ep in range(epochs):
            # 1) fix net => update each x_f
            for i in range(n):
                for _ in range(xf_steps):
                    old_vars = self.xf_modules[i]
                    old_opt_st = self.xf_opt_states[i]
                    new_vars, new_opt_st, lv_xf = self.update_xf_single(
                        old_vars, old_opt_st, i, self.net_params
                    )
                    self.xf_modules[i] = new_vars
                    self.xf_opt_states[i] = new_opt_st

            # 2) fix x_f => update net
            perm = np.random.permutation(n)
            lv_net = 0.0
            for _ in range(net_steps):
                start = 0
                while start < n:
                    b_idx = perm[start : start + batch_size]
                    new_net, new_net_st, lv_net = self.update_net_once(
                        self.net_params, self.net_opt_state, b_idx
                    )
                    self.net_params = new_net
                    self.net_opt_state = new_net_st
                    start += batch_size

            print(f"Epoch {ep+1}: last xf update = {lv_xf}, last net update = {lv_net}")

        print("Nested ADAM training complete.")




def main():
    import jax
    import jax.numpy as jnp
    import numpy as np

    # --------------------------------------------------
    # 1) Create a trivial dataset
    #    Suppose we have 2 trajectories, each with horizon=5
    #    We'll fill them with random or simple states & controls.
    # --------------------------------------------------
    horizon = 5
    num_trajs = 2

    # Each data item: {'x_history': shape (horizon, 6),
    #                  'u_obs': shape (horizon,) }
    # For demonstration, we just do random data (in practice you have real data).
    dataset = []
    for _ in range(num_trajs):
        x_history = np.random.randn(horizon, 6).astype(np.float32)
        u_obs = np.random.randn(horizon).astype(np.float32)
        dataset.append({
            'x_history': x_history,
            'u_obs': u_obs
        })

    # --------------------------------------------------
    # 2) Instantiate a BeliefUpdateTransformer
    # --------------------------------------------------
    from model.transformer_belief import BeliefUpdateTransformer
    from model.intent_model import StaticIntentModule

    # Dummy config for your network
    belief_net = BeliefUpdateTransformer(
        max_seq_len=horizon,
        state_dim=6,
        embed_dim=32,
        transformer_layers=1,
        num_heads=2,
        mlp_hidden=64,
        output_dim=1,
        dropout_rate=0.1
    )

    # --------------------------------------------------
    # 3) Instantiate the trainer
    # --------------------------------------------------
    #from nested_adam_lander import NestedADAMLanderTrainer  # or wherever your class is

    trainer = NestedADAMLanderTrainer(
        dataset=dataset,
        horizon=horizon,
        belief_model=belief_net,
        delta_weight=0.1,
        xf_lr=1e-3,
        net_lr=1e-3,
        rng_seed=42
    )

    # --------------------------------------------------
    # 4) Run a single training step (or short training)
    #    We'll do 1 epoch, 1 inner step, 1 outer step
    # --------------------------------------------------
    trainer.train(epochs=1, xf_steps=1, net_steps=1, batch_size=1)

    # --------------------------------------------------
    # 5) Optionally, inspect gradient norms for a quick check
    #    E.g. pick one trajectory, compute grad wrt x_f
    # --------------------------------------------------
    i = 0  # first trajectory
    xf_vars = trainer.xf_modules[i]
    net_p = trainer.net_params
    g_xf = trainer.grad_xf_traj(xf_vars, net_p, i)
    # Just print the norm of x_f's 'mean' gradient
    grad_mean = g_xf['params']['mean']
    print("Grad norm (x_f.mean):", float(jnp.linalg.norm(grad_mean)))

    # For net, we can do a small batch index to see if net grads are non-zero
    grads_net = jax.grad(trainer.loss_batch_net)(trainer.net_params, [0])  # just 1 traj
    # e.g. drill down to one parameter name
    net_some_weight = jax.tree_util.tree_leaves(grads_net)[0]
    print("Grad norm (some net param):", float(jnp.linalg.norm(net_some_weight)))

    print("Done with quick test. Check above prints for non-zero gradients.")


# If you want to run from command line, ensure this is guarded:
if __name__ == "__main__":
    main()
