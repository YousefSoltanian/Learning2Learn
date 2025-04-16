# train_ilq.py
import numpy as np
import torch
import random

from ilq_core.solver import predicted_control_other_full_horizon
from model.transformer_belief import BeliefTransformer
from model.xf_module import XfModule

#########################
# 1) Single trajectory mismatch
#########################
def single_trajectory_loss(net, xfmod, all_sims, i):
    """
    For trajectory i, we fix net + x_f[i], run each step t:
      - partial history => net => y_f^t
      - expansions => feedforward => compare to actual
    We'll do agent_id=1 => param => x_f => feedforward => agent2
    We'll set u1=0
    """
    sim = all_sims[i]
    x_hist = sim["x_history"]
    u_hist = sim["u_history"]

    x_f_val = xfmod.xf[i].item()  # the per-trajectory param
    Tt = len(u_hist)
    local_loss = 0.0
    for t in range(Tt):
        partial_x = x_hist[:(t+1)]
        partial_u = u_hist[:t]
        # net => y_f^t
        y_f_t = net(partial_x, partial_u).item()

        # expansions => feedforward
        big_traj = {"x": x_hist, "u": u_hist}
        # agent_id=1 => param => x_f => alpha => agent2
        pred_u2 = predicted_control_other_full_horizon(x_f_val, 1, big_traj, x_f_val, y_f_t)
        # mismatch => u1=0
        uact = u_hist[t]
        local_loss += (0.0 - uact[0])**2 + (pred_u2 - uact[1])**2
    return local_loss

def batch_loss(net, xfmod, all_sims, idxs):
    s=0.0
    for i in idxs:
        s+= single_trajectory_loss(net, xfmod, all_sims, i)
    return s/len(idxs)

#########################
# 2) Updating x_f for each trajectory (inner loop)
#########################
def update_xf_for_trajectory(i, net, xfmod, all_sims, steps=1, eps=1e-3):
    """
    Fix net => do a small derivative approach on x_f[i].
    We do a 1D finite diff for x_f[i].
    'steps' means how many times we do it, 
    but typically 1 or 2 is enough.
    """
    for step in range(steps):
        old_val = xfmod.xf[i].item()
        # measure base
        base_loss = single_trajectory_loss(net, xfmod, all_sims, i)
        # +eps
        xfmod.xf[i] = torch.tensor(old_val+eps, dtype=torch.float32)
        plus_loss = single_trajectory_loss(net, xfmod, all_sims, i)
        # -eps
        xfmod.xf[i] = torch.tensor(old_val-eps, dtype=torch.float32)
        minus_loss = single_trajectory_loss(net, xfmod, all_sims, i)
        # restore
        xfmod.xf[i] = torch.tensor(old_val, dtype=torch.float32)

        grad_approx = (plus_loss - minus_loss)/(2*eps)
        # do a small update:
        lr_inner = 1e-1  # or some alpha
        new_val = old_val - lr_inner*grad_approx
        xfmod.xf[i] = torch.tensor(new_val, dtype=torch.float32)

#########################
# 3) Update the net param (outer step)
#    fix x_f => do FD in net
#########################
def gather_net_params(net):
    arrs = []
    for p in net.parameters():
        arrs.append(p.detach().cpu().numpy().ravel())
    if len(arrs)>0:
        return np.concatenate(arrs)
    else:
        return np.array([], dtype=np.float64)

def set_net_params(net, param_array):
    offset=0
    for p in net.parameters():
        numel = p.numel()
        chunk = param_array[offset:offset+numel]
        offset+=numel
        shaped = torch.from_numpy(chunk).view(p.shape)
        with torch.no_grad():
            p.copy_(shaped)

def fd_update_net(net, xfmod, all_sims, idxs, eps=1e-3, lr=1e-3):
    """
    A single FD step on net param, holding x_f fixed.
    We'll do param[i]+eps => measure => param[i]-eps => measure => difference => grad
    Then netParam = netParam - lr * grad
    """
    p0 = gather_net_params(net)
    baseL = batch_loss(net, xfmod, all_sims, idxs)
    g = np.zeros_like(p0, dtype=np.float64)

    for i in range(len(p0)):
        old = p0[i]
        p0[i] = old + eps
        set_net_params(net, p0)
        plusL = batch_loss(net, xfmod, all_sims, idxs)

        p0[i] = old - eps
        set_net_params(net, p0)
        minusL= batch_loss(net, xfmod, all_sims, idxs)

        # restore
        p0[i] = old
        set_net_params(net, p0)
        g[i] = (plusL - minusL)/(2*eps)

    # gradient step
    p0 -= lr*g
    set_net_params(net, p0)
    return baseL, g

#########################
# 4) The main 2-loop training
#########################
def main():
    data = np.load("data/LunarLander_Data.npz", allow_pickle=True)
    all_sims = data["all_sims"]
    N_data = len(all_sims)
    print("Loaded", N_data, "trajectories")

    from model.transformer_belief import BeliefTransformer
    from model.xf_module import XfModule

    embed_dim=16
    net = BeliefTransformer(embed_dim=embed_dim, nhead=2, num_encoder_layers=1)
    xfmod= XfModule(N_data)

    # block-coordinate approach
    n_outer_epochs = 3
    batchsize = 5
    import random

    for ep in range(1, n_outer_epochs+1):
        print(f"Outer Epoch {ep} / {n_outer_epochs}")

        # (A) fix net => update each x_f^(i) 
        #     we can do 1 pass over i in [0..N_data], or do a random subset
        #     here we do all for demonstration:
        for i in range(N_data):
            update_xf_for_trajectory(i, net, xfmod, all_sims, steps=1, eps=1e-3)

        # (B) fix x_f => update net param
        # pick a random batch of indices for HPC
        idxs = [random.randint(0, N_data-1) for _ in range(batchsize)]
        baseL, gradv = fd_update_net(net, xfmod, all_sims, idxs, eps=1e-3, lr=1e-3)
        print(f"  net update: batch_loss={baseL}")

    print("Done training. Example x_f:", xfmod.xf.data[:5])

if __name__=="__main__":
    main()
