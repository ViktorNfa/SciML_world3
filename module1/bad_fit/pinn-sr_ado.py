import numpy as np, pandas as pd, torch, torch.nn as nn, torch.optim as optim
from pyworld3 import World3
import matplotlib.pyplot as plt

torch.manual_seed(0)


# -- 0. Get simulation data ---------------------------------------------------
y_min, y_max, dt = 1900, 2200, 0.1
w = World3(year_min=y_min, year_max=y_max, dt=dt)
w.set_world3_control()
w.init_world3_constants(); w.init_world3_variables()
w.set_world3_table_functions(); w.set_world3_delay_functions()
w.run_world3(fast=False)

# Choose states:
# - Persistent POlLution (PPOL).
# - Population 0-14 (P1), Population 15-44 (P2), Population 45-64 (P3), Population 65+ (P4).
state = ['ppol', 'p1', 'p2', 'p3', 'p4']
# Choose Controls:
# - Persistent Pollution Generation Factor (PPGF), Persistent Pollution Transmission Delay (PPTD).
# - Lifetime Multiplier from Health Services (LMHS).
ctrl = ['ppgf', 'pptd', 'lmhs']

X = np.column_stack([getattr(w,s) for s in state]).astype(np.float32)
U = np.column_stack([getattr(w,c) for c in ctrl]).astype(np.float32)
t = np.arange(len(X), dtype=np.float32).reshape(-1,1) * dt # relative time

# Scale data
Sx, Su = X.max(0, keepdims=True), U.max(0, keepdims=True)
Xs, Us = X/Sx, U/Su

# –– 1. Split into training / test sets (70 % / 30 %) -------------------------
split = int(0.7*len(X))
Xtr, Utr, = Xs[:split], Us[:split] # only scale training data
Xte, Ute = X[split:], U[split:]
ttr, tte = t[:split], t[split:]

# -- 2. PINN-SR components ----------------------------------------------------
device = 'cpu'

class MLP(nn.Module): # x(t) predictor
    def __init__(self, dim_out, width=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(1, width), nn.Tanh(),
            nn.Linear(width, width), nn.Tanh(),
            nn.Linear(width, dim_out)
        )
    def forward(self, tau): # tau shape: (N,1)
        return self.net(tau)

def poly_library(x, u):
    """quadratic library Phi = [1, x, u, x^2, u^2, x dot u]"""
    ones = torch.ones_like(x[:, :1])
    xu = torch.einsum('bi,bj->bij', x, u).reshape(x.size(0), -1) # outer prod
    lib = [ones, x, u, x**2, u**2, xu]
    return torch.cat(lib, dim=1) # (N, n_feat)

n_state, n_ctrl = X.shape[1], U.shape[1]
n_feat = 1 + 2*n_state + 2*n_ctrl + n_state*n_ctrl # size of library

net = MLP(n_state).to(device)
Xi = nn.Parameter(torch.zeros(n_feat, n_state, device=device)) # sparse coeffs

# –– 3. Training loop ---------------------------------------------------------
Xtr_t = torch.tensor(Xtr, device=device)
Utr_t = torch.tensor(Utr, device=device)
ttr_t = torch.tensor(ttr, device=device, requires_grad=True)

# ------------ hyper-settings --------------
n_outer = 600 # total alternations
net_steps = 5 # theta-updates per outer loop
xi_steps = 5 # Xi-updates per outer loop
lr_theta = 3e-4
lr_xi = 1e-2
lambda_pde, lambda_l1 = 10.0, 1e-3 # residual & sparsity weights
# ------------------------------------------

opt_theta = optim.Adam(net.parameters(), lr=lr_theta)
opt_xi = optim.Adam([Xi], lr=lr_xi)

for k in range(n_outer):
    # -- A) update network weights theta -------
    Xi.requires_grad_(False)
    for _ in range(net_steps):
        opt_theta.zero_grad()
        x_hat = net(ttr_t) # NN output
        x_dot = torch.autograd.grad(x_hat, ttr_t,
                                    torch.ones_like(x_hat),
                                    create_graph=True)[0]
        Phi = poly_library(x_hat, Utr_t).detach() # fix Phi
        res = x_dot - Phi @ Xi
        loss_theta = (x_hat - Xtr_t).pow(2).mean() + lambda_pde*res.pow(2).mean()
        loss_theta.backward()
        opt_theta.step()

    # -- B) update sparse coefficients Xi ------
    # 1) freeze network weights
    for p in net.parameters():
        p.requires_grad_(False)
    Xi.requires_grad_(True)

    for _ in range(xi_steps):
        opt_xi.zero_grad()

        # 2) forward pass (still tracks grad w.r.t. ttr_t, not theta)
        x_hat = net(ttr_t)  
        x_dot = torch.autograd.grad(
            x_hat, ttr_t,
            grad_outputs=torch.ones_like(x_hat),
            create_graph=False)[0]

        # 3) build the library (as a constant)
        Phi = poly_library(x_hat.detach(), Utr_t) # detach to freeze graph

        # 4) residual and sparse loss
        res = x_dot - Phi @ Xi
        loss_xi = lambda_pde * res.pow(2).mean() + lambda_l1 * Xi.abs().mean()

        # 5) backward & step (fresh graph each iteration)
        loss_xi.backward()
        opt_xi.step()

    # 6) un-freeze network for next tehta–update
    for p in net.parameters():
        p.requires_grad_(True)

    if k % 50 == 0:
        print(f"outer {k:4d}  data-loss {loss_theta.item():.2e}  "
              f"res-loss {lambda_pde*res.pow(2).mean().item():.2e}")

# -- show non-zero coefficients ------------
Xi_np = Xi.detach().cpu().numpy()
print("\nNon-zero Xi rows:")
print(pd.DataFrame(Xi_np, columns=state).loc[(Xi_np!=0).any(1)])

# –– 3. Simulate with phi·Xi --------------------------------------------------
def dxdt(x, u):
    # scale down
    x_s = x / Sx.squeeze()   # shape (5,)
    u_s = u / Su.squeeze()   # shape (3,)

    # library on the scaled point
    Phi = poly_library(
        torch.tensor(x_s, dtype=torch.float32).unsqueeze(0),
        torch.tensor(u_s, dtype=torch.float32).unsqueeze(0),
    )

    # compute scaled derivative and rescale back
    dxs = (Phi @ Xi).squeeze(0).detach().numpy()      # d x_s / dt
    return dxs * Sx.squeeze()                         # d x / dt

def euler_sim(x0, Useq):
    x_hist = [x0]
    for k,u in enumerate(Useq[:-1]):
        x_next = x_hist[-1] + dt * dxdt(x_hist[-1], u)
        x_hist.append(x_next)
    return np.vstack(x_hist)

X_pred_te = euler_sim(Xte[0], Ute) # test prediction
print(X_pred_te)

# –– 4. Reporting & plots -----------------------------------------------------
rmse_rel = np.sqrt(((X_pred_te-Xte)**2).mean(axis=0))
print("\nRMSE:")
print(pd.Series(rmse_rel*100, index=state).round(2))

rel_rmse = np.sqrt(((X_pred_te - Xte)**2).mean(axis=0)) / (np.abs(Xte).max(axis=0))
print("\nRelative RMSE-1 (%%):")
print(pd.Series(rel_rmse*100, index=state).round(2))

rel_rmse = np.sqrt(((X_pred_te - Xte)**2).mean(axis=0)) / (np.abs(Xte).max(axis=0).max(axis=0))
print("\nRelative RMSE-2 (%%):")
print(pd.Series(rel_rmse*100, index=state).round(2))

# ------------ plot test window ------------
t_plot = t[split:].flatten() + y_min
colors = plt.get_cmap('tab10', len(state))

plt.figure(figsize=(12,6))
for i,s in enumerate(state):
    plt.plot(t_plot, Xte[:,i],'--', color=colors(i), label=f'{s} true')
    plt.plot(t_plot, X_pred_te[:,i],  color=colors(i), label=f'{s} pred')
plt.axvline(y_min+split*dt, color='k', ls=':', lw=1)
plt.title('PINN-SR - test set'); plt.xlabel('Year'); plt.grid()
plt.legend(bbox_to_anchor=(1.05,1), loc='upper left'); plt.tight_layout(); plt.show()

# ------------ plot full window ------------
X_pred_full = euler_sim(X[0], U) # full run
print(X_pred_full)

t_full = t.flatten() + y_min

plt.figure(figsize=(12,6))
for i,s in enumerate(state):
    plt.plot(t_full, X[:,i],'--', color=colors(i), label=f'{s} true')
    plt.plot(t_full, X_pred_full[:,i], color=colors(i), label=f'{s} pred')
plt.axvline(y_min+split*dt, color='k', ls=':', lw=1)
plt.title('PINN-SR - full trajectory'); plt.xlabel('Year'); plt.grid()
plt.legend(bbox_to_anchor=(1.05,1), loc='upper left'); plt.tight_layout(); plt.show()

rmse_rel  = np.sqrt(((X_pred_full-X)**2).mean(axis=0))
print("\nFull-traj RMSE:")
print(pd.Series(rmse_rel*100, index=state).round(2))

rel_rmse = np.sqrt(((X_pred_full - X)**2).mean(axis=0)) / (np.abs(X).max(axis=0))
print("\nFull-traj Relative RMSE-1 (%%):")
print(pd.Series(rel_rmse*100, index=state).round(2))

rel_rmse = np.sqrt(((X_pred_full - X)**2).mean(axis=0)) / (np.abs(X).max(axis=0).max(axis=0))
print("\nFull-traj Relative RMSE-2 (%%):")
print(pd.Series(rel_rmse*100, index=state).round(2))