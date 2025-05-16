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

state = ['ppol','p1','p2','p3','p4']
ctrl  = ['ppgf','pptd','lmhs']

X  = np.column_stack([getattr(w,s) for s in state]).astype(np.float32)
U  = np.column_stack([getattr(w,c) for c in ctrl]).astype(np.float32)
t  = np.arange(len(X), dtype=np.float32).reshape(-1,1) * dt # relative time

Sx, Su = X.max(0, keepdims=True), U.max(0, keepdims=True)
Xs, Us = X/Sx, U/Su

split = int(0.7*len(X))
Xtr, Xte = Xs[:split], Xs[split:]
Utr, Ute = Us[:split], Us[split:]
ttr, tte = t[:split], t[split:]

# -- 1. PINN-SR components ----------------------------------------------------
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
    xu   = torch.einsum('bi,bj->bij', x, u).reshape(x.size(0), -1) # outer prod
    lib  = [ones, x, u, x**2, u**2, xu]
    return torch.cat(lib, dim=1) # (N, n_feat)

n_state, n_ctrl = X.shape[1], U.shape[1]
n_feat  = 1 + 2*n_state + 2*n_ctrl + n_state*n_ctrl # size of library

net  = MLP(n_state).to(device)
Xi   = nn.Parameter(torch.zeros(n_feat, n_state, device=device)) # sparse coeffs

# –– 2. Training loop ---------------------------------------------------------
Xtr_t  = torch.tensor(Xtr,  device=device)
Utr_t  = torch.tensor(Utr,  device=device)
ttr_t  = torch.tensor(ttr,  device=device, requires_grad=True)

optimizer = optim.Adam(list(net.parameters())+[Xi], lr=1e-3)
lambda_pde, lambda_l1 = 1.0, 1e-4 # weights for residual & sparsity

for epoch in range(3000):
    optimizer.zero_grad()

    x_pred = net(ttr_t) # NN output
    # autograd d/dt (vector-Jacobian)
    x_dot  = torch.autograd.grad(x_pred, ttr_t,
                                 grad_outputs=torch.ones_like(x_pred),
                                 create_graph=True)[0]

    Phi = poly_library(x_pred, Utr_t)
    res = x_dot - Phi @ Xi # PDE residual

    loss = (x_pred - Xtr_t).pow(2).mean() \
         + lambda_pde * res.pow(2).mean() \
         + lambda_l1 * Xi.abs().mean()

    loss.backward()
    optimizer.step()

    # soft thresholding (simple sparsity enforcement)
    with torch.no_grad():
        Xi.data[Xi.abs() < 0.02] = 0.

    if epoch % 500 == 0:
        print(f"epoch {epoch:4d}  loss={loss.item():.3e}")

Xi_np = Xi.detach().cpu().numpy()
print("\nSparse Xi coefficients (non-zero):")
print(pd.DataFrame(Xi_np, columns=state).loc[(Xi_np!=0).any(1)])

# –– 3. Simulate with phi·Xi --------------------------------------------------
def dxdt(x,u):
    x,u = torch.tensor(x), torch.tensor(u)
    return (poly_library(x.unsqueeze(0),u.unsqueeze(0)) @ Xi).squeeze(0).detach().numpy()

def euler_sim(x0, Useq):
    x_hist = [x0]
    for k,u in enumerate(Useq[:-1]):
        x_next = x_hist[-1] + dt * dxdt(x_hist[-1], u)
        x_hist.append(x_next)
    return np.vstack(x_hist)

X_pred_te = euler_sim(Xte[0], Ute) # test prediction
rmse_rel  = (np.sqrt(((X_pred_te-Xte)**2).mean(0))
            / np.abs(Xte).max(0))

# –– 4. Reporting & plots -----------------------------------------------------
print("\nRelative RMSE (%%):")
print(pd.Series(rmse_rel*100, index=state).round(2))

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
t_full = t.flatten() + y_min

plt.figure(figsize=(12,6))
for i,s in enumerate(state):
    plt.plot(t_full, X[:,i],'--', color=colors(i), label=f'{s} true')
    plt.plot(t_full, X_pred_full[:,i], color=colors(i), label=f'{s} pred')
plt.axvline(y_min+split*dt, color='k', ls=':', lw=1)
plt.title('PINN-SR - full trajectory'); plt.xlabel('Year'); plt.grid()
plt.legend(bbox_to_anchor=(1.05,1), loc='upper left'); plt.tight_layout(); plt.show()