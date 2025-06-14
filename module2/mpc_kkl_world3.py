# mpc_kkl_world3.py  ----------------------------------------------------------
"""
Closed-loop World-3 demo:
  • KKL observer  (measure p2, p3)
  • linear model  z⁺ = A_d z + B_d u   identified in z-space
  • on-line MPC   (H = 10 years) that keeps ppol ≈ ppol_1970
The MPC is solved *every* simulation step inside the World-3
feedback callbacks  ppgf_control(t, world, k)  and  lmhs_control(…).
"""

import numpy as np
import matplotlib.pyplot as plt
import cvxpy as cp, torch
from   pyworld3 import World3
from   kkl_observer import KKLObserver            # ← your working file

# ════════════════════════════════════════════════════════════════════════
# 0)  open-loop run  →  data  (1900-2200)  &  normalisers
# ════════════════════════════════════════════════════════════════════════
y0, y1, dt = 1900, 2200, 1.0
w_train = World3(year_min=y0, year_max=y1, dt=dt)
w_train.set_world3_control()
w_train.init_world3_constants(); w_train.init_world3_variables()
w_train.set_world3_table_functions(); w_train.set_world3_delay_functions()
w_train.run_world3()

state = ["ppol", "p2", "p3", "p4"]
meas  = ["p2", "p3"]                       # sensors
ctrl  = ["ppgf", "lmhs"]                   # two levers

X_raw = np.column_stack([getattr(w_train, s) for s in state])
Y_raw = np.column_stack([getattr(w_train, m) for m in meas])

X_mu, X_std = X_raw.mean(0), X_raw.std(0)
Y_mu, Y_std = Y_raw.mean(0), Y_raw.std(0)

Xn = (X_raw - X_mu)/X_std
Yn = (Y_raw - Y_mu)/Y_std

# ════════════════════════════════════════════════════════════════════════
# 1)  KKL observer  (4-state slice  →  4-dim z)
# ════════════════════════════════════════════════════════════════════════
A = -np.diag([.3, .4, .5, .6])             # stable diagonal
B = np.eye(4, 2)                           # simple, controllable
C_dummy = np.zeros((2, 4))

obs = KKLObserver(A=A, B=B, C=C_dummy,
                  state_names=state, meas_names=meas, v_phys=1e-2)
print('[KKL] training   T,  T⁻¹  …')
obs.fit_T   (Xn, Yn, epochs=800, lr=2e-3, t_trunc=30)
obs.fit_Tinv(Xn,        epochs=800, lr=2e-3)

def lift_x_to_z(xn):
    with torch.no_grad():
        return obs.T(torch.as_tensor(xn, dtype=torch.float32)).numpy()

# ════════════════════════════════════════════════════════════════════════
# 2)  identify  z⁺ ≈ A_d z + B_d u    (use a mild input excitation)
# ════════════════════════════════════════════════════════════════════════
def excite_u(t):
    return np.column_stack([0.8 + 0.2*np.sin(0.05*t),
                            0.9 + 0.3*np.sin(0.07*t+1.0)])

N = len(Xn)-1
U_ex = excite_u(np.arange(N))

Zk, Zkp1 = [], []
for k in range(N):
    Zk.append (lift_x_to_z(Xn[k]))
    Zkp1.append(lift_x_to_z(Xn[k+1]))
Zk, Zkp1 = np.vstack(Zk), np.vstack(Zkp1)

Phi   = np.hstack([Zk, U_ex])
theta, *_ = np.linalg.lstsq(Phi, Zkp1, rcond=None)
nz, nu = 4, 2
A_d = theta[:nz , :].T
B_d = theta[nz:, :].T
print('[ID]  ||A_d||₂ = %.3f   ||B_d||₂ = %.3f'
      % (np.linalg.norm(A_d), np.linalg.norm(B_d)))

# ════════════════════════════════════════════════════════════════════════
# 3)  MPC setup (solved online)
# ════════════════════════════════════════════════════════════════════════
H   = 20                                     # horizon  (years)
Qz  = np.diag([10.0, 1, 1, 1])               # weigh pollution heavily
Ru  = 1e-3*np.eye(2)
umin, umax = np.array([0.5,0.6]), np.array([2000.0, 1000.3])

# reference = pollution at 1970  (in z-coordinates)
k_ref      = int((1970-y0)/dt)
z_ref      = lift_x_to_z(Xn[k_ref])

# ------------------------------------------------------------------
#  one-shot quadratic MPC  (penalises estimated pollution directly)
# ------------------------------------------------------------------
def solve_mpc(z_now):
    """
    Return optimal input u_now (shape (2,)) for current lifted state z_now.
    All intermediate PyTorch tensors are detached before .numpy() so that
    cvxpy receives plain NumPy arrays (no autograd graph attached).
    """
    # ---------- parameters -----------------------------------------------
    z0 = cp.Parameter(nz,  value=z_now)
    ppol_ref = 0.0                         # pollution target (normalised)

    # ---------- build (and cache)  ppol_row  ------------------------------
    if not hasattr(solve_mpc, "_ppol_row"):
        eps       = 1e-4
        ppol_row  = np.zeros(nz)
        with torch.no_grad():
            for j in range(nz):
                dz        = np.zeros(nz);  dz[j] = eps
                z_plus     = torch.as_tensor((z_ref + dz)[None, :],
                                            dtype=torch.float32)
                z_minus    = torch.as_tensor((z_ref - dz)[None, :],
                                            dtype=torch.float32)

                p_plus  = obs.T_inv(z_plus).detach().numpy()[0, 0]
                p_minus = obs.T_inv(z_minus).detach().numpy()[0, 0]
                ppol_row[j] = (p_plus - p_minus) / (2 * eps)
        solve_mpc._ppol_row = ppol_row

    p_row = solve_mpc._ppol_row            # shape (nz,)

    # ---------- decision variables ---------------------------------------
    U = cp.Variable((nu, H))
    Z = cp.Variable((nz, H + 1))

    # ---------- objective & constraints ----------------------------------
    cost        = 0
    constraints = [Z[:, 0] == z0]

    for k in range(H):
        ppol_k = p_row @ Z[:, k]
        cost  += 100 * cp.square(ppol_k - ppol_ref)      # tracking term
        cost  += cp.quad_form(U[:, k], Ru)               # effort term
        constraints += [
            Z[:, k + 1] == A_d @ Z[:, k] + B_d @ U[:, k],
            umin <= U[:, k], U[:, k] <= umax,
        ]

    # terminal stage
    ppol_H = p_row @ Z[:, H]
    cost  += 1000 * cp.square(ppol_H - ppol_ref)

    # ---------- solve -----------------------------------------------------
    prob = cp.Problem(cp.Minimize(cost), constraints)
    prob.solve(solver=cp.OSQP, warm_start=True)

    if prob.status not in ("optimal", "optimal_inaccurate"):
        raise RuntimeError("MPC QP did not solve!")

    return np.asarray(U.value[:, 0])        # first control move


# ════════════════════════════════════════════════════════════════════════
# 4)  feedback controller object  ========================================
# ════════════════════════════════════════════════════════════════════════
class MPCController:
    def __init__(self):
        self.u_prev = np.array([1.0,1.0])     # start-up default
        self.t_prev = None

    def step(self, t, world, k):
        """return 2-vector   [ppgf , lmhs]   for year-index k"""
        global obs

        # -- measurement (p2,p3)  normalised
        y_raw = np.array([world.p2[k], world.p3[k]])
        y_n   = (y_raw - Y_mu) / Y_std

        # -- KKL observer update  (uses last control)
        x_hat = obs.step(y_n, self.u_prev, dt)

        # -- lift -> z,   MPC
        z_hat = lift_x_to_z(x_hat)
        u_now = solve_mpc(z_hat)

        # store + give to plant
        self.u_prev = u_now
        self.t_prev = t
        return u_now


ctrl = MPCController()

# helper wrappers – World-3 calls them separately
ppgf_cb = lambda t, w, k: float( ctrl.step(t, w, k)[0] )
lmhs_cb = lambda t, w, k: float( ctrl.step(t, w, k)[1] )

# ════════════════════════════════════════════════════════════════════════
# 5)  closed-loop run
# ════════════════════════════════════════════════════════════════════════
w3 = World3(year_min=1900, year_max=2100, dt=1.0)
w3.set_world3_control(ppgf_control = ppgf_cb,
                      lmhs_control = lmhs_cb)
w3.init_world3_constants(); w3.init_world3_variables()
w3.set_world3_table_functions(); w3.set_world3_delay_functions()

print('▶ running World-3 closed loop …')
w3.run_world3(fast=False)                    # the feedback is active!

# ════════════════════════════════════════════════════════════════════════
# 6)  quick plots
# ════════════════════════════════════════════════════════════════════════
yrs = np.arange(1900, 2101)

plt.figure(figsize=(7,4))
plt.subplot(2,1,1)
plt.plot(yrs, w3.ppol, label='ppol')
plt.axhline(w3.ppol[k_ref], ls='--', c='k', label='ppol @1970')
plt.ylabel('persistent pollution'); plt.legend(); plt.grid(ls=':')

plt.subplot(2,1,2)
plt.step(yrs, w3.ppgf_control_values, where='post', label='ppgf')
plt.step(yrs, w3.lmhs_control_values, where='post', label='lmhs')
plt.xlabel('year'); plt.ylabel('control'); plt.legend(); plt.grid(ls=':')
plt.tight_layout(); plt.show()