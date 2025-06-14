# kkl_demo_world3.py  – sequential PINN training demo

import numpy as np, matplotlib.pyplot as plt
from pyworld3 import World3
from kkl_observer import KKLObserver

# ------------------------------------------------ Data ------------------------------------------------
y0, y1, dt = 1900, 2200, 1.0
w = World3(year_min=y0, year_max=y1, dt=dt)
w.set_world3_control();  w.init_world3_constants();  w.init_world3_variables()
w.set_world3_table_functions();  w.set_world3_delay_functions();  w.run_world3()

state = ["ppol", "p2", "p3", "p4"]
meas  = ["p2", "p3"]
ctrl  = ["ppgf", "lmhs"]

X = np.column_stack([getattr(w, s) for s in state])
Y = np.column_stack([getattr(w, m) for m in meas])      # h(x)
# Simple z-dim choice nz = ny + 2*nx + 1 = 2 + 8 + 1 = 11
nz = 11

# When World-3 variables differ by orders of magnitude we pre-scale
mu, sigma = X.mean(axis=0), X.std(axis=0)
Xn   = (X - mu) / sigma
Yn   = (Y - Y.mean(axis=0)) / Y.std(axis=0)

# ------------------------------------------------ Observer ------------------------------------------------
A = -np.diag(np.linspace(0.3, 0.3+0.05*(nz-1), nz))     # stable diagonal
print("Eigen-values(A):", np.linalg.eigvals(A))
B = np.zeros((nz, Yn.shape[1]));  B[:Yn.shape[1], :] = np.eye(Yn.shape[1])   # controllable (A,B)
C_dummy = np.zeros((Yn.shape[1], nz))                   # not used here but kept for completeness

obs = KKLObserver(A=A, B=B, C=C_dummy,
                  state_names=state, meas_names=meas,
                  v_phys=1e-2)

# ---- Stage-1: learn T -----------------------------------------------------
obs.fit_T(Xn, Yn, epochs=1000, lr=2e-3, t_trunc=30)

# ---- Stage-2: learn inverse ----------------------------------------------
obs.fit_Tinv(Xn, epochs=1000, lr=2e-3)

# ------------------------------------------------ Test roll-outs ----------------------------------------------
perc_err_runs = []
rng = np.random.default_rng(0)
for run in range(3):
    k0 = rng.integers(100, Xn.shape[0] - 120)
    obs.z = None
    errs = []
    for k in range(k0, k0+120):
        yk = Yn[k]
        x_hat = obs.step(yk, dt)
        num = np.linalg.norm((x_hat - Xn[k]) * sigma)   # back-scale for fairness
        den = np.linalg.norm(X[k]) + 1e-8
        errs.append(100 * num / den)                    # percentage error
    perc_err_runs.append(errs)

# ------------------------------------------------ Plot ------------------------------------------------
plt.figure(figsize=(7,3))
for r,e in enumerate(perc_err_runs):
    plt.plot(e, label=f'run {r+1}')
plt.xlabel('years (Δt=1)');  plt.ylabel('% estimation error');  plt.title('KKL observer - percentage error')
plt.grid(True, which='both', ls=':');  plt.legend();  plt.tight_layout();  plt.show()