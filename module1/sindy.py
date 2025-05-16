import numpy as np, pandas as pd, pysindy as ps
from pyworld3 import World3
import matplotlib.pyplot as plt


# –– 0. simulate standard World3 run ––-----------------------------------------
y_min = 1900
y_max = 2200
dt = 0.1
w = World3(year_min=y_min, year_max=y_max, dt=dt)

# def exciting_signal(t):
#     return np.vstack([
#         1.0 + 0.2*np.sin(0.1*t),               # ppgf
#         1.0 + 0.3*np.sin(0.07*t + 1.0),        # pptd
#         1.0 + 0.1*np.sin(0.2*t + 2.0)          # lmhs
#     ]).T

# # Define exciting input functions for each control
# t_input = np.arange(y_min, y_max, dt)
# exciting_u = exciting_signal(t_input)

# w.set_world3_control(
#     ppgf_control = lambda t: float(np.interp(t, t_input, exciting_u[:,0])),
#     pptd_control = lambda t: float(np.interp(t, t_input, exciting_u[:,1])),
#     lmhs_control = lambda t: float(np.interp(t, t_input, exciting_u[:,2]))
# )

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
ctrl  = ['ppgf', 'pptd', 'lmhs']

X  = np.column_stack([getattr(w, s) for s in state])
U  = np.column_stack([getattr(w, c) for c in ctrl])
dt = w.dt

# –– 1. split into training / test sets (70 % / 30 %) --------------------------
split = int(0.7 * len(X))
X_train, X_test = X[:split], X[split:]
U_train, U_test = U[:split], U[split:]

# –– 2. fit SINDy on **training** data -----------------------------------------
model = ps.SINDy(feature_library=ps.PolynomialLibrary(degree=2))
model.fit(X_train, u=U_train, t=dt)
model.print()
print("\nTraining score: %.4f" % model.score(X_train, u=U_train, t=dt))
print("\nTest score: %f" % model.score(X_test, u=U_test, t=dt))
print("\nModel score: %f" % model.score(X, u=U, t=dt))

# -- 3. forecast 150 yr & compute *relative* RMSE -----------------------------
n_pred  = len(X_test)
t_vec   = np.arange(n_pred) * dt
X_pred  = model.simulate(X_test[0], t=t_vec, u=U_test)

rel_rmse = np.sqrt(((X_pred - X_test[:-1])**2).mean(axis=0)) / (np.abs(X_test[:-1]).max(axis=0))
print("\nRelative RMSE (%%):")
print(pd.Series(rel_rmse*100, index=state).round(2))

# –– 3. plot -------------------------------------------------------------------
# Create time vector (in years)
t_start = 1900 + split * dt 
t_plot  = t_vec[:-1] + t_start
n_vars = len(state)

# Generate a color map
colors  = plt.get_cmap('tab10', len(state))

# Plot each state variable: true vs predicted
plt.figure(figsize=(12, 6))
for i, name in enumerate(state):
    color = colors(i % colors.N)
    plt.plot(t_plot, X_test[:-1, i], label=f"{name} (true)",linestyle='--', color=color)
    plt.plot(t_plot, X_pred[:, i], label=f"{name} (pred)", linewidth=1.2, color=color)

plt.xlabel("Year")
plt.ylabel("State value")
plt.title("Predicted vs. True Trajectories (test set)")
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', ncol=1)
plt.tight_layout()
plt.grid(True)
plt.show()

# –– 4. EXTRA PLOT : whole data set (training + test) –– ----------------------
t_full = np.arange(len(X)) * dt + 1900
X_pred_full = model.simulate(X[0], t_full, u=U)

t_plot_full = t_full[:-1]

plt.figure(figsize=(12, 6))
for i, name in enumerate(state):
    color = colors(i % colors.N)
    plt.plot(t_plot_full, X[:-1, i], label=f"{name} (true)",linestyle='--', color=color)
    plt.plot(t_plot_full, X_pred_full[:, i], label=f"{name} (pred)", linewidth=1.2, color=color)

# vertical line at the split year
plt.axvline(1900 + split * dt, color='k', linestyle=':', lw=1, label='train / test split')

plt.xlabel("Year")
plt.ylabel("State value")
plt.title("Predicted vs. True Trajectories")
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', ncol=1)
plt.tight_layout()
plt.grid(True)
plt.show()