import numpy as np, pandas as pd, pysindy as ps
from pyworld3 import World3
import matplotlib.pyplot as plt

from sklearn.metrics import r2_score


# –– RK4-SINDy class -----------------------------------------------------------
class RK4SINDy:
    def __init__(self, feature_library, threshold=0.1, max_iter=10):
        self.library   = feature_library
        self.threshold = threshold
        self.max_iter  = max_iter

    def fit(self, X, u, t):
        self.dt = t

        # Stack states and controls as a single input to the library
        Z = np.hstack([X, u])
        self.library.fit(Z)
        Theta_full = self.library.transform(Z)

        # Build RK4-integration design matrix
        Theta_i = Theta_full[:-1]
        Theta_ip1 = Theta_full[1:]
        X_mid = (X[:-1] + X[1:]) / 2
        U_mid = (u[:-1] + u[1:]) / 2
        Z_mid = np.hstack([X_mid, U_mid])
        Theta_mid = self.library.transform(Z_mid)

        B = self.dt / 6 * (Theta_i + 4*Theta_mid + Theta_ip1)
        Y = X[1:] - X[:-1]

        # Sequential threshold least-squares
        Xi = np.linalg.lstsq(B, Y, rcond=None)[0]
        for _ in range(self.max_iter):
            small = np.abs(Xi) < self.threshold
            Xi[small] = 0
            for j in range(Y.shape[1]):
                inds = ~small[:, j]
                if np.any(inds):
                    Xi[inds, j] = np.linalg.lstsq(B[:, inds], Y[:, j], rcond=None)[0]

        self.coefficients = Xi

    def print(self):
        names = self.library.get_feature_names()
        for i in range(self.coefficients.shape[1]):
            coefs = self.coefficients[:, i]
            terms = [f"{c:.5f}*{n}" for c, n in zip(coefs, names) if abs(c) > 0]
            print(f"dx_{i}/dt = " + " + ".join(terms))

    def score(self, X, u, t):
        # true derivatives by forward difference
        f_true = (X[1:] - X[:-1]) / t
        Z = np.hstack([X, u])
        Theta = self.library.transform(Z)
        f_pred = Theta.dot(self.coefficients)[:-1]
        scores = [r2_score(f_true[:, j], f_pred[:, j]) for j in range(f_true.shape[1])]
        return np.mean(scores)

    def simulate(self, x0, t_vec, u):
        dt = self.dt
        n_steps = len(t_vec)
        n_states = x0.size
        Xp = np.zeros((n_steps, n_states))
        Xp[0] = x0

        for i in range(n_steps - 1):
            def f(x, u_val):
                z = np.hstack([x, u_val]).reshape(1, -1)
                return self.library.transform(z).dot(self.coefficients).flatten()

            k1 = f(Xp[i], u[i])
            k2 = f(Xp[i] + dt/2*k1, (u[i] + u[i+1])/2)
            k3 = f(Xp[i] + dt/2*k2, (u[i] + u[i+1])/2)
            k4 = f(Xp[i] + dt*k3, u[i+1])

            Xp[i+1] = Xp[i] + dt/6*(k1 + 2*k2 + 2*k3 + k4)

        return Xp


# –– 0. Simulate standard World3 run ––-----------------------------------------
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
ctrl = ['ppgf', 'pptd', 'lmhs']

X = np.column_stack([getattr(w, s) for s in state])
U = np.column_stack([getattr(w, c) for c in ctrl])

# –– 1. Split into training / test sets (70 % / 30 %) --------------------------
split = int(0.7 * len(X))
X_train, X_test = X[:split], X[split:]
U_train, U_test = U[:split], U[split:]

# –– 2. Fit RK4-SINDy ––--------------------------------------------------------
lib = ps.PolynomialLibrary(degree=2)
model = RK4SINDy(feature_library=lib, threshold=0.001, max_iter=100)
model.fit(X_train, U_train, dt)
model.print()
print("\nTraining score: %.4f" % model.score(X_train, U_train, dt))
print("\nTest score: %f" % model.score(X_test, U_test, dt))
print("\nModel score: %f" % model.score(X, U, dt))

# -- 3. forecast 150 yr & compute RMSE -----------------------------------------
n_pred = len(X_test)
t_vec  = np.arange(n_pred)*dt
X_pred = model.simulate(X_test[0], t_vec, U_test)

# If any of the state variables are negative, set them to zero
X_pred[X_pred < 0] = 0

rmse = np.sqrt(((X_pred[:-1]-X_test[:-1])**2).mean(axis=0))
print("\nRMSE (%):")
print(pd.Series(rmse*100, index=state).round(2))

# rel1 = rmse/np.abs(X_test[:-1]).max(axis=0)
# print("\nRelative RMSE-1 (%):")
# print(pd.Series(rel1*100, index=state).round(2))

rel2 = rmse/np.abs(X_test[:-1]).max()
print("\nRelative RMSE-2 (%):")
print(pd.Series(rel2*100, index=state).round(2))

# –– 3. Plot -------------------------------------------------------------------
t_start = y_min + split*dt
t_plot  = t_vec[:-1] + t_start
n_vars = len(state)

# Generate a color map
colors = plt.get_cmap('tab10', len(state))

# Plot each state variable: true vs predicted
plt.figure(figsize=(12,6))
for i,name in enumerate(state):
    color = colors(i)
    plt.plot(t_plot, X_test[:-1,i], '--', label=f"{name} (true)", color=color)
    plt.plot(t_plot, X_pred[:-1,i], '-', label=f"{name} (pred)", color=color)

plt.xlabel("Year")
plt.ylabel("State value")
plt.title("Predicted vs. True Trajectories (test set)")
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', ncol=1)
plt.tight_layout()
plt.grid(True)
plt.show()

# –– 4. Extra plot: whole data set (training + test) –– ------------------------
t_full = np.arange(len(X))*dt + y_min
X_pred_full = model.simulate(X[0], t_full, U)

# If any of the state variables are negative, set them to zero
X_pred_full[X_pred_full < 0] = 0

t_plot_full = t_full[:-1]

plt.figure(figsize=(12,6))
for i,name in enumerate(state):
    color = colors(i)
    plt.plot(t_plot_full, X[:-1,i], '--', label=f"{name} (true)", color=color)
    plt.plot(t_plot_full, X_pred_full[:-1,i], '-',  label=f"{name} (pred)",  color=color)

# vertical line at the split year
plt.axvline(1900 + split * dt, color='k', linestyle=':', lw=1, label='train / test split')

plt.xlabel("Year")
plt.ylabel("State value")
plt.title("Predicted vs. True Trajectories")
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', ncol=1)
plt.tight_layout()
plt.grid(True)
plt.show()