#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Fast 2-input SciML MPC for World3
=================================

- Controls   : PPGF & PPTD
- Surrogate  : degree-2 polynomial (learned once, cached)
- MPC        : gradient descent, 3-year horizon, control updated every 4 steps
"""

import pathlib, pickle, numpy as np, torch, matplotlib.pyplot as plt
from pyworld3 import World3


# ---------------------- Tunables (speed & quality) ----------------------
DT            = 0.5                      # [yr] sim & data step
T_DATA        = (1900, 2024)             # training period
T_SIM         = (2024, 2100)             # closed-loop run
HORIZON_YEARS = 3.0                      # MPC horizon
ADAM_STEPS    = 20                       # grad iterations per plan
MPC_SKIP      = 4                        # plan only every 4th step
CTRL_BOX      = (0.01, 4.0)              # bounds for both inputs
W_PP, W_POP, W_U = 1.0, 3e-2, 1e-5       # cost weights

# ---------------------- 1. Cached data and surrogate ----------------------
HERE = pathlib.Path(__file__).with_suffix("")
DATA_F = HERE / "world3_data.pkl"
THETA_F = HERE / "theta.pt"

if not DATA_F.exists(): # generate once
    print("- Generating training data …")
    w = World3(year_min=T_DATA[0], year_max=T_DATA[1], dt=DT)

    ppgf_exc = lambda t: 1 + 0.25*np.sin(0.15*t) + 0.15*np.sin(0.04*t + 1.1)
    pptd_exc = lambda t: 1 + 0.25*np.sin(0.12*t + 0.4) + 0.20*np.sin(0.08*t)

    w.set_world3_control(ppgf_control=ppgf_exc, pptd_control=pptd_exc)
    w.init_world3_constants(); w.init_world3_variables()
    w.set_world3_table_functions(); w.set_world3_delay_functions()
    w.run_world3(fast=False)

    X = np.column_stack([w.ppol, w.p1, w.p2, w.p3, w.p4])   # 5 states
    U = np.column_stack([w.ppgf, w.pptd])                   # 2 controls
    pickle.dump((X, U, DT), DATA_F.open("wb"))

if not THETA_F.exists(): # fit once
    print("- Fitting polynomial surrogate …")
    X, U, dt = pickle.load(DATA_F.open("rb"))
    dX = np.vstack([(X[1] - X[0]) / dt,
                    (X[2:] - X[:-2]) / (2*dt),
                    (X[-1] - X[-2]) / dt])
    nx, nu = X.shape[1], U.shape[1]

    def feats(x, u):
        f = list(x) + list(u)
        f += [x[i]*x[j] for i in range(nx) for j in range(i, nx)]
        f += [x[i]*u[j] for i in range(nx) for j in range(nu)]
        f += [u[p]*u[q] for p in range(nu) for q in range(p, nu)]
        return f

    phi = np.vstack([feats(x, u) for x, u in zip(X, U)]).astype(np.float32)
    theta, *_ = np.linalg.lstsq(phi, dX, rcond=None)
    torch.save({"Theta": theta, "nx": nx, "nu": nu}, THETA_F)

theta_pack = torch.load(THETA_F, weights_only=False)
theta, NX, NU = torch.tensor(theta_pack["Theta"].T), theta_pack["nx"], theta_pack["nu"]

# ---------------------- 2. Torch surrogate (degree-2 polynomial) ----------------------
class Poly(torch.nn.Module):
    def __init__(self, theta, nx, nu):
        super().__init__(); self.register_buffer("theta", theta); self.nx, self.nu = nx, nu
    def _phi(self, x, u):
        if x.ndim == 1: x = x.unsqueeze(0); u = u.unsqueeze(0)
        feats = [x, u]
        feats += [x[..., i, None]*x[..., j, None] for i in range(self.nx) for j in range(i, self.nx)]
        feats += [x[..., i, None]*u[..., j, None] for i in range(self.nx) for j in range(self.nu)]
        feats += [u[..., p, None]*u[..., q, None] for p in range(self.nu) for q in range(p, self.nu)]
        return torch.cat(feats, -1)
    def forward(self, x, u):
        phi = self._phi(x, u)
        dx  = torch.einsum("...k,ik->...i", phi, self.Θ)
        return dx.squeeze(0) if dx.shape[0] == 1 else dx

SURRO = Poly(theta.float(), NX, NU)

# ---------------------- 3. Fast MPC (re-plan every MPC_SKIP steps) ----------------------
class MPC:
    def __init__(self, f, dt):
        self.f, self.dt = f, torch.tensor(dt)
        self.N  = int(round(HORIZON_YEARS / dt))
        self.u  = torch.ones(self.N, NU, requires_grad=True)
        self.opt = torch.optim.Adam([self.u], lr=5e-2)
        self.pop_ref = None
        self.last_u = (1.0, 1.0)

    def _roll(self, x0):
        xs, x = [x0], x0
        for k in range(self.N):
            x = x + self.dt * self.f(x, self.u[k])
            xs.append(x)
        return torch.stack(xs)

    def plan(self, x0):
        x0 = torch.tensor(x0, dtype=torch.float32)
        if self.pop_ref is None:
            self.pop_ref = x0[1:].sum().item()
        for _ in range(ADAM_STEPS):
            self.opt.zero_grad()
            traj = self._roll(x0)
            ppol, pop = traj[:, 0], traj[:, 1:].sum(-1)
            cost = (W_PP * ppol.pow(2).sum()
                    + W_POP * (pop - self.pop_ref).pow(2).sum()
                    + W_U * self.u.pow(2).sum())
            cost.backward(); self.opt.step()
            with torch.no_grad(): self.u.clamp_(*CTRL_BOX)
        return self.u.detach().clone()

    def control(self, world, k):
        if k % MPC_SKIP != 0:
            return self.last_u
        if k == 0:
            return self.last_u

        x0 = np.array([world.ppol[k-1],
                       world.p1[k-1], world.p2[k-1],
                       world.p3[k-1], world.p4[k-1]], dtype=np.float32)
        u_now = self.plan(x0)[0].numpy()
        self.last_u = (float(u_now[0]), float(u_now[1]))

        # shift warm start (safe, non-overlapping write)
        with torch.no_grad():
            self.u[:-1].copy_(self.u[1:].clone())
            self.u[-1].copy_(self.u[-2])

        return self.last_u

# ---------------------- 4. Closed-loop World3 run ----------------------
def main():
    w = World3(year_min=T_SIM[0], year_max=T_SIM[1], dt=DT)
    mpc = MPC(SURRO, DT)

    w.set_world3_control(
        ppgf_control=lambda t, world, k: mpc.control(world, k)[0],
        pptd_control=lambda t, world, k: mpc.control(world, k)[1]
    )
    w.init_world3_constants(); w.init_world3_variables()
    w.set_world3_table_functions(); w.set_world3_delay_functions()
    print("• Running World3 with MPC …")
    w.run_world3(fast=False)

    # -- diagnostics --------------------------------------------------
    pop_ref = mpc.pop_ref
    pop_rmse = np.sqrt(np.mean((w.p1+w.p2+w.p3+w.p4 - pop_ref)**2))
    ppol_rmse = np.sqrt(np.mean(w.ppol**2))
    print(f"RMSE population wrt {pop_ref:,.0f} persons  : {pop_rmse:,.0f}")
    print(f"RMSE pollution wrt zero                    : {ppol_rmse:,.2f}")

    # -- plotting ------------------------------------------------------

    t = w.time
    plt.figure(figsize=(8, 4))
    plt.subplot(1, 2, 1); plt.plot(t, w.ppol); plt.title("PPOL"); plt.grid(); plt.xlabel("year")
    plt.subplot(1,2,2)
    plt.plot(t, w.p1+w.p2+w.p3+w.p4, label="Population")
    plt.hlines(pop_ref, t[0], t[-1], colors="k", linestyles="--",
            label="Pop ref")
    plt.title("Population"); plt.xlabel("year"); plt.grid(); plt.legend()
    plt.tight_layout(); plt.show()

    plt.figure()
    plt.plot(t, w.ppgf, label="PPGF")
    plt.plot(t, w.pptd, label="PPTD")
    plt.legend(); plt.grid(); plt.xlabel("year")
    plt.title("Applied Controls"); plt.tight_layout(); plt.show()

if __name__ == "__main__":
    main()