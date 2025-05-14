# world3_sindy_multiblock.py
import itertools
import numpy as np
import pandas as pd
import pysindy as ps
from pyworld3 import World3

# ────────────────────────────────────────────────────────────────────────────
# 0.  RUN A REFERENCE WORLD-3 TRAJECTORY
# ────────────────────────────────────────────────────────────────────────────
w = World3(year_min=1900, year_max=2100, dt=0.1)
w.set_world3_control()
w.init_world3_constants(); w.init_world3_variables()
w.set_world3_table_functions(); w.set_world3_delay_functions()
w.run_world3(fast=False)

dt      = w.dt
t_grid  = np.arange(len(getattr(w, "al"))) * dt   # common timeline

# ────────────────────────────────────────────────────────────────────────────
# 1.  DEFINE SECTORS
#     Every sector is a dict with a name, a list of *state* variables
#     and a list of *control* variables.
# ────────────────────────────────────────────────────────────────────────────
SECTORS = [
    dict(name="agriculture",
         states=['al', 'pal', 'uil', 'lfert', 'ppol', 'p1','p2','p3','p4'],
         ctrls =['alai','lyf','ifpc','lymap','llmy','fioaa']),
    
    dict(name="capital",
         states=['ic','sc'],
         ctrls =['icor','scor','alic','alsc','fioac','isopc','fioas']),
    
    dict(name="resources",
         states=['nr'],
         ctrls =['nruf','fcaor']),
]

# flatten lists for later bookkeeping
state_vars = list(itertools.chain.from_iterable(sec["states"] for sec in SECTORS))
ctrl_vars  = list(itertools.chain.from_iterable(sec["ctrls"]  for sec in SECTORS))

# full matrices
X_full = np.column_stack([getattr(w, s) for s in state_vars])
U_full = np.column_stack([getattr(w, c) for c in ctrl_vars])

# ────────────────────────────────────────────────────────────────────────────
# 2.  FIT ONE MODEL PER SECTOR
# ────────────────────────────────────────────────────────────────────────────
models   = {}          # sector name → fitted SINDy model
pred_mat = np.zeros_like(X_full)

state_offset = 0
ctrl_offset  = 0

for sec in SECTORS:
    s_names = sec["states"]
    c_names = sec["ctrls"]
    ns, nc  = len(s_names), len(c_names)

    X_sec = X_full[:, state_offset : state_offset + ns]
    U_sec = U_full[:, ctrl_offset  : ctrl_offset  + nc]

    model = ps.SINDy()
    model.fit(X_sec, u=U_sec if nc else None, t=dt)
    models[sec["name"]] = model
    print(f"\n### {sec['name'].upper()}  ({ns} states, {nc} ctrls)")
    model.print()

    # ---- simulate ----------------------------------------------------------
    X_hat_sec = model.simulate(X_sec[0], t_grid, u=U_sec if nc else None)
    rel_rmse = np.sqrt(((X_hat_sec - X_sec[:-1])**2).mean(axis=0)) / (np.abs(X_sec[:-1]).max(axis=0))
    print("\nRelative RMSE (%%):")
    print(pd.Series(rel_rmse*100, index=s_names).round(2))
    print("Model score: %f" % model.score(X_sec, u=U_sec, t=dt))

    pred_mat[1:, state_offset : state_offset + ns] = X_hat_sec
    state_offset += ns
    ctrl_offset  += nc

# ────────────────────────────────────────────────────────────────────────────
# 3.  ERROR METRICS
# ────────────────────────────────────────────────────────────────────────────
rel_rmse = (np.sqrt(((pred_mat[1:] - X_full[:-1])**2).mean(axis=0))
            / np.abs(X_full[:-1]).max(axis=0))

print("\n=== RELATIVE RMSE PER STATE (%) ===")
print(pd.Series(rel_rmse*100, index=state_vars).round(2))

overall = rel_rmse.mean()*100
print(f"\nOverall mean relative RMSE: {overall:.2f} %")