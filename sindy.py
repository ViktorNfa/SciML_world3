import numpy as np, pandas as pd, pysindy as ps
from pyworld3 import World3


# -- 0. simulate standard World3 run -----------------------------------------
w = World3(year_min=1900, year_max=2100, dt=0.1)
w.set_world3_control()
w.init_world3_constants(); w.init_world3_variables()
w.set_world3_table_functions(); w.set_world3_delay_functions()
w.run_world3(fast=False)


# State variables -----------------------------------------
# - Arable Land (AL), Potentially Arable Land (PAL), Urban-Industrial Lan (UIL), Land FERTility (LFERT).
# - Industrial Capital (IC), Service Capital (SC).
# - Persistent POlLution (PPOL).
# - Population 0-14 (P1), Population 15-44 (P2), Population 45-64 (P3), Population 65+ (P4).
# - Natural Resources (NR).

# Control variables -----------------------------------------
# - Average Lifetime of Agricultural Inputs (ALAI), Loss Yield Factor (LYF).
# Indicated Food per Capita (IFPC), Land Yield Multiplier from Air Pollution (LYMAP).
# Land Life Multiplier from Yield (LLMY).
# Fraction of Industrial Output Allocated to Agriculture (FIOAA).

# - Industrial Capital-Output ratio (ICOR), Service Capital-Output ratio (SCOR).
# Average Lifetime of Industrial Capital (ALIC), Average Lifetime of Service Capital (ALSC).
# Fraction of Industrial Output Allocated to Consumption (FIOAC).
# Indicated Service Output Per Capita (ISOPC), Fraction of Industrial Output allocated to Services (FIOAS).

# - Persistent Pollution Generation Factor (PPGF), Persistent Pollution Transmission Delay (PPTD).

# - Lifetime Multiplier from Health Services (LMHS).

# - Nonrenewable Resource Usage Factor (NRUF), Fraction of Capital Allocated to Obtaining Resources (FCAOR).

state = []
ctrl  = []

# Agriculture
state += ['al','pal','uil','lfert']
ctrl  += ['alai','lyf','ifpc','lymap','llmy','fioaa']

# Capital sector
# state += ['ic','sc']
# ctrl  += ['icor','scor','alic','alsc','fioac','isopc','fioas']

# Pollution
state += ['ppol']
ctrl  += ['ppgf','pptd']

# Population
state += ['p1','p2','p3','p4']
ctrl  += ['lmhs']

# Resource sector
# state += ['nr']
# ctrl  += ['nruf','fcaor']

X = np.column_stack([getattr(w,s) for s in state])   # (T,12)
U = np.column_stack([getattr(w,c) for c in ctrl])    # (T,18)
dt = w.dt

# -- 1. fit with controls -----------------------------------------------------
model   = ps.SINDy()
model.fit(X, u=U, t=dt)
model.print()

# -- 2. forecast 150 yr & compute *relative* RMSE -----------------------------
n_pred = 151
X_pred = model.simulate(X[0], t=np.arange(n_pred)*dt, u=U[:n_pred])
rel_rmse = np.sqrt(((X_pred - X[:n_pred-1])**2).mean(axis=0)) / (np.abs(X[:n_pred-1]).max(axis=0))
print("\nRelative RMSE (%%):")
print(pd.Series(rel_rmse*100, index=state).round(2))

print("Model score: %f" % model.score(X, u=U, t=dt))