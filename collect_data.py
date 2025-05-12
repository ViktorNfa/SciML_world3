# collect_data.py  –  runs Business As Usual (BAU) and writes a tidy CSV
import numpy as np, pandas as pd
from pyworld3 import World3

# 0.  Build & initialise the model -------------------------------------------------
w3 = World3(year_min=1900, year_max=2100, dt=1)   # yearly grid, 201 points
w3.set_world3_control()
w3.init_world3_constants()
w3.init_world3_variables()
w3.set_world3_table_functions()
w3.set_world3_delay_functions()

# 1.  Integrate --------------------------------------------------------------------
w3.run_world3(fast=False)                         # backward-Euler solver

# 2.  Pack the data needed ---------------------------------------------------------
state_names = ['ic', 'sc', 'ppol', 'nr']          # 4-state slice for Module 1
state_arrays = [getattr(w3, s) for s in state_names]

df = pd.DataFrame(
        data = np.column_stack(state_arrays),
        columns = state_names,
        index = w3.time)                          # w3.time already in *years*

df.to_csv('world3_BAU_1900-2100.csv')
print(df.head())

# 3.  Plot the data ---------------------------------------------------------------
import matplotlib.pyplot as plt
plt.plot(w3.time, w3.ic, label='Industrial capital')
plt.plot(w3.time, w3.ppol, label='Persistent pollution')
plt.plot(w3.time, w3.sc, label='Service capital')
plt.plot(w3.time, w3.nr, label='Nonrenewable resources')
plt.legend(); plt.show()
