import numpy as np
from pyworld3 import World3

# 0.  Build & initialise the model -------------------------------------------------
w3 = World3(year_min=1900, year_max=2100, dt=1)                 # yearly grid, 201 points
w3.set_world3_control()
w3.init_world3_constants()
w3.init_world3_variables()
w3.set_world3_table_functions()
w3.set_world3_delay_functions()

# 1.  Integrate --------------------------------------------------------------------
w3.run_world3(fast=False)                                   # backward-Euler solver

# 2.  Print state variables --------------------------------------------------------
ts_len = w3.time.size                                       # length of the time grid

state_vars = [
    name                                                    # attribute name
    for name, val in vars(w3).items()                       # walk through w3.__dict__
    if isinstance(val, np.ndarray) and val.size == ts_len   # keeps only arrays matching time length
]

print("time-varying variables:", state_vars)