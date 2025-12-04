import numpy as np
import pandas as pd
from scipy.integrate import solve_ivp

def generate_lorenz_data(
    duration=400.0,
    dt=0.01,
    initial_state=[0.0, 1.0, 1.05],
    sigma=10.0,
    rho=28.0,
    beta=8.0/3.0,
    transient_duration=20.0,
    filepath="lorenz_data.csv"
):
    """
    Generates time-series data for the Lorenz system and saves it to a CSV file.
    """
    print(f"Generating Lorenz system data for {duration}s with dt={dt}s...")
    def lorenz_system(t, xyz):
        x, y, z = xyz
        dxdt = sigma * (y - x); dydt = x * (rho - z) - y; dzdt = x * y - beta * z
        return [dxdt, dydt, dzdt]
    t_span = [0, duration]; t_eval = np.arange(t_span[0], t_span[1], dt)
    sol = solve_ivp(fun=lorenz_system, t_span=t_span, y0=initial_state, t_eval=t_eval,
                    dense_output=True, rtol=1e-8, atol=1e-8)
    data = sol.y.T
    transient_steps = int(transient_duration / dt)
    attractor_data = data[transient_steps:]
    df = pd.DataFrame(attractor_data, columns=['x', 'y', 'z'])
    print(f"Generated data shape after removing transient: {df.shape}")
    if filepath:
        print(f"Saving data to '{filepath}'..."); df.to_csv(filepath, index=False); print("Save complete.")
    return df
