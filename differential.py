import matplotlib.pyplot as plt
from scipy.integrate import odeint
import numpy as np


def fitter(ts, Vs, solve_function, param_ranges):
    """
    Fits the given models/functions to the data.
    ts, Vs: user give data
    solve_function: the formula for the growth model
    param_ranges: the ranges in which the parameters are to be contained
    """
    def mse(observed, predicted): #calculate mse
        return np.mean((observed - predicted) ** 2)

    V0 = Vs[0]  #determine starting position (first point in data)
    param_names = [f"param_{i}" for i in range(len(param_ranges))]
    best_mse = float('inf')
    best_params = None

    # Grid search
    for param_values in np.array(np.meshgrid(*param_ranges)).T.reshape(-1, len(param_ranges)):
        Vs_pred = solve_function(ts, V0, *param_values)
        error = mse(Vs, Vs_pred)
        if error < best_mse:
            best_mse = error
            best_params = param_values

    # Generate the fitted curve using the optimal parameters
    ts_fine = np.linspace(min(ts), max(ts), 500)
    Vs_fitted = solve_function(ts_fine, V0, *best_params)

    return best_mse, best_params, ts_fine, Vs_fitted

def gompertz(ts, Vs):
    def gompertz_equation(V, t, c, Vmax): #Define formula
        return c * V * np.log(Vmax / V)

    def solve_gompertz(ts, V0, c, Vmax):
        return odeint(gompertz_equation, V0, ts, args=(c, Vmax)).flatten()

    c_range = np.linspace(0.01, 0.5, 50)  # Range for c
    Vmax_range = np.linspace(max(Vs) - 1, max(Vs) + 5, 50)  # Range for Vmax
    param_ranges = [c_range, Vmax_range]

    return fitter(ts, Vs, solve_gompertz, param_ranges)


def mendelsohn(ts, Vs):
    def mendelsohn_diff_eq(V, t, c, d): #Define formula
        return c * V**d

    def solve_mendelsohn(ts, V0, c, d):
        return odeint(mendelsohn_diff_eq, V0, ts, args=(c, d)).flatten()

    c_range = np.linspace(0.01, 0.5, 50)  # Range for c
    d_range = np.linspace(0.5, 3.0, 50)  # Range for d
    param_ranges = [c_range, d_range]

    return fitter(ts, Vs, solve_mendelsohn, param_ranges)


def von_bertalanffy(ts, Vs):
    def von_bertalanffy_equation(V, t, c, d): #Define formula
        return c * V**(3/4) - d * V

    # Solve the Von Bertalanffy equation
    def solve_von_bertalanffy(ts, V0, c, d):
        return odeint(von_bertalanffy_equation, V0, ts, args=(c, d)).flatten()

    # Loss function: Mean squared error
    def mse(observed, predicted):
        return np.mean((observed - predicted) ** 2)

    # Initial volume and parameter ranges
    V0 = Vs[0]
    c_range = np.linspace(0.1, 5.0, 100)  # Expanded range for c
    d_range = np.linspace(0.01, 2.0, 100)  # Expanded range for d

    # Grid search for optimal parameters
    best_mse = float('inf')
    best_c, best_d = None, None

    for c in c_range:
        for d in d_range:
            Vs_pred = solve_von_bertalanffy(ts, V0, c, d)

            # Ensure predictions are valid and avoid numerical issues
            if np.any(Vs_pred <= 0):
                continue

            error = mse(Vs, Vs_pred)
            if error < best_mse:
                best_mse = error
                best_c, best_d = c, d

    if best_c is None or best_d is None:
        raise ValueError("Failed to find suitable parameters. Check data or parameter ranges.")

    # Generate the fitted curve using the optimal parameters
    ts_fine = np.linspace(min(ts), max(ts), 500)
    Vs_fitted = solve_von_bertalanffy(ts_fine, V0, best_c, best_d)
    return best_mse, ts_fine, Vs_fitted

def linear_growth(ts, Vs):
    def linear_equation(V, t, c): #Define formula
        return c

    # Solve the linear growth equation
    def solve_linear(ts, V0, c):
        return odeint(linear_equation, V0, ts, args=(c,)).flatten()

    # Loss function: Mean squared error
    def mse(observed, predicted):
        return np.mean((observed - predicted) ** 2)

    # Volume and parameter range
    V0 = Vs[0]
    c_range = np.linspace(0.01, 5.0, 100)  # Range for c

    # Grid search for best parameter
    best_mse = float('inf')
    best_c = None

    for c in c_range:
        Vs_pred = solve_linear(ts, V0, c)
        error = mse(Vs, Vs_pred)
        if error < best_mse:
            best_mse = error
            best_c = c

    # Print the found parameter
    #print(f"c (growth rate constant): {best_c:.4f}")

    ts_fine = np.linspace(min(ts), max(ts), 500)
    Vs_fitted = solve_linear(ts_fine, V0, best_c)
    return best_mse, ts_fine, Vs_fitted

def exponential_growth(ts, Vs):
    def exponential_equation(V, t, c): #Define formula
        return c * V

    def solve_exponential(ts, V0, c):
        return odeint(exponential_equation, V0, ts, args=(c,)).flatten()

    c_range = np.linspace(0.01, 1.0, 100)  # Range for c
    param_ranges = [c_range]

    return fitter(ts, Vs, solve_exponential, param_ranges)

def allee_effect(ts, Vs):
    def allee_equation(V, t, c, Vmin, Vmax): #Define formula
        return c * (V - Vmin) * (Vmax - V)

    def solve_allee(ts, V0, c, Vmin, Vmax):
        return odeint(allee_equation, V0, ts, args=(c, Vmin, Vmax)).flatten()

    c_range = np.linspace(0.01, 0.5, 10)  # Range for c
    Vmin_range = np.linspace(min(Vs) - 5, min(Vs) + 1, 20)  # Range for Vmin
    Vmax_range = np.linspace(max(Vs) - 1, max(Vs) + 5, 20)  # Range for Vmax
    param_ranges = [c_range, Vmin_range, Vmax_range]

    return fitter(ts, Vs, solve_allee, param_ranges)

def combined_growth(ts, Vs):
    # Deze functie combineert de formules van Exponentieel afvlakkende groei + logistische groei
    def combined_growth_equation(V, t, c1, c2, Vmax): #Define formula
        return c1 * (Vmax - V) + c2 * V * (Vmax - V)

    def solve_combined_growth(ts, V0, c1, c2, Vmax):
        return odeint(combined_growth_equation, V0, ts, args=(c1, c2, Vmax)).flatten()

    c1_range = np.linspace(0.01, 0.5, 50)
    c2_range = np.linspace(0.01, 0.5, 50)
    Vmax_range = np.linspace(max(Vs) - 1, max(Vs) + 5, 50)
    param_ranges = [c1_range, c2_range, Vmax_range]

    return fitter(ts, Vs, solve_combined_growth, param_ranges)

def plot(ts,Vs,x,y,label,title):
    plt.figure(figsize=(10, 6))
    plt.plot(ts, Vs, 'ro', label='Observed Data')
    plt.plot(x, y, '-b', label=label)
    plt.xlabel('$t$ (days)')
    plt.ylabel('$V(t)$ (mm³)')
    plt.title(title)
    plt.legend()
    plt.grid(True)
    plt.show()
