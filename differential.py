import matplotlib.pyplot as plt
from scipy.integrate import odeint
import numpy as np

def gompertz(ts, Vs):
    # Gompertz differential equation
    def gompertz_equation(V, t, c, Vmax):
        return c * V * np.log(Vmax / V)

    # Solve the Gompertz equation
    def solve_gompertz(ts, V0, c, Vmax):
        return odeint(gompertz_equation, V0, ts, args=(c, Vmax)).flatten()

    # Loss function: Mean squared error
    def mse(observed, predicted):
        return np.mean((observed - predicted) ** 2)

    # Initial volume and parameter ranges
    V0 = Vs[0]
    c_range = np.linspace(0.01, 0.5, 50)  # Range for c
    Vmax_range = np.linspace(max(Vs) - 1, max(Vs) + 5, 50)  # Range for Vmax

    # Grid search for optimal parameters
    best_mse = float('inf')
    best_c, best_Vmax = None, None

    for c in c_range:
        for Vmax in Vmax_range:
            Vs_pred = solve_gompertz(ts, V0, c, Vmax)
            error = mse(Vs, Vs_pred)
            if error < best_mse:
                best_mse = error
                best_c, best_Vmax = c, Vmax

    #print(f"c (growth rate constant): {best_c:.4f}")
    #print(f"Vmax (carrying capacity): {best_Vmax:.2f}")

    # Generate the fitted curve using the optimal parameters
    ts_fine = np.linspace(min(ts), max(ts), 500)
    Vs_fitted = solve_gompertz(ts_fine, V0, best_c, best_Vmax)

    # Plot observed data and fitted curve
    plt.figure(figsize=(10, 6))
    plt.plot(ts, Vs, 'ro', label='Observed Data')
    plt.plot(ts_fine, Vs_fitted, '-b', label='Fitted Gompertz Curve')
    #plt.title('Tumor Volume Growth (Gompertz Model)')
    plt.xlabel('$t$ (days)')
    plt.ylabel('$V(t)$ (mm³)')
    plt.legend()
    plt.grid(True)
    return best_mse, plt

def mendelsohn(ts, Vs):
    # Mendelsohn differential equation
    def mendelsohn_diff_eq(V, t, c, d):
        return c * V**d

    # Solve the Mendelsohn equation
    def solve_mendelsohn(ts, V0, c, d):
        return odeint(mendelsohn_diff_eq, V0, ts, args=(c, d)).flatten()

    # Loss function: Mean squared error
    def mse(observed, predicted):
        return np.mean((observed - predicted) ** 2)

    # Initial volume and parameter ranges
    V0 = Vs[0]
    c_range = np.linspace(0.01, 0.5, 50)  # Range for c
    d_range = np.linspace(0.5, 3.0, 50)  # Range for d

    # Grid search for optimal parameters
    best_mse = float('inf')
    best_c, best_d = None, None

    for c in c_range:
        for d in d_range:
            Vs_pred = solve_mendelsohn(ts, V0, c, d)
            error = mse(Vs, Vs_pred)
            if error < best_mse:
                best_mse = error
                best_c, best_d = c, d

    #print(f"c (growth rate constant): {best_c:.4f}")
    #print(f"d (growth power): {best_d:.4f}")

    # Generate the fitted curve using the optimal parameters
    ts_fine = np.linspace(min(ts), max(ts), 500)
    Vs_fitted = solve_mendelsohn(ts_fine, V0, best_c, best_d)

    # Plot observed data and fitted curve
    plt.figure(figsize=(10, 6))
    plt.plot(ts, Vs, 'ro', label='Observed Data')
    plt.plot(ts_fine, Vs_fitted, '-b', label='Fitted Mendelsohn Curve')
    #plt.title('Tumor Volume Growth (Mendelsohn Model)')
    plt.xlabel('$t$ (days)')
    plt.ylabel('$V(t)$ (mm³)')
    plt.legend()
    plt.grid(True)
    return best_mse, plt

def Von_Bertalanffy(ts, Vs):
    # Von Bertalanffy Model: V(t) = c * V^(2/3) - d * V
    def von_bertalanffy(t, c, d):
        return c * (t ** (2/3)) - d * t

    # Error function (Mean Squared Error)
    def mse(c, d, ts, Vs):
        predicted = von_bertalanffy(np.array(ts), c, d)
        return np.mean((predicted - np.array(Vs))**2)

    # Initial guesses for c and d
    c_init = 0.1
    d_init = 0.1

    # Optimization: Gradient Descent to minimize MSE
    learning_rate = 0.0001
    iterations = 10000
    c, d = c_init, d_init

    for i in range(iterations):
        grad_c = 0
        grad_d = 0
        for t, V in zip(ts, Vs):
            pred = von_bertalanffy(np.array([t]), c, d)
            grad_c += 2 * (pred - V) * (t ** (2/3))  # Derivative w.r.t c
            grad_d += 2 * (pred - V) * (-t)  # Derivative w.r.t d
        c -= learning_rate * grad_c / len(ts)
        d -= learning_rate * grad_d / len(ts)

    #print(c = {c})
    #print(d = {d})

    # Plot the data and the fit
    plt.figure(figsize=(10, 6))
    plt.plot(ts, Vs, 'ro', label='Observed Data')
    t_fit = np.linspace(min(ts), max(ts), 100)
    V_fit = von_bertalanffy(t_fit, c, d)
    plt.plot(t_fit, V_fit, label='Von Bertalanffy Fit', color='blue')
    plt.xlabel('Time (t)')
    plt.ylabel('Volume (V)')
    #plt.title('Von Bertalanffy Growth Model')
    plt.legend()
    return mse(c, d, ts, Vs), plt