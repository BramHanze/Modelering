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
    return best_mse, ts_fine, Vs_fitted

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

    return best_mse, ts_fine, Vs_fitted


def von_bertalanffy(ts, Vs):
    # Von Bertalanffy growth differential equation
    def von_bertalanffy_equation(V, t, c, d):
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
    # Linear growth equation
    def linear_equation(V, t, c):
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
    # Exponential growth model: dV/dt = c * V
    def exponential_equation(V, t, c):
        return c * V

    # Solve the exponential growth equation
    def solve_exponential(ts, V0, c):
        return odeint(exponential_equation, V0, ts, args=(c,)).flatten()

    # Loss function: Mean squared error
    def mse(observed, predicted):
        return np.mean((observed - predicted) ** 2)

    # Volume and parameter range
    V0 = Vs[0]
    c_range = np.linspace(0.01, 1.0, 100)  # Range for c

    # Grid search for best parameter
    best_mse = float('inf')
    best_c = None

    for c in c_range:
        Vs_pred = solve_exponential(ts, V0, c)
        error = mse(Vs, Vs_pred)
        if error < best_mse:
            best_mse = error
            best_c = c

    # Print the found parameter
    #print(f"c (growth rate constant): {best_c:.4f}")

    ts_fine = np.linspace(min(ts), max(ts), 500)
    Vs_fitted = solve_exponential(ts_fine, V0, best_c)
    return best_mse, ts_fine, Vs_fitted

def allee_effect(ts, Vs):
    # Allee effect differential equation
    def allee_equation(V, t, c, Vmin, Vmax):
        return c * (V - Vmin) * (Vmax - V)

    # Solve the Allee effect equation
    def solve_allee(ts, V0, c, Vmin, Vmax):
        return odeint(allee_equation, V0, ts, args=(c, Vmin, Vmax)).flatten()

    # Loss function: Mean squared error
    def mse(observed, predicted):
        return np.mean((observed - predicted) ** 2)

    # Initial volume and parameter ranges
    V0 = Vs[0]
    c_range = np.linspace(0.01, 0.5, 10)  # Range for c
    Vmin_range = np.linspace(min(Vs) - 5, min(Vs) + 1, 20)  # Range for Vmin
    Vmax_range = np.linspace(max(Vs) - 1, max(Vs) + 5, 20)  # Range for Vmax

    # Grid search for optimal parameters
    best_mse = float('inf')
    best_c, best_Vmin, best_Vmax = None, None, None

    for c in c_range:
        for Vmin in Vmin_range:
            for Vmax in Vmax_range:
                Vs_pred = solve_allee(ts, V0, c, Vmin, Vmax)
                error = mse(Vs, Vs_pred)
                if error < best_mse:
                    best_mse = error
                    best_c, best_Vmin, best_Vmax = c, Vmin, Vmax

    # Generate the fitted curve using the optimal parameters
    ts_fine = np.linspace(min(ts), max(ts), 500)
    Vs_fitted = solve_allee(ts_fine, V0, best_c, best_Vmin, best_Vmax)
    return best_mse, ts_fine, Vs_fitted

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




def unused():
    plt.figure(figsize=(10, 6))
    plt.plot(ts, Vs, 'ro', label='Observed Data')
    plt.plot(ts_fine, Vs_fitted, '-b', label='Fitted Exponential Growth Curve')
    plt.xlabel('$t$ (days)')
    plt.ylabel('$V(t)$ (mm³)')
    plt.legend()
    plt.grid(True)


    # Plot the data and fit
    plt.figure(figsize=(10, 6))
    plt.plot(ts, Vs, 'ro', label='Observed Data')
    plt.plot(ts_fine, Vs_fitted, '-b', label='Fitted Linear Growth Curve')
    plt.xlabel('$t$ (days)')
    plt.ylabel('$V(t)$ (mm³)')
    plt.legend()
    plt.grid(True)


    # Plot the data and the fit
    plt.figure(figsize=(10, 6))
    plt.plot(ts, Vs, 'ro', label='Observed Data')

    plt.plot(t_fit, V_fit, label='Von Bertalanffy Fit', color='blue')
    plt.xlabel('Time (t)')
    plt.ylabel('Volume (V)')
    #plt.title('Von Bertalanffy Growth Model')
    plt.legend()



    # Plot observed data and fitted curve
    plt.figure(figsize=(10, 6))
    plt.plot(ts, Vs, 'ro', label='Observed Data')
    plt.plot(ts_fine, Vs_fitted, '-b', label='Fitted Mendelsohn Curve')
    #plt.title('Tumor Volume Growth (Mendelsohn Model)')
    plt.xlabel('$t$ (days)')
    plt.ylabel('$V(t)$ (mm³)')
    plt.legend()
    plt.grid(True)


    # Plot observed data and fitted curve
    plt.figure(figsize=(10, 6))
    plt.plot(ts, Vs, 'ro', label='Observed Data')
    plt.plot(ts_fine, Vs_fitted, '-b', label='Fitted Gompertz Curve')
    #plt.title('Tumor Volume Growth (Gompertz Model)')
    plt.xlabel('$t$ (days)')
    plt.ylabel('$V(t)$ (mm³)')
    plt.legend()
    plt.grid(True)
