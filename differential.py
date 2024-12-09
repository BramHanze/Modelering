import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import curve_fit

def plot(ts, Vs):
    def differential():
        def gompertz_model(t, K, r, t0):
            return K * np.exp(-np.exp(-r * (t - t0))) #e^x

        initial_guess = [max(Vs), 0.1, ts[np.argmax(np.gradient(Vs))]]
        params, covariance = curve_fit(gompertz_model, ts, Vs, p0=initial_guess)

        K_fit, r_fit, t0_fit = params
        print(f"K: {K_fit:.2f}\nr: {r_fit:.4f}\nt0: {t0_fit:.2f}")

        ts_fine = np.linspace(min(ts), max(ts), 500) #create between-values in given range
        Vs_fitted = gompertz_model(ts_fine, K_fit, r_fit, t0_fit)

        plt.figure(figsize=(10, 6))
        plt.plot(ts, Vs, 'ro', label='Observed Data')
        plt.plot(ts_fine, Vs_fitted, '-b', label='Fitted Gompertz Curve')
        plt.title('Tumor Volume Growth (Gompertz Model)')
        plt.xlabel('$t$ (days)')
        plt.ylabel('$V(t)$ (mm³)')
        plt.legend()
        plt.grid(True)
        plt.show()
    differential()
    
