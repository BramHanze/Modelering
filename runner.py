import differential as diff
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np

ts = [
    3.46,  4.58,  5.67,  6.64,  7.63,  8.41,  9.32, 10.27, 11.19,
    12.39, 13.42, 15.19, 16.24, 17.23, 18.18, 19.29, 21.23, 21.99,
    24.33, 25.58, 26.43, 27.44, 28.43, 30.49, 31.34, 32.34, 33.00,
    35.20, 36.34, 37.29, 38.50, 39.67, 41.37, 42.58, 45.39, 46.38,
    48.29, 49.24, 50.19, 51.14, 52.10, 54.00, 56.33, 57.33, 59.38,
]
Vs = [
    0.0158, 0.0264, 0.0326, 0.0445, 0.0646, 0.0933, 0.1454, 0.2183, 0.2842,
    0.4977, 0.6033, 0.8441, 1.2163, 1.4470, 2.3298, 2.5342, 3.0064, 3.4044,
    3.2046, 4.5241, 4.3459, 5.1374, 5.5376, 4.8946, 5.0660, 6.1494, 6.8548,
    5.9668, 6.6945, 6.6395, 6.8971, 7.2966, 7.2268, 6.8815, 8.0993, 7.2112,
    7.0694, 7.4971, 6.9974, 6.7219, 7.0523, 7.1095, 7.0694, 8.0562, 7.2268, 
]
n = len(ts)

gompertz_mse, gompertz_fig = diff.gompertz(ts, Vs)
mendelsohn_mse, mendelsohn_fig = diff.mendelsohn(ts, Vs)
vb_mse, vb_fig = diff.Von_Bertalanffy(ts, Vs)

models = [
    {'name': 'Gompertz', 'fig': gompertz_fig, 'mse': gompertz_mse},
    {'name': 'Mendelsohn', 'fig': mendelsohn_fig, 'mse': mendelsohn_mse},
    {'name': 'Von Bertalanffy', 'fig': vb_fig, 'mse': vb_mse}
]

#Display figures in the order of best  AIC
for model in models:
    rss = model['mse'] * n
    ln_likelihood = -n / 2 * np.log(rss / n)
    model['aic'] = 2 * len(models) - 2 * ln_likelihood
for model in sorted(models, key=lambda x: x['aic']):
    print(f'Model: {model['name']}, AIC: {model['aic']}')
    #print(model['fig'])
    #model.title(f'Tumor Volume Growth {model["name"]} model (AIC={model["aic"]:.2f})')
    model['fig'].show()

