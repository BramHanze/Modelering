# Differential equations

This public project is a tools to fit growth models to given data. When running it creates a plot with every model and shows them in order of lowest AIC.

Automatically applies the following growth models:
* Gompertz Model
* Mendelsohn Model
* Von Bertalanffy Growth Model
* Linear Growth Model
* Exponential Growth Model
* Allee Effect Model

## Customization
Parameter ranges can be adjusted in every model.

## Scalable
New models can be added. Define the function and the ranges for the parameters in 'differential.py' and call the 'fitter()' function for fitting. Call the new function, and add it to the 'models' dict in 'runner.ipynb' (second code chunk).

## Prerequisites/Dependencies
* Python 3.11.2 and up, not tested on earlier versions
* numpy
* scipy
* matplotlib
* pandas

## Installation and use
#### Clone the repository:
```bash
git clone https://github.com/BramHanze/Modellering.git
```
#### Add your data
Replace the data in data.txt with your desired data.

#### Run the application: 
Press 'Run All' in the included 'runner.ipynb'

## Support
If you encounter issues or bugs in this tool feel free to reach out via e-mail (see Authors).

## Authors
    Bram Koobs - b.l.koobs@st.hanze.nl
    Demi van 't Oever - d.van.t.oever@st.hanze.nl
