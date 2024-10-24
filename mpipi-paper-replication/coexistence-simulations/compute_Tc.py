import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
import matplotlib.ticker as ticker
import json
import argparse


plt.style.use("/Users/cbe-jaj/Documents/yashraj/mpipi-paper-replication/ymw.mplstyle")

"""
Here we want to compute the critical temperature of a given protein(IDR)
using the measurements obtained from the slab simulations.

Known quantities:
1. Temperatures at which simulations were performed (T)
2. dense phase density (rho_h)
3. dilute phase density (rho_l)


Law of coexistence densities:
[rho_h - rho_l]^3.06 = d(1-T/T_c)               (1)
Here d and Tc are the fitting parameters

Law of rectilinear diameters:
rho_h + rho_l = 2*rho_c + 2A*(T-Tc)             (2)
Here rho_c and A are fitting parameters
"""

# Input data
parser = argparse.ArgumentParser()
# PATH to the directory that contains slab simulation data for a given temperature
parser.add_argument("-p", "--path", type=str, required=True)
parser.add_argument('--temperatures', nargs='+', type=int, required=True)
args = parser.parse_args()


temp_vals = []; rho_l_vals = []; rho_h_vals = []; delta_rho_vals = []
for temp in args.temperatures:
    with open(f'{args.path}/{temp}K/restart-sim/analyzed_data.json', 'r') as file:
        data = json.load(file)
        temp_vals.append(data['Temperature'])
        rho_l_vals.append(data['rho_low'])
        rho_h_vals.append(data['rho_high'])

# Look at the input data
# fig, ax = plt.subplots()
# ax.scatter(rho_l_vals, temp_vals)
# ax.scatter(rho_h_vals, temp_vals)
# ax.set_xlabel("density")
# ax.set_ylabel("Temperature")
# plt.tight_layout()
# plt.show()

# Estimate critical temperature by fitting to the first equation

# Define the function that we want to fit
def function_1(X, Tc, d):
    rho_l, rho_h = X
    delta_rho = rho_h - rho_l
    T = Tc * (1 - delta_rho**3.06 / d)
    return T


# Fit the curve
parameters, covariance = curve_fit(function_1, xdata=[rho_l_vals, rho_h_vals], ydata=temp_vals, p0=np.array([350., 1.]))
Tc_fitted = parameters[0]
d_fitted = parameters[1]
print(f"Critical temperature = {Tc_fitted}")


# Compute the critical density

# Define the function that we want to fit
def function_2(X, rho_c, A):
    global Tc_fitted
    rho_l, rho_h = X
    T = Tc_fitted - (rho_h + rho_l - 2*rho_c)/(2*A)
    return T

# Fit the curve
parameters, covariance = curve_fit(function_2, xdata=[rho_l_vals, rho_h_vals], ydata=temp_vals, p0=np.array([0.25, 0.001]),
                                   bounds=([0.0, 0], [0.3, np.inf]))
rho_c_fitted = parameters[0]

fig, ax = plt.subplots()

ax.scatter(rho_l_vals, temp_vals, color="blue", label="FUS-PLD", s=15)
ax.scatter(rho_h_vals, temp_vals, color="blue", s=15)
ax.set_xlabel(r"$\rho \, (\mathrm{g/cm^3})$")
ax.set_ylabel(r"$T \, (\mathrm{K})$")

xvals_1 = np.linspace(0., rho_c_fitted, 100)
xvals_2 = np.linspace(0.7, rho_c_fitted, 100)
yvals = function_1([xvals_1, xvals_2], Tc_fitted, d_fitted)

ax.plot(xvals_1, yvals, color="blue")
ax.plot(xvals_2, yvals, color="blue")

ax.scatter(rho_c_fitted, Tc_fitted, edgecolor='blue', facecolor='none', s=15, label=rf"$T_\mathrm{{c}} = {Tc_fitted:.0f} \, \mathrm{{K}}$")

ax.plot([-0.01, 0.72], [332,332], color='red', linestyle="dashed")
# ax.plot([-0.01, 0.72], [Tc_fitted,Tc_fitted], color='green', label=rf"$T_\mathrm{{c}} = {Tc_fitted:.0f} \, \mathrm{{K}}$")

ax.xaxis.set_major_locator(ticker.MultipleLocator(0.2))
ax.xaxis.set_minor_locator(ticker.MultipleLocator(0.1))
ax.yaxis.set_major_locator(ticker.MultipleLocator(50))
ax.yaxis.set_minor_locator(ticker.MultipleLocator(25))

ax.set_ylim(250, 350)
ax.set_xlim(-0.01, 0.72)

ax.legend()
plt.tight_layout()
# plt.show()
plt.savefig("phase-diagram.pdf", dpi=600)