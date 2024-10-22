import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
import matplotlib.ticker as ticker

plt.style.use("/home/yashraj/Documents/scripts/plotting/ymw.mplstyle")

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
temp_vals = np.array([280., 300., 310., 320., 330.]) # xvals
rho_l_vals = np.array([0., 0., 0., 0.01, 0.])
rho_h_vals = np.array([0.63, 0.58, 0.5, 0.45, 0.41])
delta_rho_vals = rho_h_vals - rho_l_vals # yvals

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

ax.scatter(rho_l_vals, temp_vals)
ax.scatter(rho_h_vals, temp_vals)
ax.set_xlabel(r"$\rho \, (\mathrm{g/cm^3})$")
ax.set_ylabel(r"$T \, (\mathrm{K})$")

xvals_1 = np.linspace(0., rho_c_fitted, 100)
xvals_2 = np.linspace(0.7, rho_c_fitted, 100)
yvals = function_1([xvals_1, xvals_2], Tc_fitted, d_fitted)

ax.plot(xvals_1, yvals, ls="dashed")
ax.plot(xvals_2, yvals, ls="dashed")

ax.scatter(rho_c_fitted, Tc_fitted)

ax.xaxis.set_major_locator(ticker.MultipleLocator(0.2))
ax.xaxis.set_minor_locator(ticker.MultipleLocator(0.1))
ax.yaxis.set_major_locator(ticker.MultipleLocator(50))
ax.yaxis.set_minor_locator(ticker.MultipleLocator(25))

ax.set_ylim(250, 350)

plt.tight_layout()
plt.show()
