import numpy as np
import matplotlib.pyplot as plt
import argparse
from scipy.optimize import curve_fit
import json
plt.style.use("/home/yashraj/Documents/scripts/plotting/ymw.mplstyle")

parser = argparse.ArgumentParser()
# PATH to the directory that contains slab simulation data for a given temperature
parser.add_argument("-p", "--path", type=str, required=True)
parser.add_argument("-t", "--temperature", type=float, required=True)
args = parser.parse_args()

"""
Step 1: 
(a) From the "densities_chunked2.dat" files, extract the averaged density profile.
(b) Recenter the density profile such that the COM is at the center of the box
"""
def parse_density_file(filename):    
    with open(filename, 'r') as data:
        densities = []
        chunk_data = []
        timesteps = []
        for line in data:
            line = line.strip() # Remove leading/trailing whitespace
            parts = line.split() # Split the line to get the different components

            if line.startswith('#'):
                # Skip comment lines
                continue
            
            elif (len(parts) == 3):
                # Line with 3 components: (Timestep, Number-of-bins, Total-count)
                timesteps.append(int(parts[0]))
                if (bool(chunk_data)):
                    densities.append(chunk_data)
                    chunk_data = []
                
            else:
                # Line with 4 components: (bin, Coord1, Ncount, density/mass)
                chunk_data.append([float(parts[1]), float(parts[3])])
        densities.append(chunk_data)
    timesteps = np.array(timesteps)
    densities = np.array(densities)
    return timesteps, densities


# Extract data from the density file
tsteps, density_profile = parse_density_file(f"{args.path}/densities_chunked2.dat")

# Compute the average density profile
nframes = density_profile.shape[0]
avg_density_profile = np.mean(density_profile[int(0.2*nframes):,:,:], axis=0) # discard the first 20% of data to prevent noise

# Shift the density profile such that it is centered.
# Find the center of mass of the density profile
bins = avg_density_profile[:,0]
bin_density = avg_density_profile[:,1]
cm = np.sum(bins * bin_density) / np.sum(bin_density)
# Adjust the bin positions such that the COM is at the center of the box (i.e. 0.5)
bins -= cm - 0.5
bins[bins < 0] += 1.0

"""
Step 2: 
(a) Find the interfaces from the density profile.
"""
## super gaussian second derivative method
def super_gaussian_sd_method(coords, avg_profile, derivative_threshold=1e-5, percentile_low=15, percentile_high=85, min_dilute_size_multiplier=0.3):
    # fit super gaussian
    super_gaussian = lambda x, A, x0, sigma, p: A*np.exp(-2*((x-x0)/sigma)**(2*np.round(p)))
    initial_guess = [np.percentile(avg_profile, 80), np.mean(coords), np.std(coords), 2]
    popt, pcov = curve_fit(super_gaussian, coords, avg_profile, p0=initial_guess, maxfev=10000)
    A, x0, sigma, p = popt
    P = 2*np.round(p)
    fine_coords = np.linspace(min(coords), max(coords), num=1000)
    fitted_profile = super_gaussian(fine_coords, *popt)
    # compute second derivative
    first_derivative = np.gradient(fitted_profile)
    second_derivative = np.gradient(first_derivative)
    # binarize normalized second derivatives with derivative threshold and get interfaces
    threshold = derivative_threshold # units: (density per length per length) / length
    binarized = [abs(x)/max(fitted_profile) <= threshold for x in second_derivative]
    num_clusters = 1
    ## keep track of "True" cluster boundaries for plotting
    true_cluster_boundaries = []
    curr_cluster = [0] if binarized[0] else []
    for idx in range(1, len(binarized)):
        if binarized[idx] != binarized[idx-1]:
            num_clusters += 1
            if binarized[idx]:
                curr_cluster.append(fine_coords[idx])
            else:
                curr_cluster.append(fine_coords[idx-1])
                true_cluster_boundaries.append(curr_cluster)
                curr_cluster = []
    if len(curr_cluster) == 1: # ends on True cluster that doesn't get closed in loop, which will be common
        curr_cluster.append(fine_coords[-1])
        true_cluster_boundaries.append(curr_cluster)
    num_clusters += 1
    num_clusters /= 2
    if binarized[0]==True and binarized[-1]==True and num_clusters == 5: # clean condensate; zero second derivative on ends, slope up/down and the middle
        left_interface_coord = fine_coords[round(([idx for idx in range(1, len(binarized)) if binarized[idx] == False and binarized[idx-1] == True][1] + [idx for idx in range(1, len(binarized)) if binarized[idx] == True and binarized[idx-1] == False][0])/2)]
        right_interface_coord = fine_coords[round(([idx for idx in list(reversed(range(0, len(binarized)-2))) if binarized[idx] == False and binarized[idx+1] == True][1] + [idx for idx in list(reversed(range(0, len(binarized)-2))) if binarized[idx] == True and binarized[idx+1] == False][0])/2)]
        if (left_interface_coord + (max(fine_coords)-right_interface_coord)) <= min_dilute_size_multiplier * right_interface_coord - left_interface_coord: # ensure dilute phase isn't too small
            left_interface_coord, right_interface_coord = np.percentile(coords, percentile_low), np.percentile(coords, percentile_high)
    else:
        left_interface_coord, right_interface_coord = np.percentile(coords, percentile_low), np.percentile(coords, percentile_high)
    
    return left_interface_coord, right_interface_coord, fine_coords, fitted_profile

left_intrfc, right_intrfc, fine_coords, fitted_profile = super_gaussian_sd_method(bins, bin_density)


# Visualization
fig, ax = plt.subplots()
# Recentered density profile
ax.plot(bins, bin_density, marker='o', linestyle='')
# Fitted curve
ax.plot(fine_coords, fitted_profile)
# Interfaces
yvals = np.linspace(min(bin_density), max(bin_density), 3)
ax.plot(np.repeat(left_intrfc, 3), yvals, color="black", linestyle="dotted")
ax.plot(np.repeat(right_intrfc, 3), yvals, color="black", linestyle="dotted")
# Set labels
ax.set_xlabel(r"$z/L_\mathrm{z}$")
ax.set_ylabel(r"$\rho \, (\mathrm{g/cm^3})$")
# save figure
plt.savefig(f"{args.path}/temporary_plot.pdf", dpi=600)

"""
Step 3: Since the interfaces are now known, we can compute the 
dense and dilute phase densities.
"""
# Create a mask for bins between the left and the right interface
mask_high = (bins >= left_intrfc) & (bins <= right_intrfc)
rho_high = np.mean(bin_density[mask_high])
rho_low = np.mean(bin_density[~mask_high])

# Save data
final_data = {
    "Temperature": args.temperature,
    "rho_high": rho_high,
    "rho_low": rho_low
}

with open(f"{args.path}/analyzed_data.json", "w") as file:
    json.dump(final_data, file)
