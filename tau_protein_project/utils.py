import json
import random
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import sem
from scipy.optimize import curve_fit
import pathlib
import random
current_dir = pathlib.Path(__file__).resolve().parent
plt.style.use(f"{current_dir}/../plotting/ymw.mplstyle")

def compute_SCD(seq):
    """
    Compute sequence charge decoration parameter for a given sequence
    The equation can be found in this paper: https://doi.org/10.1063/1.4929391 eq.(14)
    """
    N = len(seq) # Length of protein
    
    # Load amino acid dictionary
    parent_dir = Path(__file__).parent
    with open(f'{parent_dir}/amino_acid_dict.json', 'r') as f:
        amino_acid_dict = json.load(f)
    print(parent_dir)

    SCD = 0.
    for m in range(1,N):
        qm = amino_acid_dict[seq[m]]["charge"]
        for n in range(m):
            qn = amino_acid_dict[seq[n]]["charge"]
            SCD += qm * qn * (m-n)**0.5
    SCD /= N
    
    return SCD


def estimate_num_PEG_chains(phi, v_box=4e7, Nm_PEG=182, sigma_PEG=4.644, r0_PEG=2.322):
    """
    Estimate the number of PEG chains required in the simulations given a volume fraction (phi)

    Parameters:
    - phi (float): Volume fraction
    """
    # First we need to compute the effective volume of a single PEG chain
    # NOTE: There is a overlap between consecutive monomers.
    vol_overlap = (Nm_PEG-1) * overlap_volume_two_spheres(sigma_PEG/2., r0_PEG)
    vol_monomer = (np.pi/6.) * sigma_PEG**3
    vol_PEG = Nm_PEG * vol_monomer - vol_overlap

    # Work backwords given that we know the required volume fraction
    num_chains = int(np.round(phi * v_box / vol_PEG))
    return num_chains


def overlap_volume_two_spheres(R, d):
    return (np.pi/12.) * (4*R + d) * (2*R - d)


def parse_density_profile_file(filename):
    """
    This function reads the density profile file created from LAMMPS simulations using the 
    "compute .. chunk/atom ..." command and extracts the timesteps and density profiles at
    the different time points
    
    Parameters:
    - filename (str): Name of the file containing the density profile

    Output:
    - timesteps (numpy array): Timesteps at which the density profile was measured and saved
    - densities (numpy array): Density profiles at every timestep
    """    
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


def compute_averaged_density_profile(density_prof):
    """
    This function takes the raw density profile extracted from the LAMMPS density profile file
    and averages it over all the frames. In addition, the density profile is also recentered 
    such that COM of condensate lies at 0.5 (reduced units)

    Parameters:
    - density_prof (numpy array): raw density profile obtained from LAMMPS simulations.
                                  Expected shape (nframes, nbins, 2)
                                  The final 2 columns are just the bin_centers and bin_density

    Output:
    - bins: bin centers
    - bin_density: bin densities
    """
    avg_density_prof = np.mean(density_prof, axis=0) # Assuming no noise in the data, so we don't discard any part of it
    sem_density_prof = sem(density_prof[:,:,1], axis=0)
    # Shift the density profile such that it is centered.
    # Find the center of mass of the density profile
    bins = avg_density_prof[:,0]
    bin_density = avg_density_prof[:,1]
    cm = np.sum(bins * bin_density) / np.sum(bin_density)
    # Adjust the bin positions such that the COM is at the center of the box (i.e. 0.5)
    bins -= (cm - 0.51)
    bins[bins <= 0] += 1.0
    return bins, bin_density, sem_density_prof


def find_interfaces(coords, avg_profile, derivative_threshold=1e-5, percentile_low=15, percentile_high=85, min_dilute_size_multiplier=0.3):
    """
    This function takes averaged density profile for a direct coexistence simulations
    and fits a super gaussian function to the profile. From the second derivative of the 
    super gaussian one can extract the interface coordinates of the slab.

    Parameters:
    - coords (array): bin_centers of the density profile
    - avg_profile (array): bin_densities of the density profile

    Output:
    - left_interface_coord
    - right_interface_coord
    - fine_coords: finely spaced bin centers
    - fitted_profile: Fitted profile on the finely spaced bin centers
    """
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





if __name__ == "__main__":
    # Protein
    tsteps, densities = parse_density_profile_file("/Users/yw9071_admin/Documents/tau-protein-project/Tau_slabs/Tau_2.5_PEG/densities_chunked2_protein.dat")
    bins, bin_density, bin_density_sem = compute_averaged_density_profile(densities)
    left_interf, right_interf, fine_coords, fitted_profile = find_interfaces(bins, bin_density)
    # PEG
    tsteps_, densities_ = parse_density_profile_file("/Users/yw9071_admin/Documents/tau-protein-project/Tau_slabs/Tau_2.5_PEG/densities_chunked2_PEG.dat")
    bins_, bin_density_, bin_density_sem_ = compute_averaged_density_profile(densities_)
    
    fig, ax = plt.subplots()
    # protein
    ax.fill_between(bins, bin_density-bin_density_sem, bin_density+bin_density_sem)
    # ax.plot(bins, bin_density, marker='o', linestyle='', markerfacecolor='none')
    # ax.plot(fine_coords, fitted_profile)
    # PEG
    # ax.plot(bins_, bin_density_, marker='s', linestyle='', markerfacecolor='none')
    
    ax.set_xlabel(r"$z/L_\mathrm{z}$")
    ax.set_ylabel(r"$\rho \, (\mathrm{g/cm^3})$")
    plt.tight_layout()
    plt.show()
