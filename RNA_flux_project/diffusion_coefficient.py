import numpy as np
import matplotlib.pyplot as plt
import pathlib
from tqdm import tqdm
from pathlib import Path
from scipy.optimize import curve_fit
import json
from matplotlib.cm import viridis
current_dir = pathlib.Path(__file__).resolve().parent
plt.style.use(f"{current_dir}/../plotting/ymw.mplstyle")
import argparse

"""
Here we want to measure the diffusion coefficient of the guest chains while they are
inside the host chain condensate.

Diffusion coefficient can be measured by tracking the mean squared displacement of the
chains over time and the slope of MSD vs time can give us the diffusion coefficient.

In this specific case the measurement is going to be more nuanced, since the MSD of the
guest chains will vary by their spatial location in the simulation box.
We can split the simulation box into three regions:
1. Cavity (middle of the simulation box)
2. Host condensate (Surrounding the cavity)
3. Dilute phase (Region where the guest chains should preferrably go)

Essentially, it would only be useful to measure the MSD of the chains when they are inside
the host chain condensate.

The steps involved in this measurement would be:
1. Determine the three regions using the trajectory containing both host and guest chains.
    > Cavity region is trivial.
    > To determine the condensate interface/boundary we will first need to use the density profile
    and by fitting it to half of a super-gaussian or a tanh function, we can determine the location
    of the interface. 
    > Dilute phase is everything beyond the condensate.
2. Determine the COM positions of the guest chains over time from the trajectory tracking only
   the guest chains.
3. Determine when the chains are inside the host condensate and when they are outside.
4. When the chains are inside the condensate we can measure their MSD

QUESTIONS:
1. Since we are in a slab geometry should we be dividing the slope of MSD by 6 or 2? We essentially
   have one dimensional diffusion of the chains.

"""

parser = argparse.ArgumentParser()
parser.add_argument("--path", type=str, required=True) # dir where data is stored
parser.add_argument("--traj_guest", type=str, required=True) # trajectory file tracking the guest chains only
parser.add_argument("--traj_all", type=str, required=True) # trajectory file tracking all chains
parser.add_argument("--chain_length", type=int, required=True) # Length of the guest chains
args = parser.parse_args()

def read_lammps_trajectory(file_path):
    """
    Reads trajectory file to extract positions of all chains at different timesteps.
    
    Expected atom data in the trajectory file:
    atom_id mol_id type q xu yu zu
    Total 7 elements. The positional data is at the 4,5,6 indices


    Output:
    - Simulation box
    - # atoms
    - Timesteps of snapshots
    - Atom positions at every timestep
    """
    with open(file_path, 'r') as file:
        lines = file.readlines()

    timesteps = []
    box_sizes = []
    atom_positions = []
    atom_types = []
    num_atoms = 0
    current_timestep = None
    current_box_size = None
    current_positions = []
    current_types = []

    for idx, line in enumerate(tqdm(lines)):
        if 'ITEM: TIMESTEP' in line:
            if current_timestep is not None:
                timesteps.append(current_timestep)
                box_sizes.append(current_box_size)
                atom_positions.append(np.array(current_positions))
                atom_types = np.array(current_types)
                current_positions = []
                current_types = []
            current_timestep = int(lines[idx + 1].strip())
        elif 'ITEM: NUMBER OF ATOMS' in line:
            num_atoms = int(lines[idx + 1].strip())
        elif 'ITEM: BOX BOUNDS' in line:
            bounds = lines[idx + 1:idx + 4]
            current_box_size = [list(map(float, b.split())) for b in bounds]
        elif 'ITEM: ATOMS' in line:
            start_index = idx + 1
            for i in range(start_index, start_index + num_atoms):
                position = list(map(float, lines[i].strip().split()[4:7]))
                current_positions.append(position)
                type = int(lines[i].strip().split()[2])
                current_types.append(type)

    # Append the last timestep data
    if current_timestep is not None:
        timesteps.append(current_timestep)
        box_sizes.append(current_box_size)
        atom_positions.append(np.array(current_positions))
        atom_types = np.array(current_types)

    # Sanity check: Does the simulation box size change with time? (It should not.)
    box_sizes = np.array(box_sizes)
    is_constant = np.allclose(box_sizes, box_sizes[0], rtol=1e-5, atol=1e-8)
    if is_constant:
        print("Box size remains constant throughout the simulation. Great!")
        box_sizes = box_sizes[0]
    
    # Convert all lists to arrays
    timesteps = np.array(timesteps)
    atom_positions = np.array(atom_positions)
    
    return timesteps, box_sizes, num_atoms, atom_positions, atom_types

def wrap_positions_inside_sim_box(positions, simBox):
    """
    wrap positions inside simulation box.
    NOTE: Assuming simulation box with the origin at the lower left vertex
    
    Arguments:
    - positions (3d numpy array): [#frames, #chains, 3] / [#frames, #atoms, 3]
    - simBox (2d numpy array): [[xmin,xmax] [ymin,ymax] [zmin,zmax]]
    """
    # Simulation box sizes in all directions
    Lx = simBox[0][1] - simBox[0][0] 
    Ly = simBox[1][1] - simBox[1][0]
    Lz = simBox[2][1] - simBox[2][0]
    for frame_pos in tqdm(positions):
        for r in frame_pos:
            r[0] -= np.floor(r[0]/Lx)*Lx
            r[1] -= np.floor(r[1]/Ly)*Ly
            r[2] -= np.floor(r[2]/Lz)*Lz
    return positions

def compute_number_density_profile(positions, simBox, nbins):
    """
    Compute the density profile of all atoms/chains along the long axis 
    of the simulation box.

    NOTE: Depending on the "positions" array we either compute the number 
    density of all atoms or of the chain center of masses.

    NOTE: We expect wrapped positions.

    Arguments:
    - positions (3d numpy array): [#frames, #chains, 3] / [#frames, #atoms, 3]
    - simBox (2d numpy array): [[xmin,xmax] [ymin,ymax] [zmin,zmax]]
    - nbins (int): # bins to histogram data. Higher bins leads to finer resolution,
            but can lead to larger noise.
    """
    nframes = positions.shape[0]
    density_profile = np.zeros((nframes, nbins))
    bin_width = (simBox[2][1] - simBox[2][0])/nbins
    bin_volume = (simBox[0][1] - simBox[0][0]) * (simBox[1][1] - simBox[1][0]) * bin_width
    for idx, frame_pos in enumerate(tqdm(positions)):
        hist_, bin_edges = np.histogram(frame_pos[:,2], bins=nbins,
                                   range=(simBox[2][0], simBox[2][1]))
        density_profile[idx] = hist_/bin_volume
    bin_centers = (bin_edges[:-1] + bin_edges[1:])/2.
    return density_profile, bin_centers

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
    super_gaussian = lambda x, A, x0, sigma, p: A*np.exp(-((x-x0)**2/(2.*sigma**2))**p)
    initial_guess = [np.percentile(avg_profile, 95), np.mean(coords), np.std(coords), 2]
    bounds = ([0, 0, 0.1, 1], [1, 1000., 1000., 20])
    popt, pcov = curve_fit(super_gaussian, coords, avg_profile, p0=initial_guess, bounds=bounds, maxfev=10000)
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

def get_mass_by_id(amino_dict, id_to_find):
    for amino_acid, data in amino_dict.items():
        if data['id'] == id_to_find:
            return data['mass']
    return None



# Call functions
if __name__=="__main__":
    file_path=args.path
    Nm = args.chain_length
    nbins = 100 # #bins for computing density profile

    # Read trajectories
    print(f"Extracting data from trajectory file: {args.traj_guest}.")
    timesteps_guest, box_sizes, num_atoms_guest, atom_positions_guest, atom_types_guest = read_lammps_trajectory(f"{file_path}/{args.traj_guest}")
    
    print(f"Trajectory file details : {args.traj_guest}")
    print(f"# frames                : {len(timesteps_guest)}")
    print(f"# guest atoms           : {num_atoms_guest}")
    
    print(f"Extracting data from trajectory file: {args.traj_all}.")
    timesteps_all, box_sizes, num_atoms_all, atom_positions_all, atom_types_all = read_lammps_trajectory(f"{file_path}/{args.traj_all}")

    print(f"Trajectory file details : {args.traj_all}")
    print(f"# frames                : {len(timesteps_all)}")
    print(f"# atoms (guest+all)     : {num_atoms_all}")
    
    
    # The second trajectory contains both guest and host chain data.
    # We want to filter out the guest chains from the data
    print(f"Filtering out guest chain positions.")
    atom_positions_host = []
    for positions in atom_positions_all:
        atom_positions_host.append([pos for pos, iden in zip(positions, atom_types_all) if iden < 21])
    atom_positions_host = np.array(atom_positions_host)
    
    
    # NOTE: The trajectories contain unwrapped positions. To find interface we need to wrap the positions
    #       inside simulation box
    print(f"Wrapping positions of host chains.")
    host_positions_wrapped = wrap_positions_inside_sim_box(atom_positions_host, box_sizes)
    # Compute density profile
    print(f"Computing density profile.")
    host_density_profile, bin_centers = compute_number_density_profile(host_positions_wrapped, box_sizes, nbins)
    # Average the density profile over all the frames
    host_avg_density_profile = np.mean(host_density_profile, axis=0)
    
    # Save density profile
    np.savetxt("condensate_density_profile.dat", np.column_stack((bin_centers, host_avg_density_profile)), header=f"bin_center\tnumber_density\t(nbins={nbins})")

    # Find interface using the density profile of the host chains
    # We want to prevent using data points where the cavity is.
    mask = (bin_centers < 440.) | (bin_centers > 560.)
    left_interface, right_interface, fine_coords, fitted_profile = find_interfaces(bin_centers[mask], host_avg_density_profile[mask])
    # Save data
    file = open("interface_fitting.dat", "w")
    file.write(f"# left interface = {left_interface}\n")
    file.write(f"# right interface = {right_interface}\n")
    file.write(f"# zcoord\tfitted_profile\n")
    for x,y in zip(fine_coords, fitted_profile):
        file.write(f"{x} {y}\n")
    file.close()
    
    # Plot the interface data
    fig, ax = plt.subplots()
    ax.plot(bin_centers[mask], host_avg_density_profile[mask], ls="", marker='o', mfc="None", label="host proteins")
    ax.plot(fine_coords, fitted_profile, label="Fitted profile")
    ax.set_xlabel(r"$z$")
    ax.set_ylabel(r"$n(z)$")
    plt.savefig("fitted_interface.pdf", dpi=600)

    """
    Now that we know where the condensate is (in between the two interfaces) we can start measuring the MSD when the 
    guest chains are inside the condensate.
    """
    # First we need to reshape position array to extract each chain position over time
    # Original shape : (#frames, #atoms, 3)
    # New shape : (#frames, #chains, #atoms-per-chain, 3)
    guest_atoms_positions = np.reshape(atom_positions_guest, (atom_positions_guest.shape[0], atom_positions_guest.shape[0]//Nm, Nm, 3))
    