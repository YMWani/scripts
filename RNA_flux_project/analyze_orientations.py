import numpy as np
import matplotlib.pyplot as plt
import pathlib
from tqdm import tqdm
from pathlib import Path
from scipy.optimize import curve_fit
from scipy.stats import sem
import json
from matplotlib.cm import viridis
current_dir = pathlib.Path(__file__).resolve().parent
plt.style.use(f"{current_dir}/../plotting/joseph_group.mplstyle")
plt.rcParams['text.usetex'] = True
plt.rcParams['text.latex.preamble'] = '\n'.join([r'\usepackage{sansmath}', r'\sansmath'])
import argparse


"""
From the guest chain trajectory we want to determine if segments of the guest chains
are oriented along the flux director (z/-z direction).

The steps are as follows:
1. Read trajectory and reshape position array such that we have shape (n_frames, n_chains, n_beads, 3)
2. For each chain, for each frame, compute
    a. segment vectors
    b. segment mid-point
    c. angle theta between segment vector and z-axis (use both +z and -z direction) and note the smaller angle
    d. get the bin index for the mid-point z-coordinate
3. Average the theta values in each bin over all chains and frames
4. Plot the average theta vs z
"""

parser = argparse.ArgumentParser()
parser.add_argument("--traj_guest", type=str, required=True) # trajectory file tracking the guest chains only
parser.add_argument("--chain_length", type=int, required=True) # Length of the guest chains
parser.add_argument("--cavity_bounds", type=float, nargs=2, required=True) # Cavity bounds in z-direction
parser.add_argument("--segment_size", type=int, required=True) # Number of beads per segment
args = parser.parse_args()

# Required functions
def read_lammps_trajectory(file_path, chain_length):
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
    print(f"Reading file: {file_path}")
    timesteps = []
    box_sizes = []
    atom_positions = []
    mol_ids = []
    current_types = []
    
    # Pre-allocate lists to avoid repeated resizing
    with open(file_path, 'r') as file:
        line = file.readline()
        while line:
            if 'ITEM: TIMESTEP' in line:
                timestep = int(file.readline().strip())
                # print(timestep)
                timesteps.append(timestep)
            
            elif 'ITEM: NUMBER OF ATOMS' in line:
                num_atoms = int(file.readline().strip())
            
            elif 'ITEM: BOX BOUNDS' in line:
                current_box_size = []
                for _ in range(3):  # x, y, z bounds
                    bounds = list(map(float, file.readline().split()))
                    current_box_size.append(bounds)
                box_sizes.append(current_box_size)
            
            elif 'ITEM: ATOMS' in line:
                # Pre-allocate arrays for this frame
                positions = np.zeros((num_atoms, 3), dtype=np.float32)
                types = np.zeros(num_atoms, dtype=np.int32)
                mol_id_frame = np.zeros(num_atoms, dtype=np.int32)
                
                # Read atom data in bulk
                for i in range(num_atoms):
                    atom_data = file.readline().strip().split()
                    positions[i] = [float(atom_data[4]), float(atom_data[5]), float(atom_data[6])]
                    types[i] = int(atom_data[2])
                    mol_id_frame[i] = int(atom_data[1])
                
                atom_positions.append(positions)
                # Only store types from the first frame
                if len(current_types) == 0:
                    current_types = types
                mol_ids.append(mol_id_frame)
                
            line = file.readline()
    
    # Convert lists to arrays
    timesteps = np.array(timesteps)
    box_sizes = np.array(box_sizes)
    atom_positions = np.array(atom_positions)
    mol_ids = np.array(mol_ids)
    
    # Sanity check: Does the simulation box size change with time?
    is_constant = np.allclose(box_sizes, box_sizes[0], rtol=1e-5, atol=1e-8)
    if is_constant:
        print("Box size remains constant throughout the simulation. Great!")
        box_sizes = box_sizes[0]
    
    # Sanity check: Since all the chains have the same identity, the atom types should be exactly same.
    atom_types = np.reshape(current_types, (len(current_types)//chain_length, chain_length))
    is_same = np.allclose(atom_types, atom_types[0])
    if is_same:
        print("All chains have the same atom types. Great!")
        atom_types = atom_types[0]
    
    print(f"Loaded {len(timesteps)} frames with {num_atoms} atoms")
    
    return timesteps, box_sizes, num_atoms, atom_positions, atom_types, mol_ids






# Call functions
if __name__=="__main__":
    traj = args.traj_guest
    Nm = args.chain_length
    cavity_bounds = args.cavity_bounds
    segment_size = args.segment_size

    # **************** READ TRAJECTORY ****************
    print(f"Extracting data from trajectory file.")
    timesteps, box_sizes, num_atoms, atom_positions, atom_types, mol_ids = read_lammps_trajectory(f"{traj}", Nm)

    Lz = box_sizes[2][1] - box_sizes[2][0] # Box length in z-direction
    
    print(f"Trajectory file details:")
    print(f"# frames   :{len(timesteps)}")
    print(f"# particles:{num_atoms}")
    print(f"Atom types: {atom_types}")

    Nchains = num_atoms // Nm
    print(f"# chains   :{Nchains}")

    # Reshape the atom positions to (n_frames, n_chains, n_beads, 3)
    atom_positions = np.reshape(atom_positions, (len(timesteps), Nchains, Nm, 3))
    
    # **************** ANALYZE ORIENTATIONS ****************
    # Create empty arrays to store theta values for the different bins
    nbins = 100 # Number of bins in z-direction
    bin_width = (box_sizes[2][1] - box_sizes[2][0]) / nbins
    
    bin_centers = np.linspace(box_sizes[2][0] + bin_width/2, box_sizes[2][1] - bin_width/2, nbins)
    theta_vals = np.zeros((nbins, 2)) # Store sum of cos(theta) and count of segments in each bin

    # Loop over frames and chains to compute segment orientations
    print(f"Analyzing segment orientations...")
    # Loop over frames
    for frame in tqdm(range(len(timesteps))):
        # Loop over chains
        for chain in range(Nchains):
            chain_position = atom_positions[frame, chain] # Shape (Nm, 3)
            
            # Loop over segments in the chain
            for start in range(0, Nm - segment_size): # start index of segments within a chain
                end = start + segment_size # end index of segments within a chain

                # Compute segment vector and mid-point
                segment_vector = chain_position[end] - chain_position[start]
                segment_mod = np.linalg.norm(segment_vector)
                mid_point = (chain_position[start] + chain_position[end]) / 2.0

                # Compute angle theta with respect to z-axis
                # NOTE: Here we use the absolute value of the z-component to account for both +z and -z directions
                theta = np.arccos(np.abs(segment_vector[2] / segment_mod))

                # Determine bin index for mid-point z-coordinate
                z_coord = mid_point[2] # unwrapped z-coordinate
                z_coord -= np.floor(z_coord / Lz) * Lz # wrap z-coordinate into the box

                bin_index = int(z_coord // bin_width)
                
                theta_vals[bin_index, 0] += theta
                theta_vals[bin_index, 1] += 1

    # Compute average theta values
    theta_avg = theta_vals[:, 0] / theta_vals[:, 1]
    # Convert radians to degrees
    theta_avg = np.degrees(theta_avg)

    np.savetxt(f"segment_orientation_vs_z_Nm{Nm}_seg{segment_size}.txt", np.vstack((bin_centers, theta_avg)).T, header="z_center\tavg_theta(degrees)")