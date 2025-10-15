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
From the trajectory of all the particles in the system, we want to analyze 
whether the cavity wall leads to host proteins being constantly adsorbed to it.

We will do this by computing the center of mass of each host protein, and then
computing the distance of each center of mass to the cavity wall over time.

The steps are as follows:
1. Load the LAMMPS trajectory file
2. Filter out the cavity wall particles and the guest particles
3. Compute the center of mass of each host protein over time
4. Wrap the center of mass positions into the simulation box
5. Compute the distance of each center of mass to the cavity wall over time
6. Make a scatter plot of the distances to the cavity wall over time where the
   x-axis is the chain index and the y-axis is the distance to the cavity wall.
   The color of the scatter points will be chosen from a colormap to represent time.
"""

parser = argparse.ArgumentParser()
parser.add_argument("--traj_all", type=str, required=True) # trajectory file tracking all particles
parser.add_argument("--chain_length", type=int, required=True) # Length of the guest chains
parser.add_argument("--cavity_bounds", type=float, nargs=2, required=True) # Cavity bounds in z-direction
args = parser.parse_args()

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
        
    atom_types = current_types
    
    print(f"Loaded {len(timesteps)} frames with {num_atoms} atoms")
    
    return timesteps, box_sizes, num_atoms, atom_positions, atom_types, mol_ids

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
    for frame_pos in positions:
        for r in frame_pos:
            r[0] -= np.floor(r[0]/Lx)*Lx
            r[1] -= np.floor(r[1]/Ly)*Ly
            r[2] -= np.floor(r[2]/Lz)*Lz
    return positions

def compute_COM_positions(positions, mass):
    """
    Compute the COM positions of chains at every timestep

    Arguments:
    - positions (4d numpy array): Atom positions for the chains at every timestep
                                  Expected shape (#frames, #chains, chain_length, 3)
    - mass (1d numpy array): Atom masses for one chain (Every chain is the same)
                             Expected shape (chain_length)
                
    Output:
    - com_positions: Center of mass positions of all the chains at every timestep
                     Shape (#frames, #chains, 3)
    """
    com_positions = np.zeros((positions.shape[0], positions.shape[1], 3))

    print("Extracting COM positions")
    for frame_idx in tqdm(range(positions.shape[0])):
        for chain_idx in range(positions.shape[1]):
            chain_pos = positions[frame_idx, chain_idx]
            com_positions[frame_idx, chain_idx, 0] = np.sum(chain_pos[:,0]*mass)/np.sum(mass)
            com_positions[frame_idx, chain_idx, 1] = np.sum(chain_pos[:,1]*mass)/np.sum(mass)
            com_positions[frame_idx, chain_idx, 2] = np.sum(chain_pos[:,2]*mass)/np.sum(mass)
    
    return com_positions

def extract_particle_masses(types):
    """
    Extract the masses of particles given an array containing particle types
    """
    parent_dir = Path(__file__).parent
    with open(f'{parent_dir}/amino_acid_dict.json', 'r') as f:
        amino_acid_dict = json.load(f)
    
    masses = []
    for type in types:
        type = type%20
        mass = get_mass_by_id(amino_acid_dict, type)
        masses.append(mass)

    return masses

def get_mass_by_id(amino_dict, id_to_find):
    for amino_acid, data in amino_dict.items():
        if data['id'] == id_to_find:
            return data['mass']
    return None


if __name__=="__main__":
    traj = args.traj_all
    Nm = args.chain_length
    cavity_zmin, cavity_zmax = args.cavity_bounds

    print(f"Extracting data from trajectory file.")
    timesteps, box_sizes, num_atoms, atom_positions, atom_types, mol_ids = read_lammps_trajectory(f"{traj}", Nm)
    
    # NOTE: By design the trajectory file contains frames just before and after the MC move.
    # Therefore the two frames are identical except for the chains that were involved in the MC move. 
    # We can remove the frames which are basically at the same timestep.
    # Let's remove the frame at every timestep=0
    zero_timestep_indices = np.where(timesteps==0)[0]
    atom_positions = np.delete(atom_positions, zero_timestep_indices[1:], axis=0)

    # Get the actual timesteps
    actual_timesteps = []
    delta_t = timesteps[1] - timesteps[0]
    for i in range(len(timesteps)):
        actual_timesteps.append(i * delta_t)
    actual_timesteps = np.array(actual_timesteps)
    
    # Remove the cavity and guest particles
    unique_mol_ids, counts = np.unique(mol_ids[0], return_counts=True)
    cavity_mol_id = 1
    guest_mol_ids = unique_mol_ids[counts==Nm]
    host_mol_ids = unique_mol_ids[counts==150] # Host proteins have 150 residues
    
    # Get indices where mol_ids[0] has elements from host_mol_ids array
    mask = np.isin(mol_ids[0], host_mol_ids)
    host_indices = np.where(mask)[0]

    # Select only the host particles
    host_positions = atom_positions[:, host_indices, :]
    host_atom_types = atom_types[host_indices]

    # Reshape positions to group by chains
    host_positions = np.reshape(host_positions, (host_positions.shape[0], len(host_mol_ids), 150, 3))
    host_atom_types = np.reshape(host_atom_types, (len(host_mol_ids), 150))[0]
    # print(np.sum((host_positions[0,0,1:,:] - host_positions[0,0,:-1,:])**2, axis=1)**0.5) # Check that the bonds are of around 3.8

    # Compute the center of mass of the host proteins
    host_masses = extract_particle_masses(host_atom_types)
    com_positions = compute_COM_positions(host_positions, host_masses)
    
    # Wrap the COM positions inside the simulation box
    com_positions_wrapped = wrap_positions_inside_sim_box(com_positions, box_sizes[0])
    

    # Compute the distance of each COM to the cavity wall over time
    # Cavity wall is at z=cavity_zmin and z=cavity_zmax
    distances_to_cavity_wall = np.minimum(np.abs(com_positions_wrapped[:,:,2]-cavity_zmin), np.abs(com_positions_wrapped[:,:,2]-cavity_zmax))
    
    # Get the maximum distance to the cavity wall for normalization
    max_distance = np.max(distances_to_cavity_wall, axis=0)
    # Get the minimum distance to the cavity wall for normalization
    min_distance = np.min(distances_to_cavity_wall, axis=0)

    # Make a scatter plot of the distances to the cavity wall
    # x-axis: minimum distance to cavity wall
    # y-axis: maximum distance to cavity wall

    fig, ax = plt.subplots(figsize=(3,3))
    ax.plot(min_distance, max_distance, marker='o', ls='', markersize=3, alpha=0.7, markeredgecolor="None")
    
    ax.set_ylim(0, 250)
    ax.set_xlim(0, 250)
    
    ax.set_xlabel(r"Minimum distance to cavity wall ($\AA$)")
    ax.set_ylabel(r"Maximum distance to cavity wall ($\AA$)")

    fig.savefig("max_min_distance_to_cavity_wall_scatter.pdf", bbox_inches='tight')