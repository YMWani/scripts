import numpy as np
import matplotlib.pyplot as plt
import pathlib
import freud
from tqdm import tqdm
from pathlib import Path
from scipy.optimize import curve_fit
from scipy.stats import sem
import json
from matplotlib.cm import viridis
current_dir = pathlib.Path(__file__).resolve().parent
plt.style.use(f"{current_dir}/../plotting/joseph_group.mplstyle")
import argparse

"""
Using the trajectory of the guest chains, we can compute their mean squared displacement (MSD)
along the different cooridinate axes. 

Here we will use the COM trajectory of the guest chains.
"""

parser = argparse.ArgumentParser()
parser.add_argument("--path", type=str, required=True) # dir where data is stored
parser.add_argument("--traj_guest", type=str, required=True) # trajectory file tracking the guest chains only
parser.add_argument("--chain_length", type=int, required=True) # Length of the guest chains
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



# ********************************************************
# Call functions
if __name__=="__main__":
    file_path = args.path
    traj = args.traj_guest
    Nm = args.chain_length

    print(f"Extracting data from trajectory file.")
    timesteps, box_sizes, num_atoms, atom_positions, atom_types, mol_ids = read_lammps_trajectory(f"{file_path}/{traj}", Nm)

    numFrames = len(timesteps)
    delta_t = timesteps[1] - timesteps[0]
    
    print(f"\nTrajectory file details:")
    print(f"# frames   :{numFrames}")
    print(f"# particles:{num_atoms}")
    print(f"Number of chains:{num_atoms//Nm}")

    # Determine the COM trajectory of the guest chains
    print(f"\nComputing COM positions of the guest chains")
    # Reshape position array to separate each chain
    atom_positions_reshaped = np.reshape(atom_positions,
                                         (atom_positions.shape[0],
                                          atom_positions.shape[1]//Nm,
                                          Nm,
                                          atom_positions.shape[2])) # [#frames, #chains, #atoms-per-chain, 3]

    # Get the mass of chain monomers using particle types
    chain_masses = extract_particle_masses(atom_types)

    com_positions = compute_COM_positions(atom_positions_reshaped, chain_masses) # unwrapped coordinates [#frames, #chains, 3]
    
    # Create empty lists/arrays to store data
    msd_values = np.zeros((4, numFrames)) # [(msd_total, msd_x, msd_y, msd_z), #frames] 
    msd_timesteps = np.arange(0, numFrames*delta_t, delta_t) # [#frames]

    # Compute MSD for each chain and average over all chains
    msd_ = freud.msd.MSD()
    msd_.compute(positions=com_positions)
    msd_values[0] = msd_.msd

    # Compute MSD along each axis
    sub_traj_x = np.zeros_like(com_positions)
    sub_traj_x[:,:,0] = com_positions[:,:,0]
    msd_x = freud.msd.MSD()
    msd_x.compute(positions=sub_traj_x)
    msd_values[1] = msd_x.msd

    sub_traj_y = np.zeros_like(com_positions)
    sub_traj_y[:,:,1] = com_positions[:,:,1]
    msd_y = freud.msd.MSD()
    msd_y.compute(positions=sub_traj_y)
    msd_values[2] = msd_y.msd

    sub_traj_z = np.zeros_like(com_positions)
    sub_traj_z[:,:,2] = com_positions[:,:,2]
    msd_z = freud.msd.MSD()
    msd_z.compute(positions=sub_traj_z)
    msd_values[3] = msd_z.msd

    # Save the MSD data to a file for later use
    np.savetxt("msd_isotropic_sims.dat", np.column_stack((msd_timesteps, msd_values.T)),
               header="Time\tMSD_total\tMSD_x\tMSD_y\tMSD_z")