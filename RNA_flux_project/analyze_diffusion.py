import numpy as np
import matplotlib.pyplot as plt
import pathlib
from tqdm import tqdm
from pathlib import Path
from scipy.optimize import curve_fit
import json
from matplotlib.cm import viridis
current_dir = pathlib.Path(__file__).resolve().parent
plt.style.use(f"{current_dir}/../plotting/joseph_group.mplstyle")
import argparse

"""
From the trajectory of the guest chains, we want to measure the diffusion of the guest
chains as they move through the host condensate. 
We will do this by measuring the mean square displacement (MSD) of the guest chains over time.

Algortithm:
1. Load the trajectory of the guest chains and determine the unique mol. ids.
2. For each mol. id, determine the timesteps at which the guest chain is in the cavity.
    (Binary mask: 1 if the guest chain is in the cavity, 0 otherwise)
    Between two 1s, the guest chain is traversing through the condensate.
3. In the time time window between two 1s, measure the MSD of the guest chain.
4. Repeat for all mol. ids.
5. Average the MSD over all mol. ids and compute MSD slope.
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
    with open(file_path, 'r') as file:
        lines = file.readlines()

    timesteps = []
    box_sizes = []
    atom_positions = []
    mol_ids = []
    atom_types = []
    num_atoms = 0
    current_timestep = None
    current_box_size = None
    current_positions = []
    current_types = []
    current_mol_ids = []

    for idx, line in enumerate(tqdm(lines)):
        if 'ITEM: TIMESTEP' in line:
            if current_timestep is not None:
                timesteps.append(current_timestep)
                box_sizes.append(current_box_size)
                atom_positions.append(np.array(current_positions))
                mol_ids.append(np.array(current_mol_ids))
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
                mol_id = int(lines[i].strip().split()[1])
                current_mol_ids.append(mol_id)

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
    
    # Sanity check: Since all the chains have the same identity, the atom types should be exactly same.
    atom_types = np.reshape(atom_types, (atom_types.shape[0]//chain_length, chain_length))
    is_same = np.allclose(atom_types, atom_types[0])
    if is_same:
        print("All chains have the same atom types. Great!")
    atom_types = atom_types[0]

    # Convert all lists to arrays
    timesteps = np.array(timesteps)
    atom_positions = np.array(atom_positions)
    mol_ids = np.array(mol_ids)
    
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




# Call functions
if __name__=="__main__":
    file_path = args.path
    traj = args.traj_guest
    Nm = args.chain_length

    print(f"Extracting data from trajectory file.")
    timesteps, box_sizes, num_atoms, atom_positions, atom_types, mol_ids = read_lammps_trajectory(f"{file_path}/{traj}", Nm)

    print(f"\nTrajectory file details:")
    print(f"# frames   :{len(timesteps)}")
    print(f"# particles:{num_atoms}")
    print(f"Number of chains:{num_atoms//Nm}")

    # Determine the unique mol. ids
    unique_mol_ids = np.unique(mol_ids)
    print(mol_ids.shape)
