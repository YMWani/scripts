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

Algorithm:
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
parser.add_argument("--cavity_bounds", type=float, nargs=2, required=True) # Cavity bounds in z-direction
parser.add_argument("--condensate_bounds", type=float, nargs=2, required=True) # Condensate bounds in z-direction
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

def count_transitions(chain_location):
    """
    Count the number of events when a chain starts from the cavity (0),
    travels through the condensate (1) and exits to the dilute phase (2).
    
    Arguments:
    - chain_location: Array with values 0 (cavity), 1 (condensate), or 2 (dilute phase)
    
    Returns:
    - count: Number of complete transitions (0->1->2)
    - transition_indices: List of tuples containing (start_idx, end_idx) for each transition
    """
    count = 0
    transition_indices = []
    
    # State tracking variables
    in_transition = False
    transition_start = -1
    
    for i in range(len(chain_location)):
        current = chain_location[i]
        
        # Start of potential transition (found a 0)
        if not in_transition and current == 0:
            in_transition = True
            transition_start = i
            
        # In the middle of transition (should be 1)
        elif in_transition and current == 0:
            # Reset if we see another 0
            transition_start = i
            
        # End of transition (found a 2)
        elif in_transition and current == 2:
            # Verify we've passed through 1s
            if np.all(chain_location[transition_start+1:i] == 1):
                count += 1
                transition_indices.append((transition_start, i))
            
            # Reset tracking
            in_transition = False
            transition_start = -1
            
    return count, transition_indices
      

# Call functions
if __name__=="__main__":
    file_path = args.path
    traj = args.traj_guest
    Nm = args.chain_length

    print(f"Extracting data from trajectory file.")
    timesteps, box_sizes, num_atoms, atom_positions, atom_types, mol_ids = read_lammps_trajectory(f"{file_path}/{traj}", Nm)
    numFrames = len(timesteps)

    print(f"\nTrajectory file details:")
    print(f"# frames   :{numFrames}")
    print(f"# particles:{num_atoms}")
    print(f"Number of chains:{num_atoms//Nm}")

    print(timesteps)
    exit()

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
    
    com_positions = compute_COM_positions(atom_positions_reshaped, chain_masses) # unwrapped coordinates
    com_positions_wrapped = wrap_positions_inside_sim_box(com_positions, box_sizes) # wrapped coordinates

    # NOTE: Every index in com_positions on axis 1 corresponds to a unique chain

    for i in range(com_positions.shape[1]):
        print(f"\nAnalyzing chain {i+1}")
        chain_traj = com_positions[:,i,:] # [#frames, 3] (unwrapped coordinates)
        chain_traj_wrapped = com_positions_wrapped[:,i,:] # (wrapped coordinates)

        # Array to store information about chain location in the simulation box
        # 0: Cavity
        # 1: Condensate
        # 2: Dilute phase
        chain_location = np.zeros(numFrames, dtype=int)
        
        mask = ((chain_traj_wrapped[:,2] >= args.condensate_bounds[0]) & (chain_traj_wrapped[:,2] <= args.cavity_bounds[0]))
        chain_location[mask] = 1 # Condensate
        mask = ((chain_traj_wrapped[:,2] >= args.cavity_bounds[1]) & (chain_traj_wrapped[:,2] <= args.condensate_bounds[1])) 
        chain_location[mask] = 1 # Condensate
        
        mask = (chain_traj_wrapped[:,2] <= args.condensate_bounds[0]) | (chain_traj_wrapped[:,2] >= args.condensate_bounds[1])
        chain_location[mask] = 2 # Dilute phase
        
        # Determine the number of events when a chain starts from the cavity, travels through the condensate and exits to the dilute phase
        count, transition_indices = count_transitions(chain_location)
        
        for start_idx, end_idx in transition_indices:
            print(chain_location[start_idx:end_idx+1])
        

        # # Determine the timesteps at which the guest chain is in the cavity
        # mask = (chain_traj[:,2] >= 950.0) & (chain_traj[:,2] <= 1050.0) # TRUE: Inside cavity; FALSE: Outside cavity
        # mask_int = mask.astype(int)

        # print(f"Chain {i+1} is in the cavity for {np.sum(mask)/numFrames * 100:.3f}% of the time")

        # diff = mask_int[:-1] - mask_int[1:]
        # indices = np.where(diff == -1)[0]
        # print(indices[1:] - indices[:-1])

        exit()
