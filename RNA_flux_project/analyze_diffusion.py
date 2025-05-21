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

def determine_slice(cavity, delta_z, atom_positions):
    """
    Determine the slice of the condensate region that a given chain is in at every timestep.
    """
    # Empty array to store the slice index for each timestep
    # 1: Close to the cavity, 2: Middle of the condensate, 3: Close to the dilute phase
    # 0: Outside the condensate
    slice_indices = np.zeros(atom_positions.shape[0], dtype=int)

    # Region 1: Close to the cavity
    mask = (atom_positions[:,2] <= cavity[0]) & (atom_positions[:,2] >= cavity[0]-delta_z)
    slice_indices[mask] = 1
    mask = (atom_positions[:,2] >= cavity[1]) & (atom_positions[:,2] <= cavity[1]+delta_z)
    slice_indices[mask] = 1

    # Region 2: Middle of the condensate
    mask = (atom_positions[:,2] <= cavity[0]-delta_z) & (atom_positions[:,2] >= cavity[0]-2*delta_z)
    slice_indices[mask] = 2
    mask = (atom_positions[:,2] >= cavity[1]+delta_z) & (atom_positions[:,2] <= cavity[1]+2*delta_z)
    slice_indices[mask] = 2

    # Region 3: Close to the dilute phase
    mask = (atom_positions[:,2] <= cavity[0]-2*delta_z) & (atom_positions[:,2] >= cavity[0]-3*delta_z)
    slice_indices[mask] = 3
    mask = (atom_positions[:,2] >= cavity[1]+2*delta_z) & (atom_positions[:,2] <= cavity[1]+3*delta_z)
    slice_indices[mask] = 3

    return slice_indices

def find_consecutive_ranges(arr):
    """
    Find consecutive integer ranges in an array.
    
    Arguments:
    - arr: 1D array/list of integers (can be unsorted)
    
    Returns:
    - ranges: List of tuples (start, end) for each consecutive range found
    
    Example:
    Input: [1, 2, 3, 5, 6, 7, 10, 15, 16]
    Output: [(1, 3), (5, 7), (10, 10), (15, 16)]
    """
    ranges = []
    
    # Initialize with first element
    range_start = arr[0]
    range_end = arr[0]
    
    for i in range(1, len(arr)):
        # If current element continues the sequence
        if arr[i] == range_end + 1:
            range_end = arr[i]
        # If gap found, store the previous range and start a new one
        else:
            ranges.append((range_start, range_end))
            range_start = arr[i]
            range_end = arr[i]
    
    # Add the last range
    ranges.append((range_start, range_end))
    
    return ranges

# Call functions
if __name__=="__main__":
    file_path = args.path
    traj = args.traj_guest
    Nm = args.chain_length

    print(f"Extracting data from trajectory file.")
    timesteps, box_sizes, num_atoms, atom_positions, atom_types, mol_ids = read_lammps_trajectory(f"{file_path}/{traj}", Nm)
    
    # NOTE: By design the trajectory file contains frames just before and after the MC move.
    # Therefore the two frames are identical except for the chains that were involved in the MC move. 
    # We can remove the frames which are basically at the same timestep.
    # Let's remove the frame at every timestep=0
    mask_tstep = np.where(timesteps == 0)[0]
    timesteps = np.delete(timesteps, mask_tstep)
    atom_positions = np.delete(atom_positions, mask_tstep, axis=0)
    mol_ids = np.delete(mol_ids, mask_tstep, axis=0)

    # Get the actual timesteps
    actual_timesteps = []
    delta_t = timesteps[1] - timesteps[0]
    for i in range(len(timesteps)):
        actual_timesteps.append(i * delta_t)
    actual_timesteps = np.array(actual_timesteps)
    
    
    numFrames = len(timesteps)

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
    
    com_positions = compute_COM_positions(atom_positions_reshaped, chain_masses) # unwrapped coordinates
    com_positions_wrapped = wrap_positions_inside_sim_box(com_positions, box_sizes) # wrapped coordinates
    
    print("Determining the avg. number of guest chains in the condensate during the simulation.")
    num_guest_in_condensate = []
    for i in tqdm(range(numFrames)):
        mask1 = ((com_positions[i,:,2] >= args.condensate_bounds[0]) & (com_positions[i,:,2] <= args.cavity_bounds[0]))
        mask2 = ((com_positions[i,:,2] >= args.cavity_bounds[1]) & (com_positions[i,:,2] <= args.condensate_bounds[1]))
        mask = np.logical_or(mask1, mask2)
        num_guest_in_condensate.append(np.sum(mask))
    num_guest_in_condensate = np.array(num_guest_in_condensate)
    avg_num_guest_in_condensate = np.mean(num_guest_in_condensate)
    err_num_guest_in_condensate = sem(num_guest_in_condensate)
    print(f"Avg. number of guest chains in the condensate: {avg_num_guest_in_condensate:.3f} ± {err_num_guest_in_condensate:.3f}")
    # Save the number of guest chains in the condensate to a file
    output_data = {
        "avg_num_guest_in_condensate": avg_num_guest_in_condensate,
        "err_num_guest_in_condensate": err_num_guest_in_condensate,
        "num_guest_in_condensate": num_guest_in_condensate.tolist()
    }
    output_file = f"{file_path}/num_guest_in_condensate.json"
    with open(output_file, "w") as f:
        json.dump(output_data, f, indent=4)


    # Create empty lists/arrays to store data
    flux_times = []
    msd_values = np.zeros((3, 5000)) # [#regions, #frames] NOTE: 5000 frames refer to 500 ns in the simulation (delta_t = 0.1 ns -> 5000*0.1 = 500 ns); 
    # NOTE: We assume the worst case scenario that a chain remains in the same region throughout the simulation
    msd_counter = np.zeros((3, 5000)) # [#regions, #frames]
    msd_timesteps = np.arange(0, 5000*delta_t, delta_t) # [#frames]


    # NOTE: Every index in com_positions on axis 1 corresponds to a unique chain
    print("\nAnalyzing the guest chains")
    for i in tqdm(range(com_positions.shape[1])):
        # print(f"\nAnalyzing chain {i+1}")
        chain_traj = com_positions[:,i,:] # [#frames, 3] (unwrapped coordinates)
        chain_traj_wrapped = com_positions_wrapped[:,i,:] # (wrapped coordinates)
        
        """ 1. Average time for a chain to completely flux through the condensate"""
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
            flux_times.append(actual_timesteps[end_idx] - actual_timesteps[start_idx])
        

        """ 2. Spatially dependent MSD"""
        # Divide the "host condensate" region in three parts.
        delta_z = np.abs((args.condensate_bounds[1] - args.cavity_bounds[1]) / 3) # Condensate on both sides of the cavity is approximately equal
        
        # Determine the slice indices for the chain COM trajectory
        slice_indices = determine_slice(args.cavity_bounds, delta_z, chain_traj_wrapped)
        
        # Find the consecutive ranges of slice indices when chain is in region 1: close to the cavity
        consecutive_ranges = find_consecutive_ranges(np.where(slice_indices == 1)[0])
        for start_idx, end_idx in consecutive_ranges:
            # Check if the range is long enough (> 5ns)
            time_ = (end_idx - start_idx)*delta_t/1e5 # in ns
            if time_ > 5: # if the chain stays in the region for more than 5ns
                # extract the trajectory of the chain for the given range
                chain_sub_traj = chain_traj[start_idx:end_idx].reshape((-1,1,3)) # unwrapped coordinates
                # compute the MSD for the given range
                msd_ = freud.msd.MSD()
                msd_.compute(positions=chain_sub_traj)
                msd_values[0, 0:msd_.msd.shape[0]] += msd_.msd
                msd_counter[0, 0:msd_.msd.shape[0]] += 1
        
        # Find the consecutive ranges of slice indices when chain is in region 2: middle of the condensate
        consecutive_ranges = find_consecutive_ranges(np.where(slice_indices == 2)[0])
        for start_idx, end_idx in consecutive_ranges:
            # Check if the range is long enough (> 5ns)
            time_ = (end_idx - start_idx)*delta_t/1e5 # in ns
            if time_ > 5:
                # extract the trajectory of the chain for the given range
                chain_sub_traj = chain_traj[start_idx:end_idx].reshape((-1,1,3)) # unwrapped coordinates
                # compute the MSD for the given range
                msd_ = freud.msd.MSD()
                msd_.compute(positions=chain_sub_traj)
                msd_values[1, 0:msd_.msd.shape[0]] += msd_.msd
                msd_counter[1, 0:msd_.msd.shape[0]] += 1
        
        # Find the consecutive ranges of slice indices when chain is in region 3: close to the dilute phase
        consecutive_ranges = find_consecutive_ranges(np.where(slice_indices == 3)[0])
        for start_idx, end_idx in consecutive_ranges:
            # Check if the range is long enough (> 5ns)
            time_ = (end_idx - start_idx)*delta_t/1e5 # in ns
            if time_ > 5:
                # extract the trajectory of the chain for the given range
                chain_sub_traj = chain_traj[start_idx:end_idx].reshape((-1,1,3)) # unwrapped coordinates
                # compute the MSD for the given range
                msd_ = freud.msd.MSD()
                msd_.compute(positions=chain_sub_traj)
                msd_values[2, 0:msd_.msd.shape[0]] += msd_.msd
                msd_counter[2, 0:msd_.msd.shape[0]] += 1
    
    # Save flux data
    flux_times = np.array(flux_times)
    avg_flux_time = np.mean(flux_times)
    err_flux_time = sem(flux_times)
    print(f"Average time for a complete flux for a chain: {avg_flux_time/1e5:.3f} ± {err_flux_time/1e5:.3f} ns")
    # Save the flux times to a file
    output_data = {
        "avg_flux_time (ns)": avg_flux_time/1e5,
        "err_flux_time (ns)": err_flux_time/1e5,
        "flux_times": flux_times.tolist()
    }
    output_file = f"{file_path}/flux_times.json"
    with open(output_file, "w") as f:
        json.dump(output_data, f, indent=4)

    
    # Process the MSD data
    msd_normalized = np.divide(msd_values, msd_counter, out=np.zeros_like(msd_values), where=msd_counter!=0) # Avoid division by zero
    np.savetxt("msd.dat", np.column_stack((msd_timesteps, msd_normalized[0], msd_normalized[1], msd_normalized[2])), header="Time(tsteps)\tMSD_region1\tMSD_region2\tMSD_region3", fmt="%1.5e")