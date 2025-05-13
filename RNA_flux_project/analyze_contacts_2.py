import numpy as np
from tqdm import tqdm
import seaborn as sns
from pathlib import Path
import pathlib
import argparse
import matplotlib.pyplot as plt
current_dir = pathlib.Path(__file__).resolve().parent
plt.style.use(f"{current_dir}/../plotting/joseph_group.mplstyle")
plt.rcParams['text.usetex'] = True
plt.rcParams['text.latex.preamble'] = '\n'.join([r'\usepackage{sansmath}', r'\sansmath'])
from multiprocessing import Pool, cpu_count

"""
From the trajectory of the guest and host chains, over a long flux simulation,
we want to measure the interactions between the guest and host chains.

We will use the contact map to measure the interactions.

"""

# Useful functions
def read_lammps_trajectory(file_path):
    """
    Efficiently reads LAMMPS trajectory file to extract positions of all chains.

    Input:
    - file_path: Path to the LAMMPS trajectory file
    Output:
    - timesteps: (num_timesteps,)
        Array of timesteps from the trajectory
    - box_sizes: (3, 2)
        Array of box sizes for each timestep
    - num_atoms: int
        Number of atoms in the trajectory
    - atom_positions: (num_timesteps, num_atoms, 3)
        Array containing atom positions at every timestep
    - atom_types: (num_atoms,)
        Array containing atom types of all atoms in system
    - mol_ids: (num_atoms,)
        Array containing molecule IDs of all atoms in system
    - atom_ids: (num_atoms,)
    Array containing atom IDs of all atoms in system
    """
    # First pass to get number of atoms and timesteps
    num_timesteps = 0
    num_atoms = 0
    
    with open(file_path, 'r') as file:
        for line in file:
            if 'ITEM: TIMESTEP' in line:
                num_timesteps += 1
            elif 'ITEM: NUMBER OF ATOMS' in line and num_atoms == 0:
                num_atoms = int(next(file).strip())
    
    # Pre-allocate arrays
    timesteps = np.zeros(num_timesteps, dtype=int)
    box_sizes = np.zeros((num_timesteps, 3, 2), dtype=float)
    atom_positions = np.zeros((num_timesteps, num_atoms, 3), dtype=float)
    atom_types = np.zeros((num_atoms,), dtype=int)
    mol_ids = np.zeros((num_atoms,), dtype=int)
    atom_ids = np.zeros((num_atoms,), dtype=int)

    # Second pass to fill arrays
    with open(file_path, 'r') as file:
        line_iter = iter(file)
        ts_idx = -1
        
        for line in tqdm(line_iter, desc="Reading trajectory"):
            if 'ITEM: TIMESTEP' in line:
                ts_idx += 1
                timesteps[ts_idx] = int(next(line_iter).strip())
            elif 'ITEM: BOX BOUNDS' in line:
                for i in range(3):
                    box_sizes[ts_idx, i] = list(map(float, next(line_iter).split()))
            elif 'ITEM: ATOMS' in line:
                for i in range(num_atoms):
                    atom_data = next(line_iter).strip().split()
                    atom_positions[ts_idx, i, 0] = float(atom_data[4])
                    atom_positions[ts_idx, i, 1] = float(atom_data[5])
                    atom_positions[ts_idx, i, 2] = float(atom_data[6])
                    # Only store atom types and mol ids once if they don't change
                    if ts_idx == 0:
                        atom_types[i] = int(atom_data[2])
                        mol_ids[i] = int(atom_data[1])
                        atom_ids[i] = int(atom_data[0])
                        
    # Check if box size is constant
    is_constant = np.allclose(box_sizes, box_sizes[0], rtol=1e-5, atol=1e-8)
    if is_constant:
        print("Box size remains constant throughout the simulation. Great!")
        box_sizes = box_sizes[0]
    
    return timesteps, box_sizes, num_atoms, atom_positions, atom_types, mol_ids, atom_ids

def read_config(file_name):
    '''
    Reads a LAMMPS configuration file and extract the number of atoms, number of bonds,
    and bond information.

    Input:
    - file_name: Path to the LAMMPS configuration file.
    Returns:
    - num_atoms: The number of atoms in the configuration file.
    - num_bonds: The number of bonds in the configuration file.
    - bonds: A numpy array of tuples containing the bond information.
        Each tuple has the form (bond ID, bond type, first atom ID, second atom ID).
    '''
    num_atoms = 0
    num_bonds = 0
    bonds = []
    in_bond_section = False

    with open(file_name, 'r') as file:
        for line in file:
            line = line.strip()

            # Read number of atoms
            if 'atoms' in line and num_atoms == 0:
                num_atoms = int(line.split()[0])

            # Read number of bonds
            elif 'bonds' in line and num_bonds == 0:
                num_bonds = int(line.split()[0])

            # Read bond information
            elif 'Bonds' in line:
                in_bond_section = True
                continue  # Skip the "Bonds" header

            # Start reading bond data if in the bond section
            if in_bond_section:
                if len(line.split()) != 4:
                    continue  # End of bond section
                bond_data = [int(x) for x in line.split()]
                bond_id = bond_data[0]
                bond_type = bond_data[1]
                first_atom_id = bond_data[2]
                second_atom_id = bond_data[3]
                bonds.append([bond_id, bond_type, first_atom_id, second_atom_id])

    bonds = np.array(bonds)
    
    return num_atoms, num_bonds, bonds

def sequence_linear_chain(atom_ids, bond_data):
    """
    Determine the correct sequence of atoms in a linear chain.
    
    Parameters:
    - atom_ids: List of atom IDs belonging to the chain
    - bond_data: List of [atom1, atom2] pairs representing bonds
    
    Returns:
    - ordered_atoms: List of atom IDs in the correct sequential order
    """
    # Create a dictionary to store connections
    connections = {}
    for atom1, atom2 in bond_data:
        if atom1 not in connections:
            connections[atom1] = []
        if atom2 not in connections:
            connections[atom2] = []
        connections[atom1].append(atom2)
        connections[atom2].append(atom1)
    
    # Find terminal atoms (atoms with only one connection)
    terminal_atoms = [atom for atom in atom_ids if len(connections.get(atom, [])) == 1]
    
    # If we don't have exactly 2 terminal atoms, something is wrong
    if len(terminal_atoms) != 2:
        raise ValueError("Expected a linear chain with 2 terminal atoms, found " + 
                         str(len(terminal_atoms)))
    
    # Start from the first terminal atom
    current = terminal_atoms[0]
    ordered_atoms = [int(current)]
    visited = {current}
    
    # Follow the chain until we reach the other end
    while len(ordered_atoms) < len(atom_ids):
        # Get the next atom in the chain
        for neighbor in connections[current]:
            if neighbor not in visited:
                current = neighbor
                ordered_atoms.append(int(current))
                visited.add(current)
                break
    
    ordered_atoms = np.array(ordered_atoms)
    
    return ordered_atoms

def fast_pbc_distance_matrix_nonvectorized(chain1_pos, chain2_pos, box_size):
    """
    Efficiently calculate all pairwise squared distances with periodic boundary conditions.
    
    Parameters:
    -----------
    chain1_pos : numpy.ndarray
        Array of shape (Nm_chain1, 3) containing 3D coordinates of a given chain
    chain2_pos : numpy.ndarray
        Array of shape (Nm_chain2, 3) containing 3D coordinates of a given chain
    box_size : numpy.ndarray
        Size of the simulation box in each dimension
        
    Returns:
    --------
    dist_matrix : numpy.ndarray
        Matrix of shape (Nm_chain1, Nm_chain2) containing minimum image squared distances
    """
    Nm_chain1 = chain1_pos.shape[0]
    Nm_chain2 = chain2_pos.shape[0]
    
    dist_matrix = np.zeros((Nm_chain1, Nm_chain2))
    
    # Calculate all pairwise displacement vectors
    for i in range(Nm_chain1):
        # Vectorized calculation for all particles j
        delta = chain2_pos - chain1_pos[i]
        
        # Apply minimum image convention
        # NOTE: simulation box assumed to be in the positive octant of the coordinate system
        delta = delta - box_size * np.round(delta / box_size)
        
        # Calculate distances
        dist = np.sum(delta**2, axis=1)
        
        # Store in matrix
        dist_matrix[i, :] = dist
    
    return dist_matrix

def fast_pbc_distance_matrix_vectorized(chain1_pos, chain2_pos, box_size):
    """
    Fully vectorized calculation of all pairwise squared distances with periodic boundary conditions.

    Parameters:
    -----------
    chain1_pos : numpy.ndarray
        Array of shape (Nm_chain1, 3) containing 3D coordinates of a given chain
    chain2_pos : numpy.ndarray
        Array of shape (Nm_chain2, 3) containing 3D coordinates of a given chain
    box_size : numpy.ndarray
        Size of the simulation box in each dimension
        
    Returns:
    --------
    dist_matrix : numpy.ndarray
        Matrix of shape (Nm_chain1, Nm_chain2) containing minimum image squared distances
    """
    # Reshape to enable broadcasting
    pos1 = chain1_pos[:, np.newaxis, :]  # Shape: (N1, 1, 3)
    pos2 = chain2_pos[np.newaxis, :, :]  # Shape: (1, N2, 3)
    
    # Calculate all pairwise displacement vectors at once
    delta = pos2 - pos1  # Shape: (N1, N2, 3)
    
    # Apply minimum image convention
    delta = delta - box_size * np.round(delta / box_size)
    
    # Calculate distances
    dist_matrix = np.sum(delta**2, axis=2)
    
    return dist_matrix

def extract_atom_sigma(filename):
    """
    Extracts the sigma values for each atom type from a LAMMPS potential file.
    
    Parameters:
    -----------
    filename : str
        Path to the LAMMPS data file
    
    Returns:
    --------
    atom_sigma : dict
        Dictionary mapping atom types to their sigma values
    """
    atom_sigma = {}
    
    with open(filename, 'r') as file:
        lines = file.readlines()
        for line in lines:
            if "wf/cut" in line and "pair_coeff" in line:
                # Extract the atom type and sigma value
                parts = line.split()
                type1 = int(parts[1])
                type2 = int(parts[2])
                sigma = float(parts[5])
                atom_sigma[(type1, type2)] = sigma
                
    return atom_sigma

def generate_distance_threshold_matrix(chain1_type, chain2_type, sigma_vals):
    """
    Generate a squared distance threshold matrix based on the atom types and their sigma values.
    
    Parameters:
    -----------
    chain1_type : numpy.ndarray
        Array of shape (chain_length, ) containing atom types for chain 1
    chain2_type : numpy.ndarray
        Array of shape (chain_length, ) containing atom types for chain 2
    sigma_vals : dict
        Dictionary mapping atom types to their sigma values
    
    Returns:
    --------
    distance_threshold : float
        The distance threshold for contact analysis
    """
    # Create empty array for storing distance thresholds
    distance_threshold = np.zeros((chain1_type.shape[0], chain2_type.shape[0]))

    # Iterate through each pair of atom types
    for i in range(chain1_type.shape[0]):
        for j in range(chain2_type.shape[0]):
            # Get the sigma value for the pair of atom types
            sigma = sigma_vals.get((chain1_type[i], chain2_type[j]))
            if sigma is None:
                sigma = sigma_vals.get((chain2_type[j], chain1_type[i]))
            distance_threshold[i, j] = (1.2 * sigma)**2 # Use squared distance for efficiency

    return distance_threshold

def generate_contact_map_intramolecular(chain_positions, box_size, chain_atom_types, sigma_vals):
    """
    Perform intramoleculcar contact analysis with periodic boundary conditions at a given timestep.
    
    Parameters:
    -----------
    chain_positions : numpy.ndarray
        Array of shape (nchains, chain_length, 3) containing 3D coordinates of protein chains
    box_size : numpy.ndarray
        Size of the simulation box in each dimension
    chain_atom_types : numpy.ndarray
        Array of shape (nchains, chain_length) containing atom types for each chain
    sigma_vals : python dict
        Dictionary mapping atom types to their sigma values
    
    Returns:
    --------
    contact_map : numpy.ndarray
        Boolean matrix of shape (chain_length, chain_length) containing contact frequencies
    """
    # Create an empty array to store contact frequencies
    chain_length = chain_positions.shape[1]
    contact_frequencies = np.zeros((chain_length, chain_length))

    # Verify if the chain types are the same across all the chains
    if not np.all(np.all(chain_atom_types == chain_atom_types[0,:], axis=0)):
        raise ValueError("Chain types are not the same across all chains.")

    # Generate distance threshold matrix (1.2 * sigma)
    # Needs to be done only once
    distance_threshold = generate_distance_threshold_matrix(chain_atom_types[0], chain_atom_types[0], sigma_vals)

    # Iterate through every chain
    for chain1 in chain_positions:
        # Iterate through every other chain
        for chain2 in chain_positions:
            if not np.array_equal(chain1, chain2):
                # Generate distance matrix for the selected chains
                dist_matrix = fast_pbc_distance_matrix_vectorized(chain1_pos=chain1, chain2_pos=chain2, box_size=box_size)
                # Create contact map (excluding self-contacts)
                contact_map = (dist_matrix <= distance_threshold) & (dist_matrix > 0)
                # Add to contact frequencies
                contact_frequencies += contact_map
    return contact_frequencies

def generate_contact_map_intermolecular(chain1_positions, chain2_positions, box_size, chain1_atom_types, chain2_atom_types, sigma_vals):
    """
    Perform intermolecular contact analysis with periodic boundary conditions at a given timestep.
    
    Parameters:
    -----------
    chain1_positions : numpy.ndarray
        Array of shape (nchains, chain_length, 3) containing 3D coordinates of protein chains of type 1
    chain2_positions : numpy.ndarray
        Array of shape (nchains, chain_length, 3) containing 3D coordinates of protein chains of type 2
    box_size : numpy.ndarray
        Size of the simulation box in each dimension
    chain1_atom_types : numpy.ndarray
        Array of shape (nchains, chain_length) containing atom types for each chain
    chain2_atom_types : numpy.ndarray
        Array of shape (nchains, chain_length) containing atom types for each chain
    sigma_vals : python dict
        Dictionary mapping atom types to their sigma values
    
    Returns:
    --------
    contact_map : numpy.ndarray
        Boolean matrix of shape (chain_length1, chain_length2) containing contact frequencies
    """
    # Create an empty array to store contact frequencies
    chain_length_1 = chain1_positions.shape[1]
    chain_length_2 = chain2_positions.shape[1]
    contact_frequencies = np.zeros((chain_length_1, chain_length_2))

    # Verify if the chain types are the same across all the chains
    if not np.all(np.all(chain1_atom_types == chain1_atom_types[0,:], axis=0)):
        raise ValueError("Chain types are not the same across all chains in chain1_positions.")
    if not np.all(np.all(chain2_atom_types == chain2_atom_types[0,:], axis=0)):
        raise ValueError("Chain types are not the same across all chains in chain2_positions.")

    # Generate distance threshold matrix (1.2 * sigma)
    # Needs to be done only once
    distance_threshold = generate_distance_threshold_matrix(chain1_atom_types[0], chain2_atom_types[0], sigma_vals)

    # Iterate through every pair of chains from different groups
    for idx1, chain1 in enumerate(chain1_positions):
        for idx2, chain2 in enumerate(chain2_positions):
            if not np.array_equal(chain1, chain2):
                # Generate distance matrix for the selected chains
                dist_matrix = fast_pbc_distance_matrix_vectorized(chain1_pos=chain1, chain2_pos=chain2, box_size=box_size)
                # Create contact map (excluding self-contacts)
                contact_map = (dist_matrix <= distance_threshold) & (dist_matrix > 0)
                # Add to contact frequencies
                contact_frequencies += contact_map

    return contact_frequencies

def process_frame_intramolecular(frame_data):
    tstep, host_positions, box_size, host_types, sigma_vals = frame_data
    # Generate contact map for the current timestep
    contact_map = generate_contact_map_intramolecular(
        chain_positions=host_positions, 
        box_size=box_size,
        chain_atom_types=host_types, 
        sigma_vals=sigma_vals)
    
    return contact_map

def process_frame_intermolecular(frame_data):
    tstep, host_positions, guest_positions, box_size, host_types, guest_types, sigma_vals = frame_data
    # Generate contact map for the current timestep
    contact_map = generate_contact_map_intermolecular(
        chain1_positions=host_positions, 
        chain2_positions=guest_positions,
        box_size=box_size,
        chain1_atom_types=host_types, 
        chain2_atom_types=guest_types,
        sigma_vals=sigma_vals)
    
    return contact_map

# **********************************************************************

# Read input arguments
parser = argparse.ArgumentParser()
parser.add_argument("--traj_file", type=str, required=True) # dir where data is stored
parser.add_argument("--stride", type=int, required=True) # process every stride-th frame
parser.add_argument("--config_file", type=str, required=True) # dir where data is stored
parser.add_argument("--Nm_host", type=int, required=True) # Number of monomers in host chain
parser.add_argument("--Nm_guest", type=int, required=True) # Number of monomers in guest chain
args = parser.parse_args()

# -------------------------------------------------------------
# Read trajectory file
# -------------------------------------------------------------
timesteps, box, num_atoms, atom_positions, type_ids, mol_ids, atom_ids = read_lammps_trajectory(args.traj_file)
num_timesteps = len(timesteps)

print(f"Number of timesteps: {num_timesteps}")
print(f"Number of atoms: {num_atoms}")

# --------------------------------------------------------------
# Read configuration file
# --------------------------------------------------------------
_, num_bonds, bond_data = read_config(args.config_file)

print(f"Number of bonds: {num_bonds}")

# --------------------------------------------------------------
# Compute the number of host and guest chains
# --------------------------------------------------------------

# Create a list to store molecule IDs for guest and host chains
guest_mol_ids = []
host_mol_ids = []

Nm_host = args.Nm_host
Nm_guest = args.Nm_guest

for umolid in np.unique(mol_ids):
    # Create a mask for the current molecule ID
    mask = (mol_ids == umolid)
    # Get the number of atoms in the current molecule
    num_atoms_in_mol = np.sum(mask)
    
    if num_atoms_in_mol == Nm_host:
        host_mol_ids.append(umolid)
    elif num_atoms_in_mol == Nm_guest:
        guest_mol_ids.append(umolid)

nchain_host = len(host_mol_ids)
nchain_guest = len(guest_mol_ids)

print(f"Number of host chains: {nchain_host}")
print(f"Number of guest chains: {nchain_guest}")

# --------------------------------------------------------------
# Unless the atom IDs and Mol IDs are sorted in the lammps trajectory
# file, we need to determine the correct sequence of atoms in the chains
# --------------------------------------------------------------

# Create a dictionary mapping atom IDs to their indices in atom_ids
atom_to_index = {atom: idx for idx, atom in enumerate(atom_ids)}

# Create empty lists to store the ordered indices of host chains
host_ordered_indices = []

for umolid in host_mol_ids:
    # Create a mask for the current molecule ID
    mask = (mol_ids == umolid)
    
    # Get the atom IDs for the current molecule
    atom_ids_in_mol = atom_ids[mask]
    
    # Get the bond data for the current molecule
    bond_data_in_mol = bond_data[np.isin(bond_data[:, 2], atom_ids_in_mol) & np.isin(bond_data[:, 3], atom_ids_in_mol)]

    # Get the ordered sequence of atoms in the chain
    ordered_atoms = sequence_linear_chain(atom_ids_in_mol, bond_data_in_mol[:, 2:4])
    
    # Convert to indices in atom_ids
    ordered_indices = [atom_to_index[atom] for atom in ordered_atoms]
    
    # Append to the list of ordered indices
    host_ordered_indices.append(ordered_indices)

# Create empty lists to store the ordered indices of guest chains
guest_ordered_indices = []

for umolid in guest_mol_ids:
    # Create a mask for the current molecule ID
    mask = (mol_ids == umolid)
    
    # Get the atom IDs for the current molecule
    atom_ids_in_mol = atom_ids[mask]
    
    # Get the bond data for the current molecule
    bond_data_in_mol = bond_data[np.isin(bond_data[:, 2], atom_ids_in_mol) & np.isin(bond_data[:, 3], atom_ids_in_mol)]

    # Get the ordered sequence of atoms in the chain
    ordered_atoms = sequence_linear_chain(atom_ids_in_mol, bond_data_in_mol[:, 2:4])
    
    # Convert to indices in atom_ids
    ordered_indices = [atom_to_index[atom] for atom in ordered_atoms]
    
    # Append to the list of ordered indices
    guest_ordered_indices.append(ordered_indices)

# Convert to numpy arrays
host_ordered_indices = np.array(host_ordered_indices)
guest_ordered_indices = np.array(guest_ordered_indices)

# Create empty arrays to store the positions of host and guest chains (ordered)
host_chain_positions = np.zeros((num_timesteps, nchain_host, Nm_host, 3))
guest_chain_positions = np.zeros((num_timesteps, nchain_guest, Nm_guest, 3))

host_chain_atom_types = np.zeros((nchain_host, Nm_host), dtype=int)
guest_chain_atom_types = np.zeros((nchain_guest, Nm_guest), dtype=int)

for tstep in range(num_timesteps):
    # Extract positions of host chains
    for idx1 in range(nchain_host):
        host_chain_positions[tstep, idx1, :, :] = atom_positions[tstep, host_ordered_indices[idx1,:]]
        host_chain_atom_types[idx1, :] = type_ids[host_ordered_indices[idx1,:]]
    # Extract positions of guest chains
    for idx2 in range(nchain_guest):
        guest_chain_positions[tstep, idx2, :, :] = atom_positions[tstep, guest_ordered_indices[idx2,:]]
        guest_chain_atom_types[idx2, :] = type_ids[guest_ordered_indices[idx2,:]]


# --------------------------------------------------------------
# Read sigma values from the LAMMPS potential file
# --------------------------------------------------------------
parent_dir = Path(__file__).parent
sigma_vals = extract_atom_sigma(f"{parent_dir}/potential_60_particle_types.dat")

# --------------------------------------------------------------
# Generate intramolecular contact map for each timestep
# --------------------------------------------------------------
frames_to_process = []
for tstep in range(0, num_timesteps, args.stride):
    frames_to_process.append((tstep,
                              host_chain_positions[tstep], 
                              [box[0][1]-box[0][0], box[1][1]-box[1][0], box[2][1]-box[2][0]], 
                              host_chain_atom_types, 
                              sigma_vals
    ))

with Pool(processes=min(cpu_count(), 8)) as pool:
    results = list(tqdm(pool.imap(process_frame_intramolecular, frames_to_process), 
                        total=len(frames_to_process), 
                        desc="Computing intramolecular contact maps"))

# Combine results
contact_map = sum(results)
contact_map /= (len(results)*nchain_host*nchain_host)

# Save the contact map to a file
np.savetxt("contact_map_host-host.dat", contact_map, fmt="%.5e")

# ----------------------------------------------------------------
# Plot and save contact map
# ----------------------------------------------------------------
fig, ax = plt.subplots(figsize=(12, 12))

sns.heatmap(contact_map, cmap="viridis", cbar=True, annot=False, cbar_kws={'label': 'Normalized contact frequency'})
ax.tick_params(axis='both', which='both', bottom=False, top=False, left=False, right=False)

ax.set_xlabel("host protein")
ax.set_ylabel("host protein")

plt.tight_layout()
plt.savefig(f"contact_map_host-host.pdf")


# --------------------------------------------------------------
# Generate intermolecular contact map for each timestep
# --------------------------------------------------------------
frames_to_process = []
for tstep in range(0, num_timesteps, args.stride):
    frames_to_process.append((tstep,
                              host_chain_positions[tstep],
                              guest_chain_positions[tstep],
                              [box[0][1]-box[0][0], box[1][1]-box[1][0], box[2][1]-box[2][0]], 
                              host_chain_atom_types, 
                              guest_chain_atom_types,
                              sigma_vals
    ))

with Pool(processes=min(cpu_count(), 8)) as pool:
    results = list(tqdm(pool.imap(process_frame_intermolecular, frames_to_process), 
                        total=len(frames_to_process), 
                        desc="Computing intermolecular contact maps"))

# Combine results
contact_map = sum(results)
contact_map /= (len(results)*nchain_host*nchain_guest)

# Save the contact map to a file
np.savetxt("contact_map_host-guest.dat", contact_map, fmt="%.5e")

# ----------------------------------------------------------------
# Plot and save contact map
# ----------------------------------------------------------------
fig, ax = plt.subplots(figsize=(12, 12))
sns.heatmap(contact_map, cmap="viridis", cbar=True, annot=False, cbar_kws={'label': 'Normalized contact frequency'})
ax.tick_params(axis='both', which='both', bottom=False, top=False, left=False, right=False)

ax.set_xlabel("guest protein")
ax.set_ylabel("host protein")

plt.tight_layout()
plt.savefig(f"contact_map_host-guest.pdf")

# --------------------------------------------------------------