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
What?
Analyze single chain simulation of protein/RNA chains to find out interactions 
between different residues.

NOTE: ****************************************************************************************
Specifically designed for LAMMPS trajectory files generated from simulations of single chains.

**********************************************************************************************

Steps are as follows:
1. Load the LAMMPS simulation trajectory.
2. Read config file to determine the chain sequence and the mapping of residues to bead types.
3. Check if atom ids are sorted in the trajectory file. If not, sort them first using the bond data from
    the config file.
4. Generate cumulative interaction map.

Input arguments:
    - trajectory_file: Path to the LAMMPS trajectory file containing the simulation data.
    - config_file: Path to the LAMMPS configuration file containing the system information.

Output:
    - contact_map.dat : Text file containing the cumulative contact map for the chain.
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

def compute_contacts(position, sigma_vals, type_ids, box_size):
    """
    Compute the contact map for a single chain based on the atom positions and sigma values.
    
    Parameters:
    -----------
    position : np.ndarray
        A 3D array of shape (num_frames, num_atoms, 3) containing the positions of
    sigma_vals : dict
        A dictionary mapping atom types to their sigma values
    type_ids : np.ndarray
        A 1D array of shape (num_atoms,) containing the types of each atom

    Returns:
    --------
    contact_map : np.ndarray
        A 2D array representing the contact map of the chain
    """
    # Generate a distance threshold matrix based on the sigma values for each pair of atom types
    distance_threshold = generate_distance_threshold_matrix(type_ids, type_ids, sigma_vals)

    # Initialize contact map
    num_atoms = type_ids.shape[0]
    contact_frequencies = np.zeros((num_atoms, num_atoms))

    # Compute contact map for each frame and accumulate
    for frame in tqdm(position, desc="Computing contact map"):
        # Compute pairwise squared distances between atoms
        dist_matrix = fast_pbc_distance_matrix_vectorized(frame, frame, box_size)
        # Create contact map (excluding self-contacts)
        contact_map = (dist_matrix <= distance_threshold) & (dist_matrix > 0)
        # Add to contact frequencies
        contact_frequencies += contact_map

    # Normalize by the number of frames to get contact frequencies
    contact_frequencies /= position.shape[0]

    return contact_frequencies


if "__main__" == __name__:
    # Read input arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--traj_file", type=str, required=True) # trajectory file to read
    parser.add_argument("--config_file", type=str, required=True) # dir where data is stored
    args = parser.parse_args()

    # Read the trajectory file
    timesteps, box, num_atoms, atom_positions, type_ids, mol_ids, atom_ids = read_lammps_trajectory(args.traj_file)
    num_timesteps = len(timesteps)

    print(f"Number of timesteps: {num_timesteps}")
    print(f"Number of atoms: {num_atoms}")


    # Read the config file to get bond information
    _, num_bonds, bond_data = read_config(args.config_file)

    print(f"Number of bonds: {num_bonds}")


    # Check if the atom ids are sorted in the trajectory file.
    if(np.all(atom_ids[:-1] < atom_ids[1:])):
        print("Atom IDs are sorted in the trajectory file. Great!")
    else:
        print("Atom IDs are not sorted in the trajectory file. Sorting them now...")
        # Create a dictionary mapping atom IDs to their indices in atom_ids
        atom_to_index = {atom: idx for idx, atom in enumerate(atom_ids)}
        
        ordered_atom_ids = sequence_linear_chain(atom_ids, bond_data[:, 2:4])
        ordered_indices = np.array([atom_to_index[atom] for atom in ordered_atom_ids])
    
        # Reorder atom_positions, type_ids, mol_ids, and atom_ids according to the new order
        atom_positions = atom_positions[:, ordered_indices, :]
        type_ids = type_ids[ordered_indices]
        mol_ids = mol_ids[ordered_indices]
        atom_ids = atom_ids[ordered_indices]
    
    # --------------------------------------------------------------
    # Read sigma values from the LAMMPS potential file
    # --------------------------------------------------------------
    parent_dir = Path(__file__).parent
    sigma_vals = extract_atom_sigma(f"{parent_dir}/potentials.dat")

    # --------------------------------------------------------------
    # Compute the contact map for the chain
    # --------------------------------------------------------------
    contact_map = compute_contacts(atom_positions, sigma_vals, type_ids, 
                                   [box[0][1]-box[0][0], box[1][1]-box[1][0], box[2][1]-box[2][0]])
    

    # Save the contact map to a text file
    np.savetxt("contact_map.dat", contact_map, fmt="%.6f")

    # Plot contact map
    fig, ax = plt.subplots(figsize=(4, 4))
    sns.heatmap(contact_map, cmap="Blues", cbar=True, annot=False, cbar_kws={'label': 'Normalized contact frequency'})
    
    ax.set_xlabel("Residue Index")
    ax.set_ylabel("Residue Index")
    
    plt.savefig("contact_map.pdf", dpi=600)
