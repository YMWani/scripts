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
import matplotlib.colors as mcolors

"""
What?
Here, we analyze measure the energy interaction map between two dimers (protein/RNA). 

Why?
To identify what residues are important for the interaction between the two dimers, and to determine a metric to measure 
binding of the two molecules.

How?
From dimer simulation trajectories, we compute the pairwise interaction map between the two dimers. 
We do this by computing the pairwise interaction energy between all residues in the two dimers for each frame of the trajectory.

Input:
- LAMMPS dimer simulation trajectories (protein/RNA)

Output:
- Pairwise interaction energy map between the two dimers (protein/RNA)
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

def extract_WF_params(filename):
    """
    Extracts the parameters for each atom type from a LAMMPS potential file.

    Parameters:
    -----------
    filename : str
        Path to the LAMMPS data file
    
    Returns:
    --------
    atom_params : dict
        Dictionary mapping atom types to their WF parameters (sigma, epsilon)
    """
    atom_params = {}
    
    with open(filename, 'r') as file:
        lines = file.readlines()
        for line in lines:
            if "wf/cut" in line and "pair_coeff" in line:
                # Extract the atom type and sigma value
                parts = line.split()
                type1 = int(parts[1])
                type2 = int(parts[2])
                epsilon = float(parts[4])
                sigma = float(parts[5])
                nu = float(parts[6])
                mu = float(parts[7])
                atom_params[(type1, type2)] = (sigma, epsilon, nu, mu)

    return atom_params

def WF_energy(r, sigma, epsilon, nu=1.0, mu=2.0):
    R = 3.*sigma

    alpha = 2. * nu * (R/sigma)**(2.*mu) * ((2.*nu + 1.)/(2. * nu * ((R/sigma)**(2.*mu) - 1.)))**(2.*nu + 1.)
    V = epsilon * alpha * ((sigma/r)**(2*mu) - 1.) * ((R/r)**(2*mu) - 1.)**(2*nu)

    return V

def coul_debye_energy(r, q1, q2):
    kappa = 0.126 # corresponds to 150 mM salt concentration
    dielectric_constant = 80.0
    cutoff = 35.0 # in Angstroms, beyond which we consider the interaction negligible
    if q1 == 0.0 or q2 == 0.0:
        return 0.0
    elif r <= cutoff:
        return (1.0 / dielectric_constant) * (q1 * q2) * np.exp(-kappa * r) / r
    else:
        return 0.0

def compute_pairwise_interaction_energy_map(mol1_positions, mol2_positions, mol1_types, mol2_types, charges, WF_params):
    """
    Computes the pairwise interaction energy map between two molecules for each frame of the trajectory.

    Parameters:
    -----------
    mol1_positions : (num_frames, num_atoms_mol1, 3)
        Array of positions for molecule 1 across all frames
    mol2_positions : (num_frames, num_atoms_mol2, 3)
        Array of positions for molecule 2 across all frames
    mol1_types : (num_atoms_mol1,)
        Array of atom types for molecule 1
    mol2_types : (num_atoms_mol2,)
        Array of atom types for molecule 2
    charges : dict
        Dictionary mapping atom types to their charges
    WF_params : dict
        Dictionary mapping pairs of atom types to their WF parameters (sigma, epsilon, nu, mu)

    Returns:
    --------
    interaction_energy_maps : (num_frames, num_atoms_mol1, num_atoms_mol2)
        Array containing the pairwise interaction energy maps for each frame
    """
    # Dimensions: frames x atoms_in_mol1 x atoms_in_mol2.
    num_frames = mol1_positions.shape[0]
    num_atoms_mol1 = mol1_positions.shape[1]
    num_atoms_mol2 = mol2_positions.shape[1]

    interaction_energy_maps = np.zeros((num_frames, num_atoms_mol1, num_atoms_mol2), dtype=float)

    # Build per-atom-pair parameter grids once so each frame reuses them.
    sigma_grid = np.zeros((num_atoms_mol1, num_atoms_mol2), dtype=float)
    epsilon_grid = np.zeros((num_atoms_mol1, num_atoms_mol2), dtype=float)
    nu_grid = np.zeros((num_atoms_mol1, num_atoms_mol2), dtype=float)
    mu_grid = np.zeros((num_atoms_mol1, num_atoms_mol2), dtype=float)
    has_wf_params = np.zeros((num_atoms_mol1, num_atoms_mol2), dtype=bool)

    missing_pairs = set()
    unique_types_1 = np.unique(mol1_types)
    unique_types_2 = np.unique(mol2_types)

    for type_i in unique_types_1:
        row_mask = mol1_types == type_i
        for type_j in unique_types_2:
            col_mask = mol2_types == type_j
            # Broadcast masks to mark all (i, j) atom-index pairs with these types.
            pair_mask = row_mask[:, None] & col_mask[None, :]

            params = WF_params.get((type_i, type_j))
            if params is None:
                missing_pairs.add((int(type_i), int(type_j)))
                continue

            sigma, epsilon, nu, mu = params
            sigma_grid[pair_mask] = sigma
            epsilon_grid[pair_mask] = epsilon
            nu_grid[pair_mask] = nu
            mu_grid[pair_mask] = mu
            has_wf_params[pair_mask] = True

    for type_i, type_j in sorted(missing_pairs):
        print(f"Warning: No WF parameters found for atom types {type_i} and {type_j}. Setting WF energy to 0.")

    q_i = np.array([charges.get(int(t), 0.0) for t in mol1_types], dtype=float)[:, None]
    q_j = np.array([charges.get(int(t), 0.0) for t in mol2_types], dtype=float)[None, :]
    # Charge product is also a reusable atom-pair grid.
    q_product = q_i * q_j
    has_charge_pair = q_product != 0.0

    kappa = 0.126
    dielectric_constant = 80.0
    cutoff = 35.0
    eps = np.finfo(float).eps

    within_cutoff_tracker = np.zeros((num_atoms_mol1, num_atoms_mol2), dtype=int)

    for frame in tqdm(range(num_frames), desc="Computing interaction energy maps"):
        # Pairwise displacement vectors for all atom pairs in this frame:
        # (n1, 1, 3) - (1, n2, 3) -> (n1, n2, 3)
        r_vec = mol1_positions[frame, :, None, :] - mol2_positions[frame, None, :, :]
        r = np.linalg.norm(r_vec, axis=2)
        # Avoid singular values for terms containing 1/r.
        r_safe = np.maximum(r, eps)
        # Update the within-cutoff tracker
        within_cutoff_tracker += (r_safe <= 35.0).astype(int)

        V_WF = np.zeros_like(r)
        if np.any(has_wf_params):
            # Evaluate WF only where parameters exist.
            V_WF[has_wf_params] = WF_energy(
                r_safe[has_wf_params],
                sigma_grid[has_wf_params],
                epsilon_grid[has_wf_params],
                nu_grid[has_wf_params],
                mu_grid[has_wf_params],
            )

        V_coulomb = np.zeros_like(r)
        # Coulomb only for charged pairs, within cutoff, and nonzero distances.
        coulomb_mask = has_charge_pair & (r <= cutoff) & (r > 0.0)
        V_coulomb[coulomb_mask] = (
            (1.0 / dielectric_constant)
            * q_product[coulomb_mask]
            * np.exp(-kappa * r[coulomb_mask])
            / r[coulomb_mask]
        )

        interaction_energy_maps[frame] = V_WF + V_coulomb
    return interaction_energy_maps, within_cutoff_tracker

def compute_sustained_interactions(interaction_energy_maps, energy_threshold=-0.3):
    num_frames = interaction_energy_maps.shape[0]
    sustained_interaction_map = np.zeros((interaction_energy_maps.shape[1], interaction_energy_maps.shape[2]), dtype=int)

    for i in range(interaction_energy_maps.shape[1]):
        for j in range(interaction_energy_maps.shape[2]):
            # Create a mask to identify frames where the interaction energy is below the threshold
            interaction_mask = interaction_energy_maps[:, i, j] < energy_threshold
            # Count the number of consecutive frames where the interaction is sustained
            padded_mask = np.pad(interaction_mask, (1, 1), mode='constant', constant_values=False)
            diff = np.diff(padded_mask.astype(int))
            
            start_indices = np.where(diff == 1)[0]
            end_indices = np.where(diff == -1)[0]
            
            sustained_interactions = end_indices - start_indices # lenght of each sustained interaction segment
            avg_sustained_interactions = np.mean(sustained_interactions) if len(sustained_interactions) > 0 else 0 # average length of sustained interaction segments

            sustained_interaction_map[i, j] = avg_sustained_interactions

    return sustained_interaction_map



# -------------------------------------------------------------------------
if "__main__" == __name__:
    # Read input arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--traj", type=str, required=True) # trajectory file to read
    args = parser.parse_args()

    # Read the trajectory file
    timesteps, box, num_atoms, atom_positions, type_ids, mol_ids, atom_ids = read_lammps_trajectory(args.traj)
    num_timesteps = len(timesteps)

    # Print all relevant information about the trajectory
    print(f"Number of timesteps: {num_timesteps}")
    print(f"Box size: {box}")
    print(f"Number of atoms: {num_atoms}")
    print(f"Atom position shape: {atom_positions.shape}")
    print(f"Molecule IDs: {np.unique(mol_ids)}")

    
    # Extract positions of the two dimers
    molecule_1_positions = atom_positions[:, mol_ids == 1, :]
    molecule_1_types = type_ids[mol_ids == 1]
    molecule_2_positions = atom_positions[:, mol_ids == 2, :]
    molecule_2_types = type_ids[mol_ids == 2]

    # Read sigma values from the LAMMPS potential file
    parent_dir = Path(__file__).parent
    WF_params = extract_WF_params(f"{parent_dir}/potentials.dat")

    # Create a mapping of atom types to their charges (for Coulomb interactions)
    charges = {3: 0.75, 23: 0.75,
               5: 0.75, 25: 0.75,
               16: 0.375, 36: 0.375,
               7: -0.75, 27: -0.75,
               8: -0.75, 28: -0.75,
               41: -0.75,
               42: -0.75,
               43: -0.75,
               44: -0.75}
    
    # Compute the pairwise interaction energy map between the two molecules for each frame
    interaction_energy_maps, within_cutoff_tracker = compute_pairwise_interaction_energy_map(molecule_1_positions, molecule_2_positions, molecule_1_types, molecule_2_types, charges, WF_params)
    
    # for i in range(interaction_energy_maps.shape[0]):
    #     print(np.max(interaction_energy_maps[i]), np.min(interaction_energy_maps[i]), np.mean(interaction_energy_maps[i]))

    # Compute sustained interactions
    sustained_interaction_map = compute_sustained_interactions(interaction_energy_maps)
    
    # Plot the sustained interaction map
    fig, ax = plt.subplots(figsize=(8, 7))
    sns.heatmap(sustained_interaction_map, cmap='viridis', ax=ax)
    ax.set_xlabel("RNA")
    ax.set_ylabel("Med1-IDR")
    fig.savefig("sustained_interaction_map.pdf", dpi=300, bbox_inches='tight')
