import numpy as np
import matplotlib.pyplot as plt
import argparse
from pathlib import Path
import json
from tqdm import tqdm

"""
What?
Analyze single chain simulation of protein/RNA chains to calculate the radius of gyration (Rg) and its distribution.

NOTE: ****************************************************************************************
Specifically designed for LAMMPS trajectory files generated from simulations of single chains.
**********************************************************************************************

Steps are as follows:
1. Load the LAMPPS simulation trajectory.
2. For each frame, calculate the radius of gyration of the chain.
3. Plot the distribution of Rg values across the trajectory. 

Input arguments:
    - trajectory_file: Path to the LAMMPS trajectory file containing the simulation data.
    - molecule type: Type of molecule (protein or RNA) to analyze.

Output:
    - Rg.dat : Text file containing the Rg values for each frame.
    - Rg_distribution.dat : Text file containing the histogram data for the Rg distribution.
"""

parser = argparse.ArgumentParser()
parser.add_argument("--path", type=str, required=True) # dir where data is stored
parser.add_argument("--mol_type", type=str, required=True, choices=["protein", "RNA"]) # Type of molecule (protein or RNA) to analyze
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

def extract_particle_masses(types, mass_dict):
    """
    Extract the masses of particles given an array containing particle types
    """    
    masses = []
    for type in types:
        mass = mass_dict.get(str(type), None)  # Convert type to string for dictionary lookup
        if mass is None:
            raise ValueError(f"Mass for particle type {type} not found in the mass dictionary.")
        masses.append(mass)

    return masses



# Call functions
if __name__=="__main__":
    file_path = args.path
    mol_type = args.mol_type

    # Read the trajectory file
    tsteps, box_size, num_atoms, atom_positions, atom_types = read_lammps_trajectory(file_path)


    # Load particle mass dictionary
    parent_dir = Path(__file__).parent
    with open(f'{parent_dir}/particle_masses.json', 'r') as f:
        particle_masses = json.load(f)


    # Extract masses for each particle type
    masses = extract_particle_masses(atom_types, particle_masses)
    

    # Compute center of mass and Rg for each frame using numpy vectorized operations
    Rg_values = []
    mass_array = np.array(masses)
    total_mass = np.sum(mass_array)
    for frame in tqdm(range(len(tsteps))):
        positions = atom_positions[frame]
        center_of_mass = np.sum(positions * mass_array[:, np.newaxis], axis=0) / total_mass
        Rg_squared = np.sum(mass_array * np.sum((positions - center_of_mass) ** 2, axis=1)) / total_mass
        Rg = np.sqrt(Rg_squared)
        Rg_values.append(Rg)
    Rg_values = np.array(Rg_values)

    # Generate histogram of Rg values
    Rg_hist, bin_edges = np.histogram(Rg_values, bins=100, density=False)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

    # Save Rg values and histogram data to text files
    np.savetxt(f'Rg.dat', Rg_values)
    np.savetxt(f'Rg_distribution.dat', np.column_stack((bin_centers, Rg_hist)), header='Rg_bin_center\t Rg_count')

