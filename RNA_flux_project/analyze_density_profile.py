import numpy as np
import matplotlib.pyplot as plt
import pathlib
from tqdm import tqdm
from pathlib import Path
import json
from matplotlib.cm import viridis
current_dir = pathlib.Path(__file__).resolve().parent
plt.style.use(f"{current_dir}/../plotting/ymw.mplstyle")
import argparse

"""
Analyze the density profile of "guest" chains in the simulation box along the
z direction (long axis).

Arguments:
- filename: trajectory file name
- chain_length: Number of monomers in a chain
"""

parser = argparse.ArgumentParser()
parser.add_argument("--path", type=str, required=True) # dir where data is stored
parser.add_argument("--fname", type=str, required=True) # trajectory file
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
    
    # Sanity check: Since all the chains have the same identity, the atom types should be exactly same.
    atom_types = np.reshape(atom_types, (atom_types.shape[0]//chain_length, chain_length))
    is_same = np.allclose(atom_types, atom_types[0])
    if is_same:
        print("All chains have the same atom types. Great!")
    atom_types = atom_types[0]

    # Convert all lists to arrays
    timesteps = np.array(timesteps)
    atom_positions = np.array(atom_positions)
    
    return timesteps, box_sizes, num_atoms, atom_positions, atom_types

def get_mass_by_id(amino_dict, id_to_find):
    for amino_acid, data in amino_dict.items():
        if data['id'] == id_to_find:
            return data['mass']
    return None

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

def compute_number_density_profile(positions, simBox, nbins):
    """
    Compute the density profile of all atoms/chains along the long axis 
    of the simulation box.

    NOTE: Depending on the "positions" array we either compute the number 
    density of all atoms or of the chain center of masses.

    NOTE: We expect wrapped positions.

    Arguments:
    - positions (3d numpy array): [#frames, #chains, 3] / [#frames, #atoms, 3]
    - simBox (2d numpy array): [[xmin,xmax] [ymin,ymax] [zmin,zmax]]
    - nbins (int): # bins to histogram data. Higher bins leads to finer resolution,
            but can lead to larger noise.
    """
    nframes = positions.shape[0]
    density_profile = np.zeros((nframes, nbins))
    bin_width = (simBox[2][1] - simBox[2][0])/nbins
    bin_volume = (simBox[0][1] - simBox[0][0]) * (simBox[1][1] - simBox[1][0]) * bin_width
    for idx, frame_pos in enumerate(tqdm(positions)):
        hist_, bin_edges = np.histogram(frame_pos[:,2], bins=nbins,
                                   range=(simBox[2][0], simBox[2][1]))
        density_profile[idx] = hist_/bin_volume
    bin_centers = (bin_edges[:-1] + bin_edges[1:])/2.
    return density_profile, bin_centers






# Call functions
if __name__=="__main__":
    file_path=args.path
    Nm = args.chain_length

    print(f"Extracting data from trajectory file.")
    timesteps, box_sizes, num_atoms, atom_positions, atom_types = read_lammps_trajectory(f"{file_path}/guest_chains.lammpstrj", Nm)
    
    print(f"Trajectory file details:")
    print(f"# frames   :{len(timesteps)}")
    print(f"# particles:{num_atoms}")
    
    # Reshape the position array to separate each chain
    # New shape - (#frames, #chains, chain_length, 3)
    atom_positions_reshaped = np.reshape(atom_positions,(atom_positions.shape[0], 
                                                         atom_positions.shape[1]//args.chain_length,
                                                         args.chain_length, 
                                                         atom_positions.shape[2]))
    # Get the mass of chain monomers using particle types
    chain_masses = extract_particle_masses(atom_types)
    # Compute center of mass positions of the chains at every timestep
    # NOTE: Particle positions correspond to unwrapped positions.
    # COM_positions shape - (#frames, #chains, 3)
    COM_positions = compute_COM_positions(atom_positions_reshaped, chain_masses)

    print(f"Wrapping atom and center of mass positions.")
    # Wrap COM positions inside simulation box
    COM_wrapped = wrap_positions_inside_sim_box(COM_positions, box_sizes)
    
    # Wrap atom positions for density profile of all atoms
    atom_positions_wrapped = wrap_positions_inside_sim_box(atom_positions, box_sizes)

    # Get the density profile of all atoms over time
    print(f"Computing density profile of all guest chain atoms over time.")
    nbins = 100
    number_density_all_atoms, bin_centers = compute_number_density_profile(atom_positions_wrapped, box_sizes,
                                            nbins=nbins)
    # Save data
    np.savetxt(f"{file_path}/number_density_all_atoms.dat",
               np.column_stack((np.tile(bin_centers, number_density_all_atoms.shape[0]), 
                                np.reshape(number_density_all_atoms, (-1,1)))),
               header=f"bin_centers\tnum_density\t(nbins={nbins})")
    
    # Get density profile of the COM of chains over time
    print(f"Computing density profile of guest chain COMs over time.")
    number_density_COM, bin_centers = compute_number_density_profile(COM_wrapped, box_sizes,
                                            nbins=nbins)
    # Save data
    np.savetxt(f"{file_path}/number_density_COM.dat",
               np.column_stack((np.tile(bin_centers, number_density_COM.shape[0]), 
                                np.reshape(number_density_COM, (-1,1)))),
               header=f"bin_centers\tnum_density\t(nbins={nbins})")
    
    