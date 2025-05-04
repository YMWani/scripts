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
plt.style.use(f"{current_dir}/../plotting/ymw.mplstyle")
import argparse

"""
From a given trajectory of the guest chains, we want to measure the radius of gyration of the 
chains at different locations inside the condensate.

Steps are as follows:
1. Read lammps trajectory file (guest chains)
2. Compute COM positions of the chains for every frame
3. Compute radius of gyration and asphericity of the chains using the gyration tensor
   at every frame.
4. Wrap COM positions inside simulation box
5. For every frame, sort the COM positions inside bins along the z direction
6. Compute the mean Rg and mean asphericity of chains inside each bin, over the entire
   trajectory.
7. Save and plot data.
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
    
def compute_gyration_tensor(com_pos, atom_positions):
    """
    For a given set of chains, this function computes the radius of gyration of the chains
    by first computing the gyration tensor and its eigenvalues.

    Gyration tensor:
    S_mn = 1/N * sum_{i=1}^{N} ri_m*ri_n,
    where N: Number of monomers in a chain
          ri: relative position of a monomer w.r.t. COM of the chain
    
    Radius of gyration:
    If a1<=a2<=a3 are the eigenvalues of the gyration tensor,
    then,
    Rg^2 = a1+a2+a3
    
    Asphericity:
    b = a3 - 0.5*(a1+a2)

    Arguments:
    - com_pos (numpy array): [#chains, 3]
    - atom_positions (numpy array): [#chains, #atoms-per-chain, 3]

    Returns:
    - Rgs (numpy array): [#chains,]
    """
    nchains = com_pos.shape[0] # number of chains
    Rgs = np.zeros(nchains) # Create empty array for storing radius of gyrations
    asphericities = np.zeros(nchains) # Create empty array for storing apshericity of the chains
    for i in range(nchains):
        relative_pos = atom_positions[i] - com_pos[i] # shape: [#atoms-per-chain, 3]
        gyration_tensor = np.einsum("jm,jn->mn", relative_pos, relative_pos)/relative_pos.shape[0]
        eig_vals, eig_vecs = np.linalg.eig(gyration_tensor)
        eig_vals_sorted = np.sort(eig_vals)
        Rgs[i] = np.sqrt(np.sum(eig_vals))
        asphericities[i] = eig_vals_sorted[2] - 0.5*(eig_vals_sorted[0]+eig_vals_sorted[1])
    return Rgs, asphericities




# Call functions
if __name__=="__main__":
    file_path = args.path
    traj = args.traj_guest
    Nm = args.chain_length

    print(f"Extracting data from trajectory file.")
    timesteps, box_sizes, num_atoms, atom_positions, atom_types = read_lammps_trajectory(f"{file_path}/{traj}", Nm)

    print(f"Trajectory file details:")
    print(f"# frames   :{len(timesteps)}")
    print(f"# particles:{num_atoms}")
    print(f"Atom types: {atom_types}")
    
    # Reshape position array to separate each chain
    atom_positions_reshaped = np.reshape(atom_positions,
                                         (atom_positions.shape[0],
                                          atom_positions.shape[1]//Nm,
                                          Nm,
                                          atom_positions.shape[2])) # [#frames, #chains, #atoms-per-chain, 3]
    # Get the mass of chain monomers using particle types
    chain_masses = extract_particle_masses(atom_types)
    print(chain_masses)
    exit()
    # Compute center of mass positions of the chains at every timestep
    # NOTE: Particle positions correspond to unwrapped positions.
    print(f"Computing COM positions of the guest chains.")
    # COM_positions shape - (#frames, #chains, 3)
    COM_positions = compute_COM_positions(atom_positions_reshaped, chain_masses)
    
    # Compute radius of gyration of chains for every frame
    print(f"Computing radius of gyration and asphericity of chains at every frame.")
    radius_gyration = []
    asphericities = []
    for idx in tqdm(range(COM_positions.shape[0])):
        Rgs, aspheris = compute_gyration_tensor(COM_positions[idx], atom_positions_reshaped[idx])
        radius_gyration.append(Rgs)
        asphericities.append(aspheris)
    radius_gyration = np.array(radius_gyration)
    asphericities = np.array(asphericities)

    # Wrap COM positions inside simulation box to histogram data
    print(f"Wrapping atom and center of mass positions.")
    COM_wrapped = wrap_positions_inside_sim_box(COM_positions, box_sizes)

    """
    With the information about chain COMs and Rgs at every timestep/frame,
    we will first sort chains into bins at every timestep and add the Rg
    values to the respective bins.
    """
    nbins = 50
    bin_edges = np.linspace(box_sizes[2][0], box_sizes[2][1], nbins+1)
    bin_indices = (np.digitize(COM_wrapped[:,:,2], bin_edges) - 1).reshape((-1,))
    radius_gyration_reshaped = radius_gyration.reshape((-1,))
    asphericities_reshaped = asphericities.reshape((-1,))
    Rg_binned = np.zeros((nbins,3))
    asphericity_binned = np.zeros((nbins,3))
    for idx in range(nbins):
        mask = (bin_indices == idx)
        mean_Rg = np.mean(radius_gyration_reshaped[mask])
        err_Rg = sem(radius_gyration_reshaped[mask])
        Rg_binned[idx,0] = (bin_edges[idx] + bin_edges[idx+1])/2.
        Rg_binned[idx,1] = mean_Rg
        Rg_binned[idx,2] = err_Rg

        mean_aspheri = np.mean(asphericities_reshaped[mask])
        err_aspheri = sem(asphericities_reshaped[mask])
        asphericity_binned[idx,0] = (bin_edges[idx] + bin_edges[idx+1])/2.
        asphericity_binned[idx,1] = mean_aspheri
        asphericity_binned[idx,2] = err_aspheri

    np.savetxt("Rg.dat", Rg_binned, header="bin_center\t<Rg>\t+/-")
    np.savetxt("asphericity.dat", asphericity_binned, header="bin_center\t<asphericity>\t+/-")

    # Plot data
    fig, ax = plt.subplots()
    ax.errorbar(Rg_binned[:,0], Rg_binned[:,1], yerr=Rg_binned[:,2], marker="o", lw=1.0)
    ax.set_xlabel(r"$z \, (\AA)$")
    ax.set_ylabel(r"$\mathrm{R_g}$")
    plt.savefig("Rg.pdf", dpi=600)

    fig, ax = plt.subplots()
    ax.errorbar(asphericity_binned[:,0], asphericity_binned[:,1], yerr=asphericity_binned[:,2],
                marker="o", lw=1.0)
    ax.set_xlabel(r"$z \, (\AA)$")
    ax.set_ylabel(r"$b$")
    plt.savefig("asphericity.pdf", dpi=600)
