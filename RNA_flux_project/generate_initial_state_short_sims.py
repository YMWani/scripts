import sys
import numpy as np
import argparse
from pathlib import Path
import json

parser = argparse.ArgumentParser()
parser.add_argument("--traj_path", type=str, required=True, help="Path to the directory containing the trajectory files for the initial state.")
parser.add_argument("--config_path", type=str, required=True, help="Path to the directory containing the configuration files for the initial state.")
parser.add_argument("--snapshot_index", type=int, required=True, help="Index of the snapshot to be used for the slab simulation.")
args = parser.parse_args()


def read_trajectory(file_name):
    """
    Reads the entire LAMMPS trajectory file and returns all timesteps, coordinates, and box sizes.
    Returns:
        timesteps  : np.ndarray, shape (num_frames,)
        coords     : np.ndarray, shape (num_frames, num_atoms, 7)
        box_sizes  : np.ndarray, shape (num_frames, 3, 2)
    """
    # First pass: count frames and number of atoms
    num_frames = 0
    num_atoms = 0

    with open(file_name, 'r') as file:
        for line in file:
            if 'ITEM: TIMESTEP' in line:
                num_frames += 1
            elif 'ITEM: NUMBER OF ATOMS' in line and num_atoms == 0:
                num_atoms = int(next(file).strip())

    if num_frames == 0:
        raise ValueError("No timestep found in the file.")

    # Pre-allocate arrays for all frames
    timesteps = np.zeros(num_frames, dtype=int)
    box_sizes = np.zeros((num_frames, 3, 2), dtype=float)
    coords = np.zeros((num_frames, num_atoms, 7), dtype=float)

    # Second pass: read all frames
    frame_idx = -1
    with open(file_name, 'r') as file:
        for line in file:
            if 'ITEM: TIMESTEP' in line:
                frame_idx += 1
                timesteps[frame_idx] = int(next(file).strip())

            elif 'ITEM: BOX BOUNDS' in line:
                for i in range(3):
                    box_sizes[frame_idx, i] = list(map(float, next(file).strip().split()))

            elif 'ITEM: ATOMS' in line:
                for i in range(num_atoms):
                    atom_data = next(file).strip().split()
                    coords[frame_idx, i, 0] = int(atom_data[0])
                    coords[frame_idx, i, 1] = int(atom_data[1])
                    coords[frame_idx, i, 2] = int(atom_data[2])
                    coords[frame_idx, i, 3] = float(atom_data[3])
                    coords[frame_idx, i, 4] = float(atom_data[4])
                    coords[frame_idx, i, 5] = float(atom_data[5])
                    coords[frame_idx, i, 6] = float(atom_data[6])

    return timesteps, coords, box_sizes

def read_config(file_name):
    '''
    Reads a LAMMPS configuration file and extracts the number of atoms, number of bonds,
    and bond information.

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
    num_angles = 0
    angles = []
    in_angle_section = False

    with open(file_name, 'r') as file:
        for line in file:
            line = line.strip()

            # Read number of atoms
            if 'atoms' in line and num_atoms == 0:
                num_atoms = int(line.split()[0])

            # Read number of bonds
            elif 'bonds' in line and num_bonds == 0:
                num_bonds = int(line.split()[0])

            # Read number of angles
            elif 'angles' in line and num_angles == 0:
                num_angles = int(line.split()[0])

            # Read bond information
            elif 'Bonds' in line:
                in_bond_section = True
                continue  # Skip the "Bonds" header

            elif 'Angles' in line:
                in_angle_section = True
                in_bond_section = False
                continue

            # Start reading bond data if in the bond section
            if in_bond_section:
                if len(line.split()) != 4:
                    continue  # End of bond section
                bond_data = [int(x) for x in line.split()]
                bond_id = bond_data[0]
                bond_type = bond_data[1]
                first_atom_id = bond_data[2]
                second_atom_id = bond_data[3]
                bonds.append((bond_id, bond_type, first_atom_id, second_atom_id))

            # Start reading angle data if in the angle section
            if in_angle_section:
                if len(line.split()) != 5:
                    continue
                angle_data = [int(x) for x in line.split()]
                angle_id = angle_data[0]
                angle_type = angle_data[1]
                first_atom_id = angle_data[2]
                second_atom_id = angle_data[3]
                third_atom_id = angle_data[4]
                angles.append((angle_id, angle_type, first_atom_id, second_atom_id, third_atom_id))

    if num_angles > 0:
        return num_atoms, num_bonds, bonds, num_angles, angles
    else:
        return num_atoms, num_bonds, bonds

def write_config(file_name, required_box_bounds, protein_coords, num_atoms, num_bonds, bond_data, angle_data=None):
    """
    Write a config file for a slab simulation given the follwing information:
    - required_box_bounds (list): A list containing the lower and upper bounds of the slab that we want to generate
    - protein_coords (list of tuples): A list of tuples containing details for each atom
                                      (atom_id, mol_id, atom_type, atom_charge, xcoord, ycoord, zcoord)
    - num_atoms (int): Number of particles in the system
    - num_bonds (int): Number of bonds in the system
    - bond_data (list of tuples): Bond data for the protein chains
                                 (bond_id, bond_type, first_atom_id, second_atom_id)
    
    Returns: Config file for slab simulations - "initialSlab.dat"
    """
    parent_dir = Path(__file__).parent
    with open(f'{parent_dir}/amino_acid_dict.json', 'r') as f:
        aa_dict = json.load(f)
    
    num_atom_types = 60 # Our setup requires 3*20 atom types

    if angle_data is not None:
        num_angles = len(angle_data)

    configFile = open(f'{file_name}','w')
    configFile.write('LAMMPS data file for slab of IDP\n\n')

    # Overall data
    configFile.write(f'{num_atoms} atoms\n')
    configFile.write(f'{num_atom_types} atom types\n')
    configFile.write(f'{num_bonds} bonds\n')
    configFile.write('1 bond types\n')
    if angle_data is not None:
        configFile.write(f'{num_angles} angles\n')
        configFile.write('1 angle types\n\n')

    # Simulation box
    configFile.write('%5f   %5f  xlo xhi\n'%(required_box_bounds[0][0],required_box_bounds[0][1]))
    configFile.write('%5f   %5f  ylo yhi\n'%(required_box_bounds[1][0],required_box_bounds[1][1]))
    configFile.write('%5f   %5f  zlo zhi\n\n'%(required_box_bounds[2][0],required_box_bounds[2][1]))
    
    # Particle masses
    configFile.write('Masses\n\n')
    # First assign ids and masses for Amino acids for host chains 
    for key, value in aa_dict.items():
        configFile.write(f"{value.get('id')} {value.get('mass')}\n")
    # Second, assign ids and masses for Amino acids for cavity monomers 
    for key, value in aa_dict.items():
        configFile.write(f"{value.get('id')+20} {value.get('mass')}\n")
    # Third, assign ids and masses for Amino acids for guest chains 
    for key, value in aa_dict.items():
        configFile.write(f"{value.get('id')+40} {value.get('mass')}\n")

    # Particle positions
    configFile.write('\nAtoms\n\n')
    
    for atom_id, mol_id, atom_type, atom_charge, xcoord, ycoord, zcoord in protein_coords:
        configFile.write('%d %d %d %f %f %f %f\n' %(atom_id, mol_id, atom_type, atom_charge, xcoord, ycoord, zcoord))

    # Bond data
    configFile.write('\nBonds\n\n')

    for bond_id, bond_type, first_atom_id, second_atom_id in bond_data:
        configFile.write('%d %d %d %d\n' %(bond_id,bond_type,first_atom_id,second_atom_id))

    # Angle data
    if angle_data is not None:
        configFile.write('\nAngles\n\n')
        for angle_id, angle_type, atom1, atom2, atom3 in angle_data:
            configFile.write('%d %d %d %d %d\n' %(angle_id, angle_type, atom1, atom2, atom3))

    # Close file
    configFile.close()



# ************************************
if __name__=="__main__":
    # Read the trajectory file
    traj_file = args.traj_path
    config_file = args.config_path
    snapshot_index = args.snapshot_index

    print("Reading trajectory from:", traj_file)
    timesteps, coords, box_sizes = read_trajectory(traj_file)
    print("Reading configuration from:", config_file)
    num_atoms, num_bonds, bonds = read_config(config_file)

    print("Number of frames in trajectory:", len(timesteps))
    print("Number of atoms:", num_atoms)
    print("Number of bonds:", num_bonds)

    print("Using snapshot at index:", snapshot_index)
    write_config(f"slab_{snapshot_index}.dat", box_sizes[snapshot_index], coords[snapshot_index], num_atoms, num_bonds, bonds)