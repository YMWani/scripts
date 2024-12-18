import json
import random
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
import pathlib
import random
from generate_sphere import sphere
current_dir = pathlib.Path(__file__).resolve().parent
plt.style.use(f"{current_dir}/../plotting/ymw.mplstyle")
import random
import pandas as pd


def check_sequence_validity(sequence):
    """
    Check if the a given sequence is a valid protein sequence or not.

    Parameters:
    - sequence (str): Any protein sequence
    """
    parent_dir = Path(__file__).parent
    with open(f'{parent_dir}/amino_acid_dict.json', 'r') as f:
        amino_acid_dict = json.load(f)

    keys = list(amino_acid_dict.keys())

    if set(sequence).issubset(set(keys)):
        print("Valid sequence entered.")
    else:
        print("ERROR: Invalid sequence entered.")
        exit()


def mutate_sequence(sequence, probability):
    """
    This function takes a protein sequence and a probability,
    then it makes mutation/changes to the sequence on each element with the given probability.

    Parameters:
    - sequence (str): The protein sequence.
    - probability (float): The probability of making a change in the sequence.

    Returns:
    - mutated_sequence (str): The mutated protein sequence.
    """
    parent_dir = Path(__file__).parent
    with open(f'{parent_dir}/amino_acid_dict.json', 'r') as f:
        amino_acid_dict = json.load(f)
    keys = list(amino_acid_dict.keys())

    mutated_sequence = ""

    for amino_acid in sequence:
        # Generate a random number between 0 and 1
        random_number = random.random()

        # If the random number is less than the probability, make a change
        if random_number < probability:
            # Randomly choose an amino acid from the set of possible amino acids
            mutated_amino_acid = random.choice(keys)
            while mutated_amino_acid == amino_acid:  # Ensure that the new amino acid is different from the old one
                mutated_amino_acid = random.choice(keys)

            mutated_sequence += mutated_amino_acid
        else:
            mutated_sequence += amino_acid

    return mutated_sequence


def gen_mixture_chain_config(sequence1, sequence2, outfile="config.dat"):
    """
    This function creates a LAMMPS config file that contains two chains (sequence1 & sequence2)
    placed in straight line configuration on two opposite ends of the simulation box.

    Parameters:
    - sequence1 (str): The first protein sequence.
    - sequence2 (str): The second protein sequence.
    - aa_dict (dict): Dictionary containing data regarding the amino acids (id, charge, mass)
    - aa_mass_dict (dict): Dictionary containg masses of the different amino acids (Made to fit the
                            potentials.dat file expectations.)
    - outfile (string): file name for the output config file.
    """
    parent_dir = Path(__file__).parent
    with open(f'{parent_dir}/amino_acid_dict.json', 'r') as f:
        aa_dict = json.load(f)
    with open(f'{parent_dir}/amino_acid_masses.json', 'r') as f:
        aa_mass_dict = json.load(f)


    chain1_length = len(sequence1)
    chain2_length = len(sequence2)

    num_atoms = chain1_length + chain2_length
    num_bonds = (chain1_length - 1) + (chain2_length-1)

    num_atom_types = 20

    out = open(f'{outfile}','w')
    out.write('LAMMPS data file for IDP\n\n')
    # General parameters
    out.write(f'{num_atoms} atoms\n')
    out.write(f'{num_bonds} bonds\n\n')
    out.write(f'{num_atom_types} atom types\n')
    out.write('1 bond types\n\n')

    # Simulation box size
    min_box = max(chain1_length, chain2_length) * 0
    max_box = max(chain1_length, chain2_length) * 6
    # min_box = max(chain1_length, chain2_length) * -3
    # max_box = max(chain1_length, chain2_length) * 3

    out.write('%5f   %5f  xlo xhi\n'%(min_box,max_box))
    out.write('%5f   %5f  ylo yhi\n'%(min_box,max_box))
    out.write('%5f   %5f  zlo zhi\n\n'%(min_box,max_box))

    # Particle masses
    out.write('Masses\n\n')
    count = 1
    while count <= num_atom_types:
        out.write('   %d %f\n' %(count,float(aa_mass_dict[f"{count}"])))
        count += 1

    # Particle positions
    out.write('\nAtoms\n\n')
    # Place chains in straight lines with the distance between the monomers
    # approximately equal to the eq. bond length
    r0 = 4.0
    # Place the first chain
    mol_id = 1
    xcoord = min_box
    ycoord = min_box + r0
    zcoord = min_box + r0
    
    idx = 1
    for aa in sequence1:
        xcoord += r0
        
        atom_id = idx
        atom_type = aa_dict[f"{aa}"]["id"]
        atom_charge = aa_dict[f"{aa}"]["charge"]

        out.write('%d %d %d  %f %f  %f  %f\n' %(atom_id, mol_id, atom_type, atom_charge, xcoord, ycoord, zcoord))
        idx += 1
    
    # Place the second chain
    mol_id = 2
    xcoord = min_box
    ycoord = max_box - r0
    zcoord = max_box - r0

    for aa in sequence2:
        xcoord += r0
        
        atom_id = idx
        atom_type = aa_dict[f"{aa}"]["id"]
        atom_charge = aa_dict[f"{aa}"]["charge"]

        out.write('%d %d %d  %f %f  %f  %f\n' %(atom_id, mol_id, atom_type, atom_charge, xcoord, ycoord, zcoord))
        idx += 1
    
    # Bond data
    out.write('\nBonds\n\n')
    # Write bonds for the first chain
    bond_type = 1
    bond_id = 1
    for atom_id in range(1, chain1_length):
        out.write('%d %d %d %d\n' %(bond_id,bond_type,atom_id,atom_id+1))
        bond_id += 1
    # Write bonds for the second chain
    for atom_id in range(chain1_length+1, chain1_length+chain2_length):
        out.write('%d %d %d %d\n' %(bond_id,bond_type,atom_id,atom_id+1))
        bond_id += 1
    
    out.close()


def parse_density_profile_file(filename):
    """
    This function reads the density profile file created from LAMMPS simulations using the 
    "compute .. chunk/atom ..." command and extracts the timesteps and density profiles at
    the different time points
    
    Parameters:
    - filename (str): Name of the file containing the density profile

    Output:
    - timesteps (numpy array): Timesteps at which the density profile was measured and saved
    - densities (numpy array): Density profiles at every timestep
    """    
    with open(filename, 'r') as data:
        densities = []
        chunk_data = []
        timesteps = []
        for line in data:
            line = line.strip() # Remove leading/trailing whitespace
            parts = line.split() # Split the line to get the different components

            if line.startswith('#'):
                # Skip comment lines
                continue
            
            elif (len(parts) == 3):
                # Line with 3 components: (Timestep, Number-of-bins, Total-count)
                timesteps.append(int(parts[0]))
                if (bool(chunk_data)):
                    densities.append(chunk_data)
                    chunk_data = []
                
            else:
                # Line with 4 components: (bin, Coord1, Ncount, density/mass)
                chunk_data.append([float(parts[1]), float(parts[3])])
        densities.append(chunk_data)
    timesteps = np.array(timesteps)
    densities = np.array(densities)
    return timesteps, densities


def parse_density_profile_file_2(filename):
    """
    This function reads the density profile file created from LAMMPS simulations using the 
    "compute .. chunk/atom ..." command and extracts the timesteps and density profiles at
    the different time points
    
    Parameters:
    - filename (str): Name of the file containing the density profile

    Output:
    - timesteps (numpy array): Timesteps at which the density profile was measured and saved
    - densities (numpy array): Density profiles at every timestep
    """    
    with open(filename, 'r') as data:
        densities = []
        chunk_data = []
        timesteps = []
        for line in data:
            line = line.strip() # Remove leading/trailing whitespace
            parts = line.split() # Split the line to get the different components

            if line.startswith('#'):
                # Skip comment lines
                continue
            
            elif (len(parts) == 3):
                # Line with 3 components: (Timestep, Number-of-bins, Total-count)
                timesteps.append(int(parts[0]))
                if (bool(chunk_data)):
                    densities.append(chunk_data)
                    chunk_data = []
                
            else:
                # Line with 5 components: (bin, Coord1, Ncount, density/mass, density/number)
                chunk_data.append([float(parts[1]), float(parts[3]), float(parts[4])])
        densities.append(chunk_data)
    timesteps = np.array(timesteps)
    densities = np.array(densities)
    return timesteps, densities


def compute_averaged_density_profile(density_prof):
    """
    This function takes the raw density profile extracted from the LAMMPS density profile file
    and averages it over all the frames. In addition, the density profile is also recentered 
    such that COM of condensate lies at 0.5 (reduced units)

    Parameters:
    - density_prof (numpy array): raw density profile obtained from LAMMPS simulations.
                                  Expected shape (nframes, nbins, 2)
                                  The final 2 columns are just the bin_centers and bin_density

    Output:
    - bins: bin centers
    - bin_density: bin densities
    """
    avg_density_prof = np.mean(density_prof, axis=0) # Assuming no noise in the data, so we don't discard any part of it
    
    # Shift the density profile such that it is centered.
    # Find the center of mass of the density profile
    bins = avg_density_prof[:,0]
    bin_density = avg_density_prof[:,1]
    cm = np.sum(bins * bin_density) / np.sum(bin_density)
    # Adjust the bin positions such that the COM is at the center of the box (i.e. 0.5)
    bins -= cm - 0.5
    bins[bins < 0] += 1.0
    
    return bins, bin_density


def compute_averaged_density_profile_2(density_prof):
    """
    This function takes the raw density profile extracted from the LAMMPS density profile file
    and averages it over all the frames. In addition, the density profile is also recentered 
    such that COM of condensate lies at 0.5 (reduced units)

    Parameters:
    - density_prof (numpy array): raw density profile obtained from LAMMPS simulations.
                                  Expected shape (nframes, nbins, 2)
                                  The final 2 columns are just the bin_centers and bin_density

    Output:
    - bins: bin centers
    - bin_density (mass): bin densities
    - bin_density (number): bin densities
    """
    avg_density_prof = np.mean(density_prof, axis=0) # Assuming no noise in the data, so we don't discard any part of it
    
    # Shift the density profile such that it is centered.
    # Find the center of mass of the density profile
    bins = avg_density_prof[:,0]
    bin_density_mass = avg_density_prof[:,1]
    bin_density_number = avg_density_prof[:,2]
    cm = np.sum(bins * bin_density_mass) / np.sum(bin_density_mass)
    # Adjust the bin positions such that the COM is at the center of the box (i.e. 0.5)
    bins -= cm - 0.5
    bins[bins < 0] += 1.0
    
    return bins, bin_density_mass, bin_density_number


def find_interfaces(coords, avg_profile, derivative_threshold=1e-5, percentile_low=15, percentile_high=85, min_dilute_size_multiplier=0.3):
    """
    This function takes averaged density profile for a direct coexistence simulations
    and fits a super gaussian function to the profile. From the second derivative of the 
    super gaussian one can extract the interface coordinates of the slab.

    Parameters:
    - coords (array): bin_centers of the density profile
    - avg_profile (array): bin_densities of the density profile

    Output:
    - left_interface_coord
    - right_interface_coord
    - fine_coords: finely spaced bin centers
    - fitted_profile: Fitted profile on the finely spaced bin centers
    """
    # fit super gaussian
    super_gaussian = lambda x, A, x0, sigma, p: A*np.exp(-2*((x-x0)/sigma)**(2*np.round(p)))
    initial_guess = [np.percentile(avg_profile, 80), np.mean(coords), np.std(coords), 2]
    popt, pcov = curve_fit(super_gaussian, coords, avg_profile, p0=initial_guess, maxfev=10000)
    A, x0, sigma, p = popt
    P = 2*np.round(p)
    fine_coords = np.linspace(min(coords), max(coords), num=1000)
    fitted_profile = super_gaussian(fine_coords, *popt)
    # compute second derivative
    first_derivative = np.gradient(fitted_profile)
    second_derivative = np.gradient(first_derivative)
    # binarize normalized second derivatives with derivative threshold and get interfaces
    threshold = derivative_threshold # units: (density per length per length) / length
    binarized = [abs(x)/max(fitted_profile) <= threshold for x in second_derivative]
    num_clusters = 1
    ## keep track of "True" cluster boundaries for plotting
    true_cluster_boundaries = []
    curr_cluster = [0] if binarized[0] else []
    for idx in range(1, len(binarized)):
        if binarized[idx] != binarized[idx-1]:
            num_clusters += 1
            if binarized[idx]:
                curr_cluster.append(fine_coords[idx])
            else:
                curr_cluster.append(fine_coords[idx-1])
                true_cluster_boundaries.append(curr_cluster)
                curr_cluster = []
    if len(curr_cluster) == 1: # ends on True cluster that doesn't get closed in loop, which will be common
        curr_cluster.append(fine_coords[-1])
        true_cluster_boundaries.append(curr_cluster)
    num_clusters += 1
    num_clusters /= 2
    if binarized[0]==True and binarized[-1]==True and num_clusters == 5: # clean condensate; zero second derivative on ends, slope up/down and the middle
        left_interface_coord = fine_coords[round(([idx for idx in range(1, len(binarized)) if binarized[idx] == False and binarized[idx-1] == True][1] + [idx for idx in range(1, len(binarized)) if binarized[idx] == True and binarized[idx-1] == False][0])/2)]
        right_interface_coord = fine_coords[round(([idx for idx in list(reversed(range(0, len(binarized)-2))) if binarized[idx] == False and binarized[idx+1] == True][1] + [idx for idx in list(reversed(range(0, len(binarized)-2))) if binarized[idx] == True and binarized[idx+1] == False][0])/2)]
        if (left_interface_coord + (max(fine_coords)-right_interface_coord)) <= min_dilute_size_multiplier * right_interface_coord - left_interface_coord: # ensure dilute phase isn't too small
            left_interface_coord, right_interface_coord = np.percentile(coords, percentile_low), np.percentile(coords, percentile_high)
    else:
        left_interface_coord, right_interface_coord = np.percentile(coords, percentile_low), np.percentile(coords, percentile_high)
    
    return left_interface_coord, right_interface_coord, fine_coords, fitted_profile


def evaluate_fitness(rhoA_center, rhoB_center, rhoA_vapour, rhoB_vapour, s):
    """
    Evaluate fitness of a multiphasic condensate.

    Parameters:
    - rhoA_center (float): density of protein A in the center of the condensate (reduced units)
    - rhoB_center (float): density of protein B in the center of the condensate (reduced units)
    - rhoA_vapour (float): density of protein A in the vapour phase (reduced units)
    - rhoB_vapour (float): density of protein B in the vapour phase (reduced units)
    - s (float): Weighting parameter

    Output:
    - fitness (float)
    """
    return abs(rhoA_center - rhoB_center) - s*(rhoA_vapour + rhoB_vapour)


def generate_children(N_tour, num_parents, population):
    """
    Function to create children from a set of population. Here we use the tournament
    algorithm to make children.

    Parameters:
    - N_tour (int): Sample size of each tournament round
    - num_parents (int): Number of parents to extract from the provided population (can repeat)
    - population (array): Array containing information about the fittest members of the initial
                          population. Information: (sequence, fitness)
    """
    sequences = np.array(population[:,0]).astype(str)
    fitness_vals = np.array(population[:,1]).astype(float)
    
    parents = []
    N_tour = 5
    num_seq = sequences.shape[0]
    seq_length = len(sequences[0])
    
    for i in range(num_parents):
        # Randomly pick 5 parents from the population
        random_sample = random.sample(range(num_seq), N_tour)
        sample_fitness = fitness_vals[random_sample]
        fittest_pick = max(sample_fitness)
        idx = fitness_vals.tolist().index(fittest_pick)
        parents.append(sequences[idx])
    
    np.random.shuffle(parents)
    pairs = [(parents[j], parents[j+1]) for j in range(0, len(parents)-1, 2)]
    
    children = {}
    counter = 1
    for a,b in pairs:
        # Pick a random point in the sequences for a cross-over
        k = np.random.randint(seq_length) + 1
        # Child 1
        child = a[:k] + b[k:]
        child_mutated = mutate_sequence(child, 0.05)
        children[f'C{counter}'] = child_mutated
        counter += 1
        # Child 2
        child = b[:k] + a[k:]
        child_mutated = mutate_sequence(child, 0.05)
        children[f'C{counter}'] = child_mutated
        counter += 1
    
    return children


def compute_SCD(seq):
    """
    Compute sequence charge decoration parameter for a given sequence
    The equation can be found in this paper: https://doi.org/10.1063/1.4929391 eq.(14)
    """
    N = len(seq) # Length of protein
    
    # Load amino acid dictionary
    parent_dir = Path(__file__).parent
    with open(f'{parent_dir}/amino_acid_dict.json', 'r') as f:
        amino_acid_dict = json.load(f)

    SCD = 0.
    for m in range(1,N):
        qm = amino_acid_dict[seq[m]]["charge"]
        for n in range(m):
            qn = amino_acid_dict[seq[n]]["charge"]
            SCD += qm * qn * (m-n)**0.5
    SCD /= N
    
    return SCD


def extract_sequence_composition(seq):
    """
    Extract the AA coposition of a given sequence. The properties extracted are:
    1. Glycine
    2. Negative
    3. Neutral
    4. Neutral with pi electrons
    5. Positive
    6. Positive with pi electrons
    7. Aromatic
    """
    Nm = len(seq)
    glycine = 0.
    negative = 0.
    neutral = 0.
    neutral_with_pi = 0.
    positive = 0.
    positive_with_pi = 0.
    aromatic = 0.
    for char in seq:
        if char == "G":
            glycine += 1
        elif char in ["D", "E"]:
            negative += 1
        elif char in ["A", "C", "I", "L", "M", "P", "S", "T", "V"]:
            neutral += 1
        elif char in ["N", "Q"]:
            neutral_with_pi += 1
        elif char == "K":
            positive += 1
        elif char in ["H", "R"]:
            positive_with_pi += 1
        elif char in ["F", "W", "Y"]:
            aromatic += 1
        else:
            print("Amino acid not found!!")
    return {"glycine": glycine/Nm,
            "negative": negative/Nm, 
            "neutral": neutral/Nm, 
            "neutral_with_pi": neutral_with_pi/Nm, 
            "positive": positive/Nm, 
            "positive_with_pi": positive_with_pi/Nm, 
            "aromatic": aromatic/Nm}


def read_trajectory(file_name):
    '''
    Reads in the unwrapped coordinates from the last frame of a LAMMPS trajectory file.
    Returns a tuple with two arrays:
    1. Box bounds as a numpy array with the format:
        [[x_min, x_max], [y_min, y_max], [z_min, z_max]]
    2. Atom data as a numpy array with tuples of the format:
        (atom number, molecule ID, type, charge, x, y, z)
    3. The highest molecule ID
    '''
    coords = []
    box_bounds = []
    last_timestep_index = None
    with open(file_name, 'r') as f:
        lines = f.readlines()

    # Find the last occurrence of "ITEM: TIMESTEP" which indicates the start of the last frame
    for i in range(len(lines)):
        if "ITEM: TIMESTEP" in lines[i]:
            last_timestep_index = i

    if last_timestep_index is None:
        raise ValueError("No timestep found in the file.")

    # Read the box bounds from the last frame
    box_start_index = last_timestep_index + 5  # Skipping TIMESTEP, NUMBER OF ATOMS, and BOX BOUNDS headers
    for i in range(box_start_index, box_start_index + 3):
        bounds = [float(val) for val in lines[i].strip().split()]
        box_bounds.append(bounds)

    # Read atom data from the last frame
    atom_data_start = box_start_index + 3  # Start of the atom data
    reading_atoms = False
    highest_mol_id = 0

    for line in lines[atom_data_start:]:
        if "ITEM: ATOMS" in line:
            reading_atoms = True
            continue
        elif "ITEM:" in line:
            break  # Stop reading when another section starts
        elif reading_atoms:
            atom_data = line.strip().split()
            atom_number = int(atom_data[0])
            mol_id = int(atom_data[1])
            atom_type = int(atom_data[2])
            charge = float(atom_data[3])
            x = float(atom_data[4])
            y = float(atom_data[5])
            z = float(atom_data[6])

            # Append the tuple to the list
            coords.append((atom_number, mol_id, atom_type, charge, x, y, z))

            if mol_id > highest_mol_id:
                highest_mol_id = mol_id

    return box_bounds, coords, highest_mol_id

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
                bonds.append((bond_id, bond_type, first_atom_id, second_atom_id))

    return num_atoms, num_bonds, bonds

def write_slab_config(required_box_bounds,
                      protein_coords, 
                      num_atoms, 
                      num_bonds, 
                      bond_data):
    """
    Create a config file for a slab simulation given the follwing information:
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
    
    num_atom_types = 20

    configFile = open('initialSlab.dat','w')
    configFile.write('LAMMPS data file for slab of IDP\n\n')

    # Overall data
    configFile.write(f'{num_atoms} atoms\n')
    configFile.write(f'{num_atom_types} atom types\n')
    configFile.write(f'{num_bonds} bonds\n')
    configFile.write('1 bond types\n\n')

    # Simulation box
    configFile.write('%5f   %5f  xlo xhi\n'%(required_box_bounds[0][0],required_box_bounds[0][1]))
    configFile.write('%5f   %5f  ylo yhi\n'%(required_box_bounds[1][0],required_box_bounds[1][1]))
    configFile.write('%5f   %5f  zlo zhi\n\n'%(required_box_bounds[2][0],required_box_bounds[2][1]))
    
    # Particle masses
    configFile.write('Masses\n\n')
    for key, value in aa_dict.items():
        configFile.write(f"{value.get('id')} {value.get('mass')}\n")

    # Particle positions
    configFile.write('\nAtoms\n\n')
    
    for atom_id, mol_id, atom_type, atom_charge, xcoord, ycoord, zcoord in protein_coords:
        configFile.write('%d %d %d  %f %f  %f  %f\n' %(atom_id, mol_id, atom_type, atom_charge, xcoord, ycoord, zcoord))

    # Bond data
    configFile.write('\nBonds\n\n')

    for bond_id, bond_type, first_atom_id, second_atom_id in bond_data:
        configFile.write('%d %d %d %d\n' %(bond_id,bond_type,first_atom_id,second_atom_id))
    
    # Close file
    configFile.close()    


def gen_seq_based_cavity(seq, diameter, subDiv=4):
    """
    construct a spherical cavity of a given diameter based on the discrete particle model such that
    the surface monomers have similar composition to the given sequence.

    Arguments:
    - seq (str): Protein sequence whose composition we want to replicate on the cavity
    - diameter (float): Cavity diameter in Angstroms.

    Output:
    - positions (float array): Array containing the positions of the surface monomers
    - types (str list): List containing the atom types of the surface monomers
    """
    # First create a discrete particle model sphere 
    shape = sphere(diameter/2., subDiv)
    
    # Extract composition of the entered sequence
    composition = extract_sequence_composition(seq)
    
    # Assign atom types to the surface monomers
    category_dict = {
        "glycine" : ["G"],
        "negative" : ["D", "E"],
        "neutral" : ["A", "C", "I", "L", "M", "P", "S", "T", "V"],
        "neutral_with_pi" : ["N", "Q"],
        "positive" : ["K"],
        "positive_with_pi" : ["H", "R"],
        "aromatic" : ["F", "W", "Y"]
    }
    atom_types = []
    gen_seq = ""
    for _ in range(shape.verts.shape[0]):
        # Choose a random category based on its probability
        category = random.choices(list(composition.keys()), weights=list(composition.values()))[0]
        
        # Select an amino acid randomly from the chosen category
        amino_acid = random.choice(category_dict[category])

        gen_seq += amino_acid
        atom_types.append(amino_acid)
    # Composition of the generated sequence (if needed)
    composition_gen_seq = extract_sequence_composition(gen_seq)

    # We exclude the last element since the particle is at the center of the cavity
    return shape.verts[:-1], atom_types[:-1]


def write_cavity_with_protein1_config(simBox, cavity_positions, cavity_types, protein_coords=None,
                        protein_bond_data=None, num_atom_types = 40):
    """
    Creates a config file for a slab containing a cavity and protein chains (None by default)

    Arguments:
    - simBox (float list): List of the lower and upper bounds for the simulation box
    - cavity_positions (float array): Positions of the monomers of the spherical cavity
    - cavity_types (char list): List of the atom_types of the surface monomers.
    - protein_coords (tuple list): A list of tuples containing details for each atom of the protein chains
                                  (atom_id, mol_id, atom_type, atom_charge, xcoord, ycoord, zcoord)
    - protein_bond_data (list of tuples): Bond data for the protein chains
                                         (bond_id, bond_type, first_atom_id, second_atom_id)

    Returns: config file for LAMMPS simulation - "cavity_protein1_config.dat"
    """
    parent_dir = Path(__file__).parent
    with open(f'{parent_dir}/amino_acid_dict.json', 'r') as f:
        aa_dict = json.load(f)
    
    configFile = open('cavity_protein1_config.dat','w')
    configFile.write('LAMMPS data file with spherical cavity to hold RNA chains\n\n')

    # Overall data
    if protein_coords == None:
        num_atoms = cavity_positions.shape[0]
    else:
        num_atoms = cavity_positions.shape[0] + len(protein_coords)
    configFile.write(f'{num_atoms} atoms\n')
    configFile.write(f'{num_atom_types} atom types\n')
    
    if protein_bond_data == None:
        num_bonds = 0
    else:
        num_bonds = len(protein_bond_data)    
    configFile.write(f'{num_bonds} bonds\n')
    configFile.write('1 bond types\n\n')

    # Simulation box
    configFile.write('%5f   %5f  xlo xhi\n'%(simBox[0][0],simBox[0][1]))
    configFile.write('%5f   %5f  ylo yhi\n'%(simBox[1][0],simBox[1][1]))
    configFile.write('%5f   %5f  zlo zhi\n\n'%(simBox[2][0],simBox[2][1]))
    
    # Particle masses
    configFile.write('Masses\n\n')
    # First assign ids and masses for Amino acids for protein chains 
    for key, value in aa_dict.items():
        configFile.write(f"{value.get('id')} {value.get('mass')}\n")
    # Second, assign ids and masses for Amino acids for cavity monomers 
    for key, value in aa_dict.items():
        configFile.write(f"{value.get('id')+20} {value.get('mass')}\n")

    # Particle positions
    configFile.write('\nAtoms\n\n')

    # First assign positions of the cavity
    mol_id = 1 # Let's assign the cavity as mol_id = 1
    for index, pos in enumerate(cavity_positions):
        atom_id = index + 1
        atom_type = aa_dict[cavity_types[index]]["id"] + 20 # Add 20 for cavity monomers
        atom_charge = aa_dict[cavity_types[index]]["charge"]
        # By construction, the cavity positions are such that the center of the cavity is at (0.,0.,0.)
        # We need to recenter it to the center of the new simulation box
        xcoord = pos[0] + (simBox[0][0] + (simBox[0][1] - simBox[0][0])/2.)
        ycoord = pos[1] + (simBox[1][0] + (simBox[1][1] - simBox[1][0])/2.)
        zcoord = pos[2] + (simBox[2][0] + (simBox[2][1] - simBox[2][0])/2.)
        configFile.write('%d %d %d  %f %f  %f  %f\n' %(atom_id, mol_id, atom_type, atom_charge, xcoord, ycoord, zcoord))
    
    # If protein positions are also given, then assign the positions of the proteins as well
    if protein_coords:
        # First we need to ensure that the proteins positions are not inside the cavity
        # Protein coords are read from trajectory file (unwrapped coordinates)
        # Therefore, we must first wrap the coordinates in the simulation box
        wrapped_zcoords = []
        # Lx = simBox[0][1] - simBox[0][0]
        # Ly = simBox[1][1] - simBox[1][0]
        Lz = simBox[2][1] - simBox[2][0]
        for atom_id, mol_id, atom_type, atom_charge, xcoord, ycoord, zcoord in protein_coords:
            zw = zcoord - (zcoord//Lz)*Lz
            wrapped_zcoords.append(zw)
        wrapped_zcoords = np.array(wrapped_zcoords)
        # Calculate the shift to place the protein condensate on the right side of cavity
        zw_min = np.min(wrapped_zcoords)
        zc_max = np.max(cavity_positions[:,2]) + Lz/2.
        z_buff = 1.0
        delta_z = zc_max + z_buff - zw_min # We add this quantity to unwrapped zcoord

        # Since Cavity is already placed we need to adjust atom_id and mol_id accordingly
        num_cavity_atoms = cavity_positions.shape[0]
        for atom_id, mol_id, atom_type, atom_charge, xcoord, ycoord, zcoord in protein_coords:
            configFile.write('%d %d %d  %f %f  %f  %f\n' %(atom_id+num_cavity_atoms, mol_id+1, atom_type, atom_charge, xcoord, ycoord, zcoord+delta_z))
    
    # Bond data
    # Only the proteins will have bonds
    if protein_bond_data:
        configFile.write('\nBonds\n\n')
        # Similar to the positions, we need to adjust atom_id here.
        # No need to worry about bond_id since cavity monomers are not bonded
        for bond_id, bond_type, first_atom_id, second_atom_id in protein_bond_data:
            configFile.write('%d %d %d %d\n' %(bond_id,bond_type,first_atom_id+num_cavity_atoms,second_atom_id+num_cavity_atoms))
    
    # Close file
    configFile.close()


def write_cavity_with_protein1and2_config(simBox,
                                          cavity_inner_prot_coords, cavity_inner_protein_bond_data, cavity_inner_protein_highest_molid,
                                          outer_protein_coords, outer_protein_bond_data,
                                          num_atom_types = 40):
    """
    Creates a config file for a slab containing a cavity and protein chains

    Arguments:
    - simBox (float list): List of the lower and upper bounds for the simulation box
    - cavity_inner_prot_coords (tuple list): A list of tuples containing details for
                                    each atom of the cavity and inner protein chains
                    (atom_id, mol_id, atom_type, atom_charge, xcoord, ycoord, zcoord)
    - cavity_inner_protein_bond_data (list of tuples): Bond data for the inner protein
                                    chains
                                    (bond_id, bond_type, first_atom_id, second_atom_id)
    - outer_protein_coords (tuple list): A list of tuples containing details for each
                                    atom of the outer protein chains
    - outer_protein_bond_data (list of tuples): Bond data for the outer protein chains
                                    (bond_id, bond_type, first_atom_id, second_atom_id)
    
    Returns: config file for LAMMPS simulation - "cavity_protein1_protein2_config.dat"
    """
    parent_dir = Path(__file__).parent
    with open(f'{parent_dir}/amino_acid_dict.json', 'r') as f:
        aa_dict = json.load(f)

    configFile = open('cavity_protein1_protein2_config.dat','w')
    configFile.write('LAMMPS data file with spherical cavity to hold RNA chains\n\n')

    # Overall data
    num_atoms = len(cavity_inner_prot_coords) + 2*len(outer_protein_coords)
    configFile.write(f'{num_atoms} atoms\n')
    configFile.write(f'{num_atom_types} atom types\n')
    
    num_bonds = len(cavity_inner_protein_bond_data) + 2*len(outer_protein_bond_data)
    configFile.write(f'{num_bonds} bonds\n')
    configFile.write('1 bond types\n\n')

    # Simulation box
    configFile.write('%5f   %5f  xlo xhi\n'%(simBox[0][0],simBox[0][1]))
    configFile.write('%5f   %5f  ylo yhi\n'%(simBox[1][0],simBox[1][1]))
    configFile.write('%5f   %5f  zlo zhi\n\n'%(simBox[2][0],simBox[2][1]))
    
    # Particle masses
    configFile.write('Masses\n\n')
    # First assign ids and masses for Amino acids for protein chains 
    for key, value in aa_dict.items():
        configFile.write(f"{value.get('id')} {value.get('mass')}\n")
    # Second, assign ids and masses for Amino acids for cavity monomers 
    for key, value in aa_dict.items():
        configFile.write(f"{value.get('id')+20} {value.get('mass')}\n")

    # Particle positions
    configFile.write('\nAtoms\n\n')

    # First assign positions of cavity and inner protein (already equilibrated around the cavity)
    # We just paste the information as is since it has already been equilibrated
    for atom_id, mol_id, atom_type, atom_charge, xcoord, ycoord, zcoord in cavity_inner_prot_coords:
        configFile.write('%d %d %d  %f %f  %f  %f\n' %(atom_id, mol_id, atom_type, atom_charge, xcoord, ycoord, zcoord))

    # Now, we place the outer protein around the cavity and inner proteins
    wrapped_zcoords = []
    Lz = simBox[2][1] - simBox[2][0]
    for atom_id, mol_id, atom_type, atom_charge, xcoord, ycoord, zcoord in outer_protein_coords:
        zw = zcoord - (zcoord//Lz)*Lz
        wrapped_zcoords.append(zw)
    wrapped_zcoords = np.array(wrapped_zcoords)
    
    # Calculate the shift to place the outer protein condensate on the right side of cavity & inner protein
    zw_min = np.min(wrapped_zcoords) # minimum z position of the outer condensate
    zci_max = np.max([t[6] for t in cavity_inner_prot_coords]) # maximum z position of cavity & inner protein cond.
    z_buff = 1.0
    delta_z = zci_max + z_buff - zw_min # We add this quantity to unwrapped zcoord
    # Since Cavity and inner proteins are already placed we need to adjust atom_id and mol_id accordingly
    num_previous_atoms = len(cavity_inner_prot_coords)
    for atom_id, mol_id, atom_type, atom_charge, xcoord, ycoord, zcoord in outer_protein_coords:
        configFile.write('%d %d %d  %f %f  %f  %f\n' %(atom_id+num_previous_atoms, mol_id+cavity_inner_protein_highest_molid,
                                                       atom_type, atom_charge, xcoord, ycoord, zcoord+delta_z))
        last_atom_id = atom_id+num_previous_atoms
        last_mol_id = mol_id+cavity_inner_protein_highest_molid
    
    # Calculate the shift to place the outer protein condensate on the left side of the cavity & inner protein
    zw_max = np.max(wrapped_zcoords) # maximum z position of the outer condensate
    zci_min = np.min([t[6] for t in cavity_inner_prot_coords]) # minimum z position of cavity & inner protein cond.
    delta_z = zci_min - zw_max - z_buff # We add this quantity to unwrapped zcoord
    for atom_id, mol_id, atom_type, atom_charge, xcoord, ycoord, zcoord in outer_protein_coords:
        configFile.write('%d %d %d  %f %f  %f  %f\n' %(atom_id+last_atom_id, mol_id+last_mol_id, atom_type,
                                                       atom_charge, xcoord, ycoord, zcoord+delta_z))


    # Bond data
    configFile.write('\nBonds\n\n')
    # Inner proteins: Bond data already accounts for the cavity monomers
    for bond_id, bond_type, first_atom_id, second_atom_id in cavity_inner_protein_bond_data:
        configFile.write('%d %d %d %d\n' %(bond_id,bond_type,first_atom_id,second_atom_id))
    # Outer proteins: Atom ids need to be ajusted for the cavity & inner protein monomers
    # Right side
    atom_id_counter = len(cavity_inner_prot_coords)
    bond_id_counter = len(cavity_inner_protein_bond_data)
    for bond_id, bond_type, first_atom_id, second_atom_id in outer_protein_bond_data:
        configFile.write('%d %d %d %d\n' %(bond_id+bond_id_counter,bond_type,
                                           first_atom_id+atom_id_counter,
                                           second_atom_id+atom_id_counter))
    # Left side
    atom_id_counter += len(outer_protein_coords)
    bond_id_counter += len(outer_protein_coords)
    for bond_id, bond_type, first_atom_id, second_atom_id in outer_protein_bond_data:
        configFile.write('%d %d %d %d\n' %(bond_id+bond_id_counter,bond_type,
                                           first_atom_id+atom_id_counter,
                                           second_atom_id+atom_id_counter))    
    
    # Close file
    configFile.close()


def write_single_chain_config(sequence, outfile="config.dat"):
    """
    Create a single chain LAMMPS config file for a gievn sequence

    Arguments:
        - sequence (str): single character AA seq.
    """
    parent_dir = Path(__file__).parent
    with open(f'{parent_dir}/amino_acid_dict.json', 'r') as f:
        aa_dict = json.load(f)
    
    chain_length = len(sequence)

    num_atoms = chain_length
    num_bonds = chain_length - 1

    num_atom_types = 20

    out = open(f'{outfile}','w')
    out.write('LAMMPS data file for IDP\n\n')
    # General parameters
    out.write(f'{num_atoms} atoms\n')
    out.write(f'{num_bonds} bonds\n\n')
    out.write(f'{num_atom_types} atom types\n')
    out.write('1 bond types\n\n')

    # Simulation box size
    min_box = 0.0
    max_box = chain_length * 6.0

    out.write('%5f   %5f  xlo xhi\n'%(min_box,max_box))
    out.write('%5f   %5f  ylo yhi\n'%(min_box,max_box))
    out.write('%5f   %5f  zlo zhi\n\n'%(min_box,max_box))

    # Particle masses
    out.write('Masses\n\n')
    for key, value in aa_dict.items():
        out.write(f"{value.get('id')} {value.get('mass')}\n")
    
    # Particle positions
    out.write('\nAtoms\n\n')
    # Place chain in straight line with the distance between the monomers
    # approximately equal to the eq. bond length
    r0 = 4.0

    mol_id = 1
    xcoord = min_box
    ycoord = min_box + r0
    zcoord = min_box + r0
    
    idx = 1
    for aa in sequence:
        xcoord += r0
        
        atom_id = idx
        atom_type = aa_dict[f"{aa}"]["id"]
        atom_charge = aa_dict[f"{aa}"]["charge"]

        out.write('%d %d %d  %f %f  %f  %f\n' %(atom_id, mol_id, atom_type, atom_charge, xcoord, ycoord, zcoord))
        idx += 1
    
    # Bond data
    out.write('\nBonds\n\n')
    bond_type = 1
    bond_id = 1
    for atom_id in range(1, chain_length):
        out.write('%d %d %d %d\n' %(bond_id,bond_type,atom_id,atom_id+1))
        bond_id += 1
    
    out.close()


def generate_cuboid_cavity(seq, xw, yw, zw):
    """
    Construct a cuboidal shaped cavity where the surfaces along the xy plane are decorated with surface
    monomers that have a similar composition as the given protein sequence. 
    
    Cavity is centered at origin

    Arguments:
    - seq (str): Protein sequence whose composition we want to replicate on the cavity
    - xw, yw, zw (float): Width of the cavity in x, y, and z direction, respectively.

    Output:
    - positions (float array): Array containing the positions of the surface monomers
    - types (str list): List containing the atom types of the surface monomers
    """
    # 1. Create cavity surface
    r0 = 3.81 # Equilibrium bond distance for protein chains
    nx = int(xw/r0) # #segments along x direction
    rx = xw/nx # distance between monomers in x direction
    ny = int(yw/r0) # #segments along y direction
    ry = yw/ny # distance between monomers in x direction

    x_vals = np.linspace(0, xw, nx+1) - xw/2.
    y_vals = np.linspace(0, yw, ny+1) - yw/2.
    z_vals = np.repeat(zw/2., (nx+1)*(ny+1))
    xy_pos = np.array(np.meshgrid(x_vals, y_vals)).T.reshape(-1,2)
    pos1 = np.column_stack((xy_pos, z_vals)) # Positions for surface z=zw/2.
    z_vals = np.repeat(-zw/2., (nx+1)*(ny+1))
    pos2 = np.column_stack((xy_pos, z_vals)) # Positions for surface z=-zw/2.
    positions = np.concatenate((pos1, pos2)) # Surface monomer positions for both the required faces of cavity

    # 2. Extract composition of the entered sequence
    composition = extract_sequence_composition(seq)
    
    # 3. Assign types for the surface monomers
    # Assign atom types to the surface monomers
    category_dict = {
        "glycine" : ["G"],
        "negative" : ["D", "E"],
        "neutral" : ["A", "C", "I", "L", "M", "P", "S", "T", "V"],
        "neutral_with_pi" : ["N", "Q"],
        "positive" : ["K"],
        "positive_with_pi" : ["H", "R"],
        "aromatic" : ["F", "W", "Y"]
    }
    atom_types = []
    gen_seq = ""
    for _ in range(positions.shape[0]):
        # Choose a random category based on its probability
        category = random.choices(list(composition.keys()), weights=list(composition.values()))[0]
        
        # Select an amino acid randomly from the chosen category
        amino_acid = random.choice(category_dict[category])

        gen_seq += amino_acid
        atom_types.append(amino_acid)
    # Composition of the generated sequence (if needed)
    composition_gen_seq = extract_sequence_composition(gen_seq)
    
    return positions, atom_types

def write_config_cuboid_cavity_with_inner_prot(simBox, cavity_positions, cavity_types,
                                               protein_coords, protein_bond_data,
                                               num_atom_types = 40):
    """
    Creates a config file for a slab containing a cuboid shaped cavity at the centre of the
    slab that is surrounded by given protein chains.

    Arguments:
    - simBox (float list): List of the lower and upper bounds for the simulation box
    - cavity_positions (float array): Positions of the monomers of the spherical cavity
    - cavity_types (char list): List of the atom_types of the surface monomers.
    - protein_coords (tuple list): A list of tuples containing details for each atom of the protein chains
                                  (atom_id, mol_id, atom_type, atom_charge, xcoord, ycoord, zcoord)
    - protein_bond_data (list of tuples): Bond data for the protein chains
                                         (bond_id, bond_type, first_atom_id, second_atom_id)

    Returns: config file for LAMMPS simulation - "config_cuboid_cavity_with_inner_prot.dat"
    """
    parent_dir = Path(__file__).parent
    with open(f'{parent_dir}/amino_acid_dict.json', 'r') as f:
        aa_dict = json.load(f)
    
    configFile = open('config_cuboid_cavity_with_inner_prot.dat','w')
    configFile.write('LAMMPS data file with cuboidal cavity to hold RNA chains\n\n')

    # Overall data
    num_atoms = cavity_positions.shape[0] + 2*len(protein_coords)
    configFile.write(f'{num_atoms} atoms\n')
    configFile.write(f'{num_atom_types} atom types\n')
    
    num_bonds = 2*len(protein_bond_data)
    configFile.write(f'{num_bonds} bonds\n')
    configFile.write('1 bond types\n\n')

    # Simulation box
    configFile.write('%5f   %5f  xlo xhi\n'%(simBox[0][0],simBox[0][1]))
    configFile.write('%5f   %5f  ylo yhi\n'%(simBox[1][0],simBox[1][1]))
    configFile.write('%5f   %5f  zlo zhi\n\n'%(simBox[2][0],simBox[2][1]))
    
    # Particle masses
    configFile.write('Masses\n\n')
    # First assign ids and masses for Amino acids for protein chains 
    for key, value in aa_dict.items():
        configFile.write(f"{value.get('id')} {value.get('mass')}\n")
    # Second, assign ids and masses for Amino acids for cavity monomers 
    for key, value in aa_dict.items():
        configFile.write(f"{value.get('id')+20} {value.get('mass')}\n")

    # Particle positions
    configFile.write('\nAtoms\n\n')

    # First assign positions of the cavity
    mol_id = 1 # Let's assign the cavity as mol_id = 1
    for index, pos in enumerate(cavity_positions):
        atom_id = index + 1
        atom_type = aa_dict[cavity_types[index]]["id"] + 20 # Add 20 for cavity monomers
        atom_charge = aa_dict[cavity_types[index]]["charge"]
        # By construction, the cavity positions are such that the center of the cavity is at (0.,0.,0.)
        # We need to recenter it to the center of the new simulation box
        xcoord = pos[0] + (simBox[0][0] + (simBox[0][1] - simBox[0][0])/2.)
        ycoord = pos[1] + (simBox[1][0] + (simBox[1][1] - simBox[1][0])/2.)
        zcoord = pos[2] + (simBox[2][0] + (simBox[2][1] - simBox[2][0])/2.)
        configFile.write('%d %d %d  %f %f  %f  %f\n' %(atom_id, mol_id, atom_type, atom_charge, xcoord, ycoord, zcoord))
    
    # Second, assign protein positions
    # We need to add proteins on both sides of the cavity.
    # ****************   1. Right side    ************************
    wrapped_zcoords = []
    Lz = simBox[2][1] - simBox[2][0]
    for atom_id, mol_id, atom_type, atom_charge, xcoord, ycoord, zcoord in protein_coords:
        zw = zcoord - (zcoord//Lz)*Lz
        wrapped_zcoords.append(zw)
    wrapped_zcoords = np.array(wrapped_zcoords)

    # There is a chance that the wrapped coordinates of the protein chains cross the PB 
    # along z direction. For the next logic to work we need them not to cross the PB
    tmp = 0.
    while (np.min(wrapped_zcoords) < simBox[2][0]+0.1*Lz or np.max(wrapped_zcoords) > simBox[2][1]-0.1*Lz):
        tmp += 50.
        wrapped_zcoords -= 50.0
        wrapped_zcoords -= (wrapped_zcoords//Lz)*Lz
    
    # Calculate the shift to place the protein condensate on the right side of cavity
    zw_min = np.min(wrapped_zcoords)
    zc_max = np.max(cavity_positions[:,2]) + Lz/2. # Since cavity is originally centered at origin
    z_buff = 2.5
    delta_z = zc_max + z_buff - zw_min - tmp # We add this quantity to unwrapped zcoord

    # Since Cavity is already placed we need to adjust atom_id and mol_id accordingly
    num_cavity_atoms = cavity_positions.shape[0]
    for atom_id, mol_id, atom_type, atom_charge, xcoord, ycoord, zcoord in protein_coords:
        configFile.write('%d %d %d  %f %f  %f  %f\n' %(atom_id+num_cavity_atoms, mol_id+1, atom_type, 
                                                       atom_charge, xcoord, ycoord, zcoord+delta_z))

    # ****************   2. Left side    ************************
    # Calculate the shift to place the protein condensate on the left side of cavity
    zw_max = np.max(wrapped_zcoords)
    zc_min = np.min(cavity_positions[:,2]) - Lz/2. # Cavity originally centered at origin
    z_buff = 5.0
    delta_z = zc_min - z_buff - zw_max - tmp

    # Since Cavity and proteins(right side) are already placed we need to adjust atom_id and mol_id accordingly
    num_atoms_already_placed = cavity_positions.shape[0] + len(protein_coords)
    last_mol_id = 1 + protein_coords[-1][1]
    for atom_id, mol_id, atom_type, atom_charge, xcoord, ycoord, zcoord in protein_coords:
        configFile.write('%d %d %d  %f %f  %f  %f\n' %(atom_id+num_atoms_already_placed, mol_id+last_mol_id, atom_type, 
                                                       atom_charge, xcoord, ycoord, zcoord+delta_z))


    # BOND DATA
    # Again, we have two sets of bond data (right side and left side)
    configFile.write('\nBonds\n\n')
    #  ***************   1. Right side    *********************
    # We need to account for the atom ids
    # Note: Cavity does not have any bonds
    for bond_id, bond_type, first_atom_id, second_atom_id in protein_bond_data:
        configFile.write('%d %d %d %d\n' %(bond_id, bond_type,
                                           first_atom_id+num_cavity_atoms,second_atom_id+num_cavity_atoms))
    #  ***************   2. Left side    *********************
    # We need to account for the atom ids and bond ids
    last_bond_id = max(t[0] for t in protein_bond_data) # bond data is jumbled up since taken from config file
    for bond_id, bond_type, first_atom_id, second_atom_id in protein_bond_data:
        configFile.write('%d %d %d %d\n' %(bond_id+last_bond_id, bond_type,
                                           first_atom_id+num_atoms_already_placed,second_atom_id+num_atoms_already_placed))
        
    # Close file
    configFile.close()

def write_config_cuboid_cavity_with_inner_and_outer_prot(simBox,
        cavity_inner_prot_coords, cavity_inner_protein_bond_data,
        outer_protein_coords, outer_protein_bond_data,
        num_atom_types = 40):
    """
    Creates a config file for a slab containing a cuboidal cavity with multiphasic
    condensate surrounding it.

    Arguments:
    - simBox (float list): List of the lower and upper bounds for the simulation box
    - cavity_inner_prot_coords (tuple list): A list of tuples containing details for
                                    each atom of the cavity and inner protein chains
                    (atom_id, mol_id, atom_type, atom_charge, xcoord, ycoord, zcoord)
    - cavity_inner_protein_bond_data (list of tuples): Bond data for the inner protein
                                    chains
                                    (bond_id, bond_type, first_atom_id, second_atom_id)
    - outer_protein_coords (tuple list): A list of tuples containing details for each
                                    atom of the outer protein chains
    - outer_protein_bond_data (list of tuples): Bond data for the outer protein chains
                                    (bond_id, bond_type, first_atom_id, second_atom_id)
    
    Returns: config file for LAMMPS simulation - "config_cuboid_cavity_with_inner_and_outer_prot.dat"
    """
    parent_dir = Path(__file__).parent
    with open(f'{parent_dir}/amino_acid_dict.json', 'r') as f:
        aa_dict = json.load(f)
    
    configFile = open('config_cuboid_cavity_with_inner_and_outer_prot.dat','w')
    configFile.write('LAMMPS data file with cuboidal cavity to hold RNA chains\n\n')

    # Overall data
    # Note: cavity_inner_prot_coords contains info. about both - cavity and inner proteins
    num_atoms = len(cavity_inner_prot_coords) + 2*len(outer_protein_coords)
    configFile.write(f'{num_atoms} atoms\n')
    configFile.write(f'{num_atom_types} atom types\n') # Atom types remain unchanged

    # Note: cavity_inner_protein_bond_data contains bond info. for inner proteins    
    num_bonds = len(cavity_inner_protein_bond_data) + 2*len(outer_protein_bond_data)
    configFile.write(f'{num_bonds} bonds\n')
    configFile.write('1 bond types\n\n') # Bond types remain unchanged

    # Simulation box
    configFile.write('%5f   %5f  xlo xhi\n'%(simBox[0][0],simBox[0][1]))
    configFile.write('%5f   %5f  ylo yhi\n'%(simBox[1][0],simBox[1][1]))
    configFile.write('%5f   %5f  zlo zhi\n\n'%(simBox[2][0],simBox[2][1]))
    
    # Particle masses
    configFile.write('Masses\n\n')
    # First assign ids and masses for Amino acids for protein chains 
    for key, value in aa_dict.items():
        configFile.write(f"{value.get('id')} {value.get('mass')}\n")
    # Second, assign ids and masses for Amino acids for cavity monomers 
    for key, value in aa_dict.items():
        configFile.write(f"{value.get('id')+20} {value.get('mass')}\n")

    # Particle POSITIONS
    configFile.write('\nAtoms\n\n')

    # First assign positions of cavity and inner protein (already equilibrated around the cavity)
    # We just paste the information as is since it has already been equilibrated
    cavity_and_inner_protein_positions = [] # We need this list in the later part
    for atom_id, mol_id, atom_type, atom_charge, xcoord, ycoord, zcoord in cavity_inner_prot_coords:
        configFile.write('%d %d %d  %f %f  %f  %f\n' %(atom_id, mol_id, atom_type, atom_charge, xcoord, ycoord, zcoord))
        cavity_and_inner_protein_positions.append([xcoord, ycoord, zcoord])
    cavity_and_inner_protein_positions = np.array(cavity_and_inner_protein_positions)

    # Second, assign outer protein positions
    # We need to add proteins on both sides of the cavity.
    wrapped_zcoords = []
    Lz = simBox[2][1] - simBox[2][0]
    for atom_id, mol_id, atom_type, atom_charge, xcoord, ycoord, zcoord in outer_protein_coords:
        zw = zcoord - (zcoord//Lz)*Lz
        wrapped_zcoords.append(zw)
    wrapped_zcoords = np.array(wrapped_zcoords)

    # There is a chance that the wrapped coordinates of the protein chains cross the PB 
    # along z direction. For the next logic to work we need them not to cross the PB
    tmp = 0.
    while (np.min(wrapped_zcoords) < simBox[2][0]+0.1*Lz or np.max(wrapped_zcoords) > simBox[2][1]-0.1*Lz):
        tmp += 50.
        wrapped_zcoords -= 50.0
        wrapped_zcoords -= (wrapped_zcoords//Lz)*Lz
    
    # ****************   1. Right side    ************************
    # Calculate the shift to place the protein condensate on the right side of cavity
    zw_min = np.min(wrapped_zcoords) # minima of the outer proteins (wrapped)
    zc_max = np.max(cavity_and_inner_protein_positions[:,2]) # maxima of inner proteins
    z_buff = 5.0 # buffer separation between the two
    delta_z = zc_max + z_buff - zw_min - tmp # We add this quantity to unwrapped zcoord

    # Since Cavity+inner_prot are already placed we need to adjust atom_id and mol_id accordingly
    num_cavity_and_inner_prot_atoms = cavity_and_inner_protein_positions.shape[0]
    highest_mol_id = max([t[1] for t in cavity_inner_prot_coords]) # molid is at index 1
    for atom_id, mol_id, atom_type, atom_charge, xcoord, ycoord, zcoord in outer_protein_coords:
        configFile.write('%d %d %d  %f %f  %f  %f\n' %(atom_id+num_cavity_and_inner_prot_atoms, 
                                                       mol_id+highest_mol_id, atom_type, 
                                                       atom_charge, xcoord, ycoord, zcoord+delta_z))

    # ****************   2. Left side    ************************
    # Calculate the shift to place the protein condensate on the left side of cavity
    zw_max = np.max(wrapped_zcoords) # maxima of the outer proteins (wrapped)
    zc_min = np.min(cavity_and_inner_protein_positions[:,2]) # minima of inner proteins
    z_buff = 2.5 # buffer separation between the two
    delta_z = zc_min - z_buff - zw_max - tmp

    # Since Cavity and proteins(right side) are already placed we need to adjust atom_id and mol_id accordingly
    num_atoms_already_placed = cavity_and_inner_protein_positions.shape[0] + len(outer_protein_coords)
    last_mol_id = highest_mol_id + max([t[1] for t in outer_protein_coords]) # data can be jumbled up
    for atom_id, mol_id, atom_type, atom_charge, xcoord, ycoord, zcoord in outer_protein_coords:
        configFile.write('%d %d %d  %f %f  %f  %f\n' %(atom_id+num_atoms_already_placed, 
                                                       mol_id+last_mol_id, atom_type, 
                                                       atom_charge, xcoord, ycoord, zcoord+delta_z))


    # BOND DATA
    configFile.write('\nBonds\n\n')
    # Inner proteins: Bond data already accounts for the cavity monomers
    for bond_id, bond_type, first_atom_id, second_atom_id in cavity_inner_protein_bond_data:
        configFile.write('%d %d %d %d\n' %(bond_id,bond_type,first_atom_id,second_atom_id))
    
    # Outer protein
    # Again, we have two sets of bond data (right side and left side)    
    #  ***************   1. Right side    *********************
    # We need to account for the atom ids and bond ids
    highest_inner_prot_bond_id = max([t[0] for t in cavity_inner_protein_bond_data]) # Data can be jumbled up
    for bond_id, bond_type, first_atom_id, second_atom_id in outer_protein_bond_data:
        configFile.write('%d %d %d %d\n' %(bond_id+highest_inner_prot_bond_id, bond_type,
                                           first_atom_id+num_cavity_and_inner_prot_atoms,
                                           second_atom_id+num_cavity_and_inner_prot_atoms))
    #  ***************   2. Left side    *********************
    # We need to account for the atom ids and bond ids
    last_bond_id = highest_inner_prot_bond_id + max([t[0] for t in outer_protein_bond_data])
    for bond_id, bond_type, first_atom_id, second_atom_id in outer_protein_bond_data:
        configFile.write('%d %d %d %d\n' %(bond_id+last_bond_id, bond_type,
                                           first_atom_id+num_atoms_already_placed,second_atom_id+num_atoms_already_placed))
    
    # Close file
    configFile.close()

def add_chains_to_cuboidal_cavity(simBox, system_coords, system_bond_data,
                                  cavity_centre_wrapped, cavity_width,
                                  cavity_peptides_coords, cavity_peptides_bond_data,
                                  num_atom_types = 60):
    """
    Function to add RNA/peptide chains inside a cavity contained inside a protein condensate

    Arguments:
    - system_coords (tuple list): A list of tuples containing details for each atom of the 
                                cavity and protein chains
                                (atom_id, mol_id, atom_type, atom_charge, xcoord, ycoord, zcoord)
    - system_bond_data (tuple list): Bond data for the protein chains
                                    (bond_id, bond_type, first_atom_id, second_atom_id)
    - cavity_centre_wrapped (float): Centre of the cavity along z direction
    - cavity_width (float): Width of cavity along z direction
    - cavity_peptides_coords (tuple list):  A list of tuples containing details for each atom of 
                                the chains inside the cavity
                                (atom_id, mol_id, atom_type, atom_charge, xcoord, ycoord, zcoord)
    - cavity_peptides_bond_data (tuple list): Bond data for peptides inside the cavity
                                    (bond_id, bond_type, first_atom_id, second_atom_id)
    """
    parent_dir = Path(__file__).parent
    with open(f'{parent_dir}/amino_acid_dict.json', 'r') as f:
        aa_dict = json.load(f)
    
    configFile = open('config_with_peptides_inside_cavity.dat','w')
    configFile.write('LAMMPS data file with cuboidal cavity and chains inside cavity\n\n')
    
    # Overall data
    # Note: system_coords contains data for the cavity, inner proteins and outer proteins
    num_atoms = len(system_coords) + len(cavity_peptides_coords)
    configFile.write(f'{num_atoms} atoms\n')
    configFile.write(f'{num_atom_types} atom types\n') # 60 atom types

    # Note: system_bond_data contains bond info. for inner and outer proteins    
    num_bonds = len(system_bond_data) + len(cavity_peptides_bond_data)
    configFile.write(f'{num_bonds} bonds\n')
    configFile.write('1 bond types\n\n') # Bond types remain unchanged

    # Simulation box
    configFile.write('%5f   %5f  xlo xhi\n'%(simBox[0][0],simBox[0][1]))
    configFile.write('%5f   %5f  ylo yhi\n'%(simBox[1][0],simBox[1][1]))
    configFile.write('%5f   %5f  zlo zhi\n\n'%(simBox[2][0],simBox[2][1]))

    # Particle masses
    configFile.write('Masses\n\n')
    # First assign ids and masses for Amino acids for protein chains 
    for key, value in aa_dict.items():
        configFile.write(f"{value.get('id')} {value.get('mass')}\n")
    # Second, assign ids and masses for Amino acids for cavity monomers 
    for key, value in aa_dict.items():
        configFile.write(f"{value.get('id')+20} {value.get('mass')}\n")
    # Third, assign ids and masses for Amino acids for chains inside cavity 
    for key, value in aa_dict.items():
        configFile.write(f"{value.get('id')+40} {value.get('mass')}\n")

    # Particle POSITIONS
    configFile.write('\nAtoms\n\n')

    # First assign positions of the cavity and condensate proteins (already equilibrated)
    # We want to recenter the cavity and condensate proteins such that it is centered at
    # the center of the simulation box
    # (Here assuming that only box simension in the z direction is changed.)
    Lz = simBox[2][1]-simBox[2][0]
    simBox_centre = simBox[2][0] + Lz/2.
    delta_z = simBox_centre - cavity_centre_wrapped # Needs to be added to the particle positions
    for atom_id, mol_id, atom_type, atom_charge, xcoord, ycoord, zcoord in system_coords:
        configFile.write('%d %d %d  %f %f  %f  %f\n' %(atom_id, mol_id, atom_type, atom_charge, xcoord, ycoord, zcoord+delta_z))

    # Second, assign positions of the peptides inside the cavity
    # Compute the width of the slab of peptides that go inside the cavity 
    # (ensuring smaller than the cavity width)
    wrapped_zcoords = []
    for atom_id, mol_id, atom_type, atom_charge, xcoord, ycoord, zcoord in cavity_peptides_coords:
        zw = zcoord - (zcoord//Lz)*Lz
        wrapped_zcoords.append(zw)
    wrapped_zcoords = np.array(wrapped_zcoords)
    # There is a chance that the wrapped coordinates of the protein chains cross the PB 
    # along z direction. For the next logic to work we need them not to cross the PB
    tmp = 0.
    while (np.min(wrapped_zcoords) < simBox[2][0]+0.1*Lz or np.max(wrapped_zcoords) > simBox[2][1]-0.1*Lz):
        tmp += 50.
        wrapped_zcoords -= 50.0
        wrapped_zcoords -= (wrapped_zcoords//Lz)*Lz
    
    width = np.max(wrapped_zcoords) - np.min(wrapped_zcoords)
    print(width)
    cav_maxz = simBox_centre + cavity_width/2.
    cav_minz = simBox_centre - cavity_width/2.
    if (width > cavity_width - 2.):
        raise ValueError("Width of the slab that goes inside the cavity is too large! compress more.")
    else:
        z_buff = 1.0
        delta_z = cav_minz + z_buff - np.min(wrapped_zcoords) - tmp

    # Since the system coordinates are already fixed, we need to adjust the atom_id and mol_id accordingly
    num_system_atoms = len(system_coords)
    highest_mol_id = max([t[1] for t in system_coords]) # molid is at index 1
    # We need to adjust atom_type as well since the slabs are generated using atom_types [1,20]
    for atom_id, mol_id, atom_type, atom_charge, xcoord, ycoord, zcoord in cavity_peptides_coords:
        configFile.write('%d %d %d  %f %f  %f  %f\n' %(atom_id+num_system_atoms, 
                                                       mol_id+highest_mol_id,
                                                       atom_type+40, 
                                                       atom_charge, xcoord, ycoord, zcoord+delta_z))

    # BOND DATA
    configFile.write('\nBonds\n\n')
    # System data contains bond data for inner and outer proteins
    for bond_id, bond_type, first_atom_id, second_atom_id in system_bond_data:
        configFile.write('%d %d %d %d\n' %(bond_id,bond_type,first_atom_id,second_atom_id))
    
    # Peptides inside cavity
    # We need to account for the bond_ids and atom_ids
    highest_system_bond_id = max([t[0] for t in system_bond_data]) # Data can be jumbled up
    for bond_id, bond_type, first_atom_id, second_atom_id in cavity_peptides_bond_data:
        configFile.write('%d %d %d %d\n' %(bond_id+highest_system_bond_id,
                                           bond_type,
                                           first_atom_id+num_system_atoms,
                                           second_atom_id+num_system_atoms))
    
    configFile.close()

def place_chains_in_confinement(seq, box_bounds, nchains):
    """
    Function to place protein chains in straight lines inside a box.

    Arguments:
    - seq (str): Protein sequence that we want to place inside box
    - box_bounds (list): Box dimensions that contains the protein chains
    - nchains (int): Number of chains to place inside the box

    Returns:
    - position_data (list of tuples): (atom_id, mol_id, atom_type, atom_charge, x, y, z)
    - bond_data (list of tuples): (bond_id, bond_type, atom_id_1, atom_id_2)
    """
    r0 = 3.81 # equilibrium bond length (Angstroms)
    seq_length = len(seq)*r0 + 10.0 # Approximate excluded length of chain with buffer
    Lx = box_bounds[0][1] - box_bounds[0][0]
    nx = int(Lx//seq_length)
    Ly = box_bounds[1][1] - box_bounds[1][0]
    ny = int(Ly//seq_length)
    Lz = box_bounds[2][1] - box_bounds[2][0]
    nz = int(Lz//seq_length)
    buffer = 5.0 # buffer distance between chains
    box_origin = [box_bounds[0][0], box_bounds[1][0], box_bounds[2][0]]
    # Extend chains along x direction if the x dimension is largest
    lattice_points = []
    if nx >= ny and nx >= nz:
        direction = "x"
        ny = int(Ly//buffer)
        nz = int(Lz//buffer)
        for j in range(ny):
            for k in range(nz):
                for i in range(nx):
                    x_pos = i*seq_length
                    y_pos = j*buffer
                    z_pos = k*buffer
                    if (0. <= x_pos <= Lx-seq_length) and (0. <= y_pos <= Ly-buffer) and (0. <= z_pos <= Lz-buffer):
                        lattice_points.append([box_origin[0]+x_pos, box_origin[1]+y_pos, box_origin[2]+z_pos])
    # Extend chains along y direction if the y dimension is largest
    elif ny >= nx and ny >= nz:
        direction = "y"
        nx = int(Lx//buffer)
        nz = int(Lz//buffer)
        for i in range(nx):
            for k in range(nz):
                for j in range(ny):
                    x_pos = i*buffer
                    y_pos = j*seq_length
                    z_pos = k*buffer
                    if (0. <= x_pos <= Lx-buffer) and (0. <= y_pos <= Ly-seq_length) and (0. <= z_pos <= Lz-buffer):
                        lattice_points.append([box_origin[0]+x_pos, box_origin[1]+y_pos, box_origin[2]+z_pos])
    # Extend chains along z direction if the z dimension is largest
    elif nz >= nx and nz >= ny:
        direction = "z"
        nx = int(Lx//buffer)
        ny = int(Ly//buffer)
        for i in range(nx):
            for j in range(ny):
                for k in range(nz):
                    x_pos = i*buffer
                    y_pos = j*buffer
                    z_pos = k*seq_length
                    if (0. <= x_pos <= Lx-buffer) and (0. <= y_pos <= Ly-buffer) and (0. <= z_pos <= Lz-seq_length):
                        lattice_points.append([box_origin[0]+x_pos, box_origin[1]+y_pos, box_origin[2]+z_pos])
    
    lattice_points = np.array(lattice_points) # lattice points for chains

    # Randomly choose nchains lattice points from the ones generated to place the chains
    random_indices = np.random.choice(np.arange(0, len(lattice_points)), nchains, replace=False)
    chosen_lattice_points = lattice_points[random_indices]

    position_data = []
    parent_dir = Path(__file__).parent
    with open(f'{parent_dir}/amino_acid_dict.json', 'r') as f:
        aa_dict = json.load(f)
    
    atom_id_counter = 0
    for idx, pos in enumerate(chosen_lattice_points):
        mol_id = idx + 1 # mol id
        xcoord = pos[0]
        ycoord = pos[1]
        zcoord = pos[2]
        for aa in seq:
            atom_id_counter += 1 # atom id
            atom_type = aa_dict[f"{aa}"]["id"] # atom type
            atom_charge = aa_dict[f"{aa}"]["charge"] # atom charge
            if direction == "x":
                position_data.append((atom_id_counter, mol_id, atom_type, atom_charge, xcoord, ycoord, zcoord))
                xcoord += r0
            elif direction == "y":
                position_data.append((atom_id_counter, mol_id, atom_type, atom_charge, xcoord, ycoord, zcoord))
                ycoord += r0
            elif direction == "z":
                position_data.append((atom_id_counter, mol_id, atom_type, atom_charge, xcoord, ycoord, zcoord))
                zcoord += r0
    
    bond_data = []
    bond_id_counter = 0
    bond_type = 1
    for i in range(nchains):
        start_atom_id = i*len(seq)
        for j in range(1,len(seq)):
            bond_id_counter += 1
            first_atom_id = start_atom_id + j
            second_atom_id = first_atom_id + 1
            bond_data.append((bond_id_counter, bond_type, first_atom_id, second_atom_id))

    return position_data, bond_data


def add_unequilibrated_chains_to_cuboidal_cavity(simBox, system_coords, system_bond_data,
                                  cavity_centre_wrapped, cavity_dimensions, seq, nchains,
                                  num_atom_types = 60):
    """
    Function to add RNA/peptide chains inside a cavity contained inside a protein condensate

    Arguments:
    - simBox (list): Simulation box coordinates
    - system_coords (tuple list): A list of tuples containing details for each atom of the 
                                cavity and protein chains
                                (atom_id, mol_id, atom_type, atom_charge, xcoord, ycoord, zcoord)
    - system_bond_data (tuple list): Bond data for the protein chains
                                    (bond_id, bond_type, first_atom_id, second_atom_id)
    - cavity_centre_wrapped (float): Centre of the cavity along z direction
    - cavity_dimensions (list): cavity coordinates
    - seq (str): Sequence of peptides that go inside the box
    """
    parent_dir = Path(__file__).parent
    with open(f'{parent_dir}/amino_acid_dict.json', 'r') as f:
        aa_dict = json.load(f)
    
    # Based on cavity coordinates, first determine the position and bond data of the peptides
    # that go inside the cavity
    cavity_box = [[cavity_dimensions[0][0] + 10., cavity_dimensions[0][1] - 10.],
                  [cavity_dimensions[1][0] + 10., cavity_dimensions[1][1] - 10.],
                  [cavity_dimensions[2][0] + 10., cavity_dimensions[2][1] - 10.]]
    cavity_peptides_coords, cavity_peptides_bond_data = place_chains_in_confinement(
                                                        seq, cavity_box, nchains)

    # Start writing config file 
    configFile = open('config_with_peptides_inside_cavity.dat','w')
    configFile.write('LAMMPS data file with cuboidal cavity and chains inside cavity\n\n')
    
    # Overall data
    # Note: system_coords contains data for the cavity, inner proteins and outer proteins
    num_atoms = len(system_coords) + len(cavity_peptides_coords)
    configFile.write(f'{num_atoms} atoms\n')
    configFile.write(f'{num_atom_types} atom types\n') # 60 atom types

    # Note: system_bond_data contains bond info. for inner and outer proteins    
    num_bonds = len(system_bond_data) + len(cavity_peptides_bond_data)
    configFile.write(f'{num_bonds} bonds\n')
    configFile.write('1 bond types\n\n') # Bond types remain unchanged

    # Simulation box
    configFile.write('%5f   %5f  xlo xhi\n'%(simBox[0][0],simBox[0][1]))
    configFile.write('%5f   %5f  ylo yhi\n'%(simBox[1][0],simBox[1][1]))
    configFile.write('%5f   %5f  zlo zhi\n\n'%(simBox[2][0],simBox[2][1]))

    # Particle masses
    configFile.write('Masses\n\n')
    # First assign ids and masses for Amino acids for protein chains 
    for key, value in aa_dict.items():
        configFile.write(f"{value.get('id')} {value.get('mass')}\n")
    # Second, assign ids and masses for Amino acids for cavity monomers 
    for key, value in aa_dict.items():
        configFile.write(f"{value.get('id')+20} {value.get('mass')}\n")
    # Third, assign ids and masses for Amino acids for chains inside cavity 
    for key, value in aa_dict.items():
        configFile.write(f"{value.get('id')+40} {value.get('mass')}\n")

    # Particle POSITIONS
    configFile.write('\nAtoms\n\n')

    # First assign positions of the cavity and condensate proteins (already equilibrated)
    # We want to recenter the cavity and condensate proteins such that it is centered at
    # the center of the simulation box
    # (Here assuming that only box simension in the z direction is changed.)
    Lz = simBox[2][1]-simBox[2][0]
    simBox_centre = simBox[2][0] + Lz/2.
    delta_z = simBox_centre - cavity_centre_wrapped # Needs to be added to the particle positions
    for atom_id, mol_id, atom_type, atom_charge, xcoord, ycoord, zcoord in system_coords:
        configFile.write('%d %d %d  %f %f  %f  %f\n' %(atom_id, mol_id, atom_type, atom_charge, xcoord, ycoord, zcoord+delta_z))

    # Second, assign positions of the peptides inside the cavity
    # Since the system coordinates are already fixed, we need to adjust the atom_id and mol_id accordingly
    num_system_atoms = len(system_coords)
    highest_mol_id = max([t[1] for t in system_coords]) # molid is at index 1
    # We need to adjust atom_type as well since peptides are generated using atom_types [1,20]
    # In addition we need to adjust positions of peptides to be inside the cavity
    for atom_id, mol_id, atom_type, atom_charge, xcoord, ycoord, zcoord in cavity_peptides_coords:
        configFile.write('%d %d %d  %f %f  %f  %f\n' %(atom_id+num_system_atoms, 
                                                       mol_id+highest_mol_id,
                                                       atom_type+40, 
                                                       atom_charge, xcoord, ycoord, zcoord+Lz/2.))

    # BOND DATA
    configFile.write('\nBonds\n\n')
    # System data contains bond data for inner and outer proteins
    for bond_id, bond_type, first_atom_id, second_atom_id in system_bond_data:
        configFile.write('%d %d %d %d\n' %(bond_id,bond_type,first_atom_id,second_atom_id))
    
    # Peptides inside cavity
    # We need to account for the bond_ids and atom_ids
    highest_system_bond_id = max([t[0] for t in system_bond_data]) # Data can be jumbled up
    for bond_id, bond_type, first_atom_id, second_atom_id in cavity_peptides_bond_data:
        configFile.write('%d %d %d %d\n' %(bond_id+highest_system_bond_id,
                                           bond_type,
                                           first_atom_id+num_system_atoms,
                                           second_atom_id+num_system_atoms))
    
    configFile.close()

def write_config_cuboidal_cavity_w_peptides_wo_outer_proteins(simBox, cavity_seq, 
                                                              peptideSeq, nchains, 
                                                              num_atom_types = 60):
    parent_dir = Path(__file__).parent
    with open(f'{parent_dir}/amino_acid_dict.json', 'r') as f:
        aa_dict = json.load(f)
    
    x_length = simBox[0][1] - simBox[0][0] # same as simulation box
    y_length = simBox[1][1] - simBox[1][0] # same as simulation box
    z_length = 100 # 10 nm width of the cavity
    # Generate cavity positions and types
    cavity_positions, cavity_types = generate_cuboid_cavity(cavity_seq, x_length, y_length, z_length)

    # Generate position and bond data for the peptides that go inside the cavity
    cavity_box = [[simBox[0][0] + 10., simBox[0][1] - 10.],
                  [simBox[1][0] + 10., simBox[1][1] - 10.],
                  [-z_length/2. + 10., z_length/2. - 10.]]
    cavity_peptides_coords, cavity_peptides_bond_data = place_chains_in_confinement(
                                                        peptideSeq, cavity_box, nchains)

    # Start writing config file 
    configFile = open('config_cavity_and_peptides_only.dat','w')
    configFile.write('LAMMPS data file with cuboidal cavity and chains inside cavity\n\n')
    
    # Overall data
    num_atoms = len(cavity_positions) + len(cavity_peptides_coords)
    configFile.write(f'{num_atoms} atoms\n')
    configFile.write(f'{num_atom_types} atom types\n') # 60 atom types

    num_bonds = len(cavity_peptides_bond_data)
    configFile.write(f'{num_bonds} bonds\n')
    configFile.write('1 bond types\n\n') # Bond types remain unchanged

    # Simulation box
    configFile.write('%5f   %5f  xlo xhi\n'%(simBox[0][0],simBox[0][1]))
    configFile.write('%5f   %5f  ylo yhi\n'%(simBox[1][0],simBox[1][1]))
    configFile.write('%5f   %5f  zlo zhi\n\n'%(simBox[2][0],simBox[2][1]))

    # Particle masses
    configFile.write('Masses\n\n')
    # First assign ids and masses for Amino acids for protein chains 
    for key, value in aa_dict.items():
        configFile.write(f"{value.get('id')} {value.get('mass')}\n")
    # Second, assign ids and masses for Amino acids for cavity monomers 
    for key, value in aa_dict.items():
        configFile.write(f"{value.get('id')+20} {value.get('mass')}\n")
    # Third, assign ids and masses for Amino acids for chains inside cavity 
    for key, value in aa_dict.items():
        configFile.write(f"{value.get('id')+40} {value.get('mass')}\n")

    # Particle POSITIONS
    configFile.write('\nAtoms\n\n')

    # First assign cavity
    mol_id = 1
    for index, pos in enumerate(cavity_positions):
        atom_id = index + 1
        atom_type = aa_dict[cavity_types[index]]["id"] + 20 # Add 20 for cavity monomers
        atom_charge = aa_dict[cavity_types[index]]["charge"]
        # By construction, the cavity positions are such that the center of the cavity is at (0.,0.,0.)
        # We need to recenter it to the center of the new simulation box
        xcoord = pos[0] + (simBox[0][0] + (simBox[0][1] - simBox[0][0])/2.)
        ycoord = pos[1] + (simBox[1][0] + (simBox[1][1] - simBox[1][0])/2.)
        zcoord = pos[2] + (simBox[2][0] + (simBox[2][1] - simBox[2][0])/2.)
        configFile.write('%d %d %d  %f %f  %f  %f\n' %(atom_id, mol_id, atom_type, atom_charge, xcoord, ycoord, zcoord))
    
    # Second, assign peptides's position
    num_cavity_atoms = len(cavity_positions)
    highest_mol_id = 1
    Lz = simBox[2][1]-simBox[2][0]
    for atom_id, mol_id, atom_type, atom_charge, xcoord, ycoord, zcoord in cavity_peptides_coords:
        configFile.write('%d %d %d  %f %f  %f  %f\n' %(atom_id+num_cavity_atoms, 
                                                       mol_id+highest_mol_id,
                                                       atom_type+40, 
                                                       atom_charge, xcoord, ycoord, zcoord+Lz/2.))
    
    # BOND DATA
    configFile.write('\nBonds\n\n')

    for bond_id, bond_type, first_atom_id, second_atom_id in cavity_peptides_bond_data:
        configFile.write('%d %d %d %d\n' %(bond_id,
                                           bond_type,
                                           first_atom_id+num_cavity_atoms,
                                           second_atom_id+num_cavity_atoms))
    
    configFile.close()

def create_interaction_matrix(filename):
    infile = open(filename, "r")
    matrix_data = np.full((60,60,2), np.nan)

    for line in infile:
        if "pair_coeff" in line and "wf/cut" in line:
            parts = line.strip().split()
            matrix_data[int(parts[1]) - 1, int(parts[2]) - 1, 0] = float(parts[4])
            matrix_data[int(parts[1]) - 1, int(parts[2]) - 1, 1] = float(parts[5])
    
    table = matrix_data[:,:,0]
    df = pd.DataFrame(table)
    
    df.to_csv("interaction_matrix.csv", index=False, header=False)


if __name__ == "__main__":
    print("Executing function calls from within the script")
    """                             ****** SECTION *******
    This section takes two sequences and generates a LAMMPS config file that contains the two chains
    in a simulation box. The configuration needs to be equilibrated through minimization/running a
    simulation for some time.
    """
    # seq1 = "AFAAF"*10
    # check_sequence_validity(seq1)
    
    # seq2 = "F"*50
    # check_sequence_validity(seq2)
    # seq2 = mutate_sequence(seq2, 0.6)
    # check_sequence_validity(seq2)
    
    # gen_mixture_chain_config(seq1, seq2, "initialConfig.dat")
    
    """                            ****** SECTION *****
    """
    # # Global density profile
    # tsteps, density_profile = parse_density_profile_file("/Users/yw9071_admin/Downloads/density_all.dat")
    # bins, bin_density = compute_averaged_density_profile(density_profile)
    # # Visualization
    # fig, ax = plt.subplots()
    # # Recentered density profile
    # ax.plot(bins, bin_density, marker='o', linestyle='')
    # left_intrfc, right_intrfc, fine_coords, fitted_profile = find_interfaces(bins, bin_density)
    # ax.plot(fine_coords, fitted_profile)
    # # Interfaces
    # yvals = np.linspace(min(bin_density), max(bin_density), 3)
    # ax.plot(np.repeat(left_intrfc, 3), yvals, color="black", linestyle="dashed")
    # ax.plot(np.repeat(right_intrfc, 3), yvals, color="black", linestyle="dashed")
    # # Set labels
    # ax.set_xlabel(r"$z/L_\mathrm{z}$")
    # ax.set_ylabel(r"$\rho \, (\mathrm{g/cm^3})$")
    # plt.tight_layout()
    # # plt.show()

    # # Protein specific density profile
    # tsteps_I, density_profile_I = parse_density_profile_file("/Users/yw9071_admin/Downloads/density_inner.dat")
    # bins_I, bin_density_I = compute_averaged_density_profile(density_profile_I)
    
    # tsteps_O, density_profile_O = parse_density_profile_file("/Users/yw9071_admin/Downloads/density_outer.dat")
    # bins_O, bin_density_O = compute_averaged_density_profile(density_profile_O)
    
    # # Recentered density profile
    # ax.plot(bins_I, bin_density_I, label="Inner protein")
    # ax.plot(bins_O, bin_density_O, label="Outer protein")
    # plt.legend()
    # plt.show()

    # mask_bins_O = (bins_O >= 0.45) & (bins_O <= 0.55)
    # rhoO_center = np.mean(bin_density_O[mask_bins_O])
    # mask_bins_I = (bins_I >= 0.45) & (bins_I <= 0.55)
    # rhoI_center = np.mean(bin_density_I[mask_bins_I])

    # mask_vapour_O = (bin_density_O < 0.05)
    # rhoO_vapour = np.mean(bin_density_O[mask_vapour_O])
    # mask_vapour_I = (bin_density_I < 0.05)
    # rhoI_vapour = np.mean(bin_density_I[mask_vapour_I])
    
    # fitness = evaluate_fitness(rhoO_center, rhoI_center, rhoO_vapour, rhoI_vapour, s=0)
    # print(fitness)

    # generate_children(5, 8, np.array([['ASDFS', '0.45'], ['JHSGD', '0.456'], ['HSJDF', '0.23'], ['HJGSD', '0.352'],
    #                                   ['SGHJD', '0.64'], ['JHSDF', '0.353'], ['HSGFJ', '0.75'], ['HSDFG', '0.566']]))


    # scd = compute_SCD("E"*25+"K"*25)
    # print(scd)
    
    # glycine, negative, neutral, neutral_with_pi, positive, positive_with_pi, aromatic = extract_sequence_composition("FYHWFVNFFFAVWFWNYRFCNRHWPWVQENFMFFWAKITGYFNEFFFDFF")
    # print(glycine, negative, neutral, neutral_with_pi, positive, positive_with_pi, aromatic)

    # box_bounds, coords, highest_mol_id = read_trajectory("result.lammpstrj")
    # num_atoms, num_bonds, bonds = read_config("final-structure.dat")


    # write_slab_config([[0.0, 200.0], [0., 200.], [0., 1000.]],
    #                   coords,
    #                   num_atoms,
    #                   num_bonds,
    #                   bonds)
    
    # positions, types = gen_seq_based_cavity("FYHWFVNFFFAVWFWNYRFCNRHWPWVQENFMFFWAKITGYFNEFFFDFF", 150.)
    # print(positions.shape)
    # # print(types)

    # write_cavity_config([[0.0, 200.0], [0., 200.], [0., 1000.]],
    #                     positions, types)

    # cavity_peptides_coords, cavity_peptides_bond_data = place_chains_in_confinement(
    #                                                     "DDDDDDDDDD", 
    #                                                     [[0.0, 200.0], [0., 200.], [-50., 50.]], 
    #                                                     500)

    # xvals = [x[4] for x in cavity_peptides_coords]
    # yvals = [x[5] for x in cavity_peptides_coords]
    # zvals = [x[6] for x in cavity_peptides_coords]
    
    # import matplotlib.pyplot as plt
    # from mpl_toolkits.mplot3d import Axes3D

    # fig = plt.figure(figsize=(10, 8))
    # ax = fig.add_subplot(111, projection='3d')
    # ax.scatter(xvals, yvals, zvals)

    # ax.set_xlabel('X Axis')
    # ax.set_ylabel('Y Axis')
    # ax.set_zlabel('Z Axis')

    # plt.show()


    # write_config_cuboidal_cavity_w_peptides_wo_outer_proteins([[0.0, 200.0], [0., 200.], [0., 1000.]],
    #                                                           "FYHWFVNFFFAVWFWNYRFCNRHWPWVQENFMFFWAKITGYFNEFFFDFF",
    #                                                           "AAAAAAAAAA",
    #                                                           50)

    # create_interaction_matrix("potential_60_particle_types.dat")