import json
import random
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
import pathlib
import random
current_dir = pathlib.Path(__file__).resolve().parent
plt.style.use(f"{current_dir}/../plotting/ymw.mplstyle")


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
    placed in straing line configuration on two opposite ends of the simulation box.

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

    num_atom_types = 40 # This is by construction of the potentials.dat file.
    # This was done so that the potentials could be maybe tuned for folded domains.

    out = open(f'{outfile}','w')
    out.write('LAMMPS data file for IDP\n\n')
    # General parameters
    out.write(f'{num_atoms} atoms\n')
    out.write(f'{num_bonds} bonds\n\n')
    out.write(f'{num_atom_types} atom types\n')
    out.write('1 bond types\n\n')

    # Simulation box size
    min_box = max(chain1_length, chain2_length) * -3
    max_box = max(chain1_length, chain2_length) * 3

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
    description
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
    return glycine/Nm, negative/Nm, neutral/Nm, neutral_with_pi/Nm, positive/Nm, positive_with_pi/Nm, aromatic/Nm


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
    
    glycine, negative, neutral, neutral_with_pi, positive, positive_with_pi, aromatic = extract_sequence_composition("FYHWFVNFFFAVWFWNYRFCNRHWPWVQENFMFFWAKITGYFNEFFFDFF")
    print(glycine, negative, neutral, neutral_with_pi, positive, positive_with_pi, aromatic)