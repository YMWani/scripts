import json
import random
from pathlib import Path


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
    mol_id = 1
    bond_id = 1
    for atom_id in range(1, chain1_length):
        out.write('%d %d %d %d\n' %(bond_id,mol_id,atom_id,atom_id+1))
        bond_id += 1
    # Write bonds for the second chain
    mol_id = 2
    for atom_id in range(chain1_length+1, chain1_length+chain2_length):
        out.write('%d %d %d %d\n' %(bond_id,mol_id,atom_id,atom_id+1))
        bond_id += 1
    
    out.close()




if __name__ == "__main__":
    print("Executing function calls from within the script")
    """                             ****** SECTION *******
    This section takes two sequences and generates a LAMMPS config file that contains the two chains
    in a simulation box. The configuration needs to be equilibrated through minimization/running a
    simulation for some time.
    """
    # parent_dir = Path(__file__).parent
    # with open(f'{parent_dir}/amino_acid_dict.json', 'r') as f:
    #     amino_acid_dict = json.load(f)
    # with open(f'{parent_dir}/amino_acid_masses.json', 'r') as f:
    #     amino_acid_mass_dict = json.load(f)

    seq1 = "AFAAF"*10
    check_sequence_validity(seq1)
    
    # seq2 = "F"*50
    # check_sequence_validity(seq2)
    # seq2 = mutate_sequence(seq2, 0.6)
    # check_sequence_validity(seq2)
    
    # gen_mixture_chain_config(seq1, seq2, "initialConfig.dat")
    
