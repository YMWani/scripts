import numpy as np

"""
We want to create a file containing the pair coefficients for the cavity simulations. 

Basically cavity consists of 20 different monomers that have the exact interaction parameters
as AA in MPIPI model. We intend to do this since we don't want the RNA molecules to interact
with the cavity monomers.

Construction:
Atom types - 61 (20 AA for proteins, 20 cavity AA, 20 AA for peptides inside cavity, 1 for crosslinker)
"""

# Open native MPIPI potentials file
potentials_file = open("potentials.dat", "r")

# First set charge of the different atoms
charges = []
bonds = []
pair_style = []
wf = np.full((20, 20, 5), np.nan)
coul = []

for line in potentials_file:
    if "charge" in line:
        parts = line.strip().split()
        charges.append((parts[2], parts[4]))

    elif "bond_coeff" in line:
        parts = line.strip().split()
        bonds.append((parts[1], parts[2], parts[3]))

    elif "pair_style" in line:
        parts = line.strip().split()
        pair_style = parts
    
    elif "pair_coeff" in line and "wf/cut" in line:
        parts = line.strip().split()
        wf[int(parts[1]) - 1, int(parts[2]) - 1, 0] = float(parts[4])
        wf[int(parts[1]) - 1, int(parts[2]) - 1, 1] = float(parts[5])
        wf[int(parts[1]) - 1, int(parts[2]) - 1, 2] = int(parts[6])
        wf[int(parts[1]) - 1, int(parts[2]) - 1, 3] = int(parts[7])
        wf[int(parts[1]) - 1, int(parts[2]) - 1, 4] = float(parts[8])

    elif "pair_coeff" in line and "coul/debye" in line:
        parts = line.strip().split()
        coul.append((parts[1], parts[2], parts[4]))

    else:
        continue


# Write new potential file that accounts for the cavity
outfile = open("potential_with_crosslinker.dat", 'w')

# AA for protein chains
for type, charge in charges:
    outfile.write(f"set type {type} charge {charge}\n")
# AA for cavity monomers
for type, charge in charges:
    outfile.write(f"set type {int(type)+20} charge {charge}\n")
# AA for peptides inside cavity
for type, charge in charges:
    outfile.write(f"set type {int(type)+40} charge {charge}\n")
# Crosslinker atom type
outfile.write(f"set type 61 charge 0.0\n")

# Bond coeff
outfile.write("\n")
# Bond type does not change
outfile.write(f"bond_coeff {bonds[0][0]} {bonds[0][1]} {bonds[0][2]}\n")

# Pair styles
outfile.write("\n")
outfile.write(f"{pair_style[0]} {pair_style[1]} {pair_style[2]} {pair_style[3]} {pair_style[4]} {pair_style[5]} {pair_style[6]} lj/cut 10.0\n")

# WF pair coeff. (Here we need to carefully account for the cavity monomers)
outfile.write("\n")

for idx1 in range(60):
    for idx2 in range(idx1, 60):
        i = idx1%20
        j = idx2%20
        # print(idx1,i, "    ", idx2, j)
        values = wf[i,j]
        if not np.any(np.isnan(values)): # Since wf is an upper traingular matrix
            if idx1 >= 20 and idx1 <= 39 and idx2 >= 20: # To ensure that peptides inside cavity don't "see" the cavity
                outfile.write(f"pair_coeff {idx1+1} {idx2+1} wf/cut {0.0} {values[1]} {int(values[2])} {int(values[3])} {values[4]}\n")
            else:
                outfile.write(f"pair_coeff {idx1+1} {idx2+1} wf/cut {values[0]} {values[1]} {int(values[2])} {int(values[3])} {values[4]}\n")
        else:
            values = wf[j,i]
            if idx1 >= 20 and idx1 <= 39 and idx2 >= 20: # To ensure that peptides inside cavity don't "see" the cavity
                outfile.write(f"pair_coeff {idx1+1} {idx2+1} wf/cut {0.0} {values[1]} {int(values[2])} {int(values[3])} {values[4]}\n")
            else:
                outfile.write(f"pair_coeff {idx1+1} {idx2+1} wf/cut {values[0]} {values[1]} {int(values[2])} {int(values[3])} {values[4]}\n")
# Add the coefficient for the crosslinker (WF_potential) with all other atom types
for idx in range(61):
    outfile.write(f"pair_coeff {idx+1} 61 wf/cut 0.0 0.0 0 0 0.0\n")

# Add the interactions for the LJ potential of all the amino acids types (set them to 0.0)
outfile.write("\n")
for idx1 in range(61):
    for idx2 in range(idx1, 61):
        if idx2 != 60:
            outfile.write(f"pair_coeff {idx1+1} {idx2+1} lj/cut 0.0 0.0 0.0\n")
        else:
            values = wf[idx1%20, 1]
            if not np.any(np.isnan(values)):
                outfile.write(f"pair_coeff {idx1+1} {idx2+1} lj/cut {values[0]} {values[1]} {2**(1/6)*values[1]}\n")
            else:
                values = wf[1, idx1%20]
                outfile.write(f"pair_coeff {idx1+1} {idx2+1} lj/cut {values[0]} {values[1]} {2**(1/6)*values[1]}\n")

# coul/debye coeff.
outfile.write("\n")
# Atom types that have charges
charged_types = np.unique(np.array([int(x[0]) for x in coul]))
charged_types_cavity = charged_types + 20
charged_types_peptides = charged_types + 40
charged_types = np.concatenate((charged_types, charged_types_cavity, charged_types_peptides))

# print(charged_types)

for idx1 in range(charged_types.shape[0]):
    for idx2 in range(idx1, charged_types.shape[0]):
        # print(f"{charged_types[idx1]} {charged_types[idx2]}")
        if charged_types[idx1] >= 20 and charged_types[idx1] <= 39 and charged_types[idx2] >= 20:
            continue
        else:
            outfile.write(f"pair_coeff {charged_types[idx1]} {charged_types[idx2]} coul/debye 35.\n")    

outfile.close()