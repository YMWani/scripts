import argparse
import os

"""
Script to take the structure and trajectory files obtained from compression simulations and generating
an initial structure file for slab simulations.

Arguments:
1. structure - path for LAMMPS data file created using write_data command in the compression simulations
2. traj - path for trajectory file obtained from the compression simulations

Stepwise procedure:
1. Position data is extracted from the last frame of the trajectory file
2. Replace the positions from the structure file with the positions from the trajectory file
   (written into a temporary file)
3. Use the temporary file and delete the velocities
   (delete temporary file and create a final slab file)
"""

# Input data
parser = argparse.ArgumentParser()
parser.add_argument("-s", "--structure", type=str, required=True)
parser.add_argument("-t", "--traj", type=str, required=True)
args = parser.parse_args()

# Function to extract the data from the last frame of a LAMMPS trajectory file
def extract_last_frame_positions(trajectory_file):
    last_frame = []
    inside_atoms_section = False

    with open(trajectory_file, 'r') as file:
        for line in file:
            # Identify sections in the dump file by the `ITEM` header
            if line.startswith("ITEM: TIMESTEP"):
                last_frame = []  # Reset for a new frame
                inside_atoms_section = False  # Reset atom section flag
                timestep = next(file).strip()  # Skip to the timestep line
                
            elif line.startswith("ITEM: ATOMS"):
                inside_atoms_section = True  # Start reading positions in the current frame
                
            elif inside_atoms_section:
                # atom_data = line.strip().split()
                last_frame.append(line)

    return last_frame, timestep

# Function to replace atom positions in the structure file with positions from trajectory file, which are in unwrapped coordinates 
def replace_positions(structFile, updated_data, outfile = "temp.dat"):
    updated_lines = []
    in_atoms_section = False
    counter = 0
    
    with open(structFile, 'r') as infile:
        for line in infile:
            # Detect start of the "Atoms" section
            if line.startswith("Atoms"):
                in_atoms_section = True
                updated_lines.append(line)  # Keep the "Atoms" header
                updated_lines.append(next(infile))  # Include blank line after "Atoms"
                continue

            # If in the "Atoms" section, replace positions
            if in_atoms_section:
                if line.strip() == "":  # Blank line marks end of the "Atoms" section
                    in_atoms_section = False
                    updated_lines.append(line)
                else:
                    updated_lines.append(updated_data[counter])
                    counter += 1
            else:
                updated_lines.append(line)  # Keep lines outside the "Atoms" section as-is

    # Write updated data to the output file
    with open(outfile, 'w') as outfile:
        outfile.writelines(updated_lines)

# Function to remove the velocities of the particles from the temp.dat file
def remove_velocities(input_file, output_file):
    with open(input_file, 'r') as infile, open(output_file, 'w') as outfile:
        in_velocities_section = False
        for line in infile:
            # Check if the line marks the beginning of the "Velocities" section
            if line.strip() == "Velocities":
                in_velocities_section = True
                outfile.write(line)
                # Move to the next line (usually a blank line after "Velocities")
                outfile.write(next(infile))
                continue
            
            # If in "Velocities" section, skip lines until it ends
            if in_velocities_section:
                if line.strip() == "":
                    # End of "Velocities" section; reset the flag
                    in_velocities_section = False
                continue
            
            # Write the line to the output file if not in "Velocities" section
            outfile.write(line)

# Function to extend the simulation box in z-direction
def create_slab(input_file, output_file):
    with open(input_file, 'r') as infile, open(output_file, 'w') as outfile:
        for line in infile:
            if "zlo zhi" in line:
                elements = line.strip().split()
                zlo = float(elements[0])
                zhi = float(elements[1])
                diff = zhi - zlo
                zhi_new = zhi + 2*diff
                zlo_new = zlo - 2*diff
                outfile.write(f"{zlo_new} {zhi_new} zlo zhi\n")
            
            else:
                outfile.write(line)
    

# Call functions in order
last_frame_info, timestep = extract_last_frame_positions(args.traj)
print(f"Exctracted positions from the last frame (timestep = {timestep}) of the trajectory file {args.traj}")

replace_positions(args.structure, last_frame_info)

remove_velocities(input_file="temp.dat", output_file="temp2.dat")
os.remove("temp.dat")

create_slab("temp2.dat", "initialSlab.dat")
os.remove("temp2.dat")
print(f"Updated structure file has been written to initialSlab.dat")
