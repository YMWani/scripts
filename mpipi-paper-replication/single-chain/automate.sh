#!/bin/bash

# Specify the file name
filename="sequences.txt"

# Check if the file exists
if [ -f "$filename" ]; then
  # Read the file line by line
  while IFS= read -r line
  do
    # Split the line into two components using `read`
    component1=$(echo "$line" | awk '{print $1}')
    component2=$(echo "$line" | awk '{print $2}')
    # Print the components
    # echo "Component 1: $component1"
    # echo "Component 2: $component2"
    # Create a directory with the name of the protein
    mkdir ${component1}
    cd ${component1}
    python3 ../../mpipi/gen-lammps-structure-file/seq2config.py 1 ${component2}
    cd ..
  done < "$filename"
else
  echo "File not found!"
fi
