# Purpose of the repository
This repository contains scripts that are useful for analysis and simulations of proteins (written during my post-doctorate research at Princeton University).

### Directory

#### mpipi
seq2config.py is a python file that will read in an amino acid sequence and provide a LAMMPs structure file to simulate that sequence in Mpipi.

potentials.dat is a LAMMPS parameter file needed for simulation of proteins using Mpipi model.

#### mpipi-paper-replication/coexistence-simulations
Analyze-slab-simulations.py file analyzes the density profiles obtained from slab simulations performed on LAMMPS software.
<u>Input parameters</u>:
1. path: PATH to the directory that contains slab simulation data for a given temperature.
2. temperature: Temperature at which the slab simulations were performed.

<u>Output</u>:
1. analyzed_data.json file that contins the following data: temperature, dense phase concentration, dilute phase concentration.

#### plotting
<u>ymw.mplstyle</u> is a matplotlib style file that contains all the necessary parameters for generating neat plots using python.