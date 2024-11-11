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
    print(parent_dir)

    SCD = 0.
    for m in range(1,N):
        qm = amino_acid_dict[seq[m]]["charge"]
        for n in range(m):
            qn = amino_acid_dict[seq[n]]["charge"]
            SCD += qm * qn * (m-n)**0.5
    SCD /= N
    
    return SCD

def estimate_num_PEG_chains(phi, v_box=4e7, Nm_PEG=182, sigma_PEG=4.644, r0_PEG=2.322):
    """
    Estimate the number of PEG chains required in the simulations given a volume fraction (phi)

    Parameters:
    - phi (float): Volume fraction
    """
    # First we need to compute the effective volume of a single PEG chain
    # NOTE: There is a overlap between consecutive monomers.
    vol_overlap = (Nm_PEG-1) * overlap_volume_two_spheres(sigma_PEG/2., r0_PEG)
    vol_monomer = (np.pi/6.) * sigma_PEG**3
    vol_PEG = Nm_PEG * vol_monomer - vol_overlap

    # Work backwords given that we know the required volume fraction
    num_chains = int(np.round(phi * v_box / vol_PEG))
    return num_chains

def overlap_volume_two_spheres(R, d):
    return (np.pi/12.) * (4*R + d) * (2*R - d)

if __name__ == "__main__":
    num_PEG_chains = estimate_num_PEG_chains(phi=0.025)
    print(num_PEG_chains)