import numpy as np
from tqdm import tqdm
import pathlib
import argparse
import matplotlib.pyplot as plt
current_dir = pathlib.Path(__file__).resolve().parent
plt.style.use(f"{current_dir}/../plotting/joseph_group.mplstyle")
plt.rcParams['text.usetex'] = True
plt.rcParams['text.latex.preamble'] = '\n'.join([r'\usepackage{sansmath}', r'\sansmath'])
import scipy.optimize as opt
from scipy.stats import sem

"""
Here we will compute the pore size distribution for any given protein condensate.

WHY?
Generally, people use the correlation length to estimate the mesh size in a polymer solution.
This method has also been used to estimate the pore size in a protein condensate. However,
the correlation length is not a good measure of the pore size in a protein condensate since the
underlying assumptions are:
1. The system is in the semi-dilute regime.
2. The network is isotropic/homogeneous.

Therefore, here we will compute the pore size distribution using the method outlined in the paper:
"Determining the Mesh Size of Polymer Solutions via the Pore Size Distribution"
doi:10.1021/acs.macromol.9b02166


Algorithm:
1. Read a trajectory file containing the positions of the particles in a protein condensate.
2. Generate a python dictionary file containing the particle radii.
3. Create an array of the particle radii for the particles in the system.
    positions = numpy array (# frames, # particles, 3)
    radii = numpy array (# particles)
4. For each frame, compute the pore size distribution using the method outlined in the paper.
5. Save the pore size distribution to a file.
"""

# Useful functions
def read_lammps_trajectory(file_path):
    """
    Efficiently reads LAMMPS trajectory file to extract positions of all chains.

    Input:
    - file_path: Path to the LAMMPS trajectory file
    Output:
    - timesteps: (num_timesteps,)
        Array of timesteps from the trajectory
    - box_sizes: (3, 2)
        Array of box sizes for each timestep
    - num_atoms: int
        Number of atoms in the trajectory
    - atom_positions: (num_timesteps, num_atoms, 3)
        Array containing atom positions at every timestep
    - atom_types: (num_atoms,)
        Array containing atom types of all atoms in system
    - mol_ids: (num_atoms,)
        Array containing molecule IDs of all atoms in system
    - atom_ids: (num_atoms,)
    Array containing atom IDs of all atoms in system
    """
    # First pass to get number of atoms and timesteps
    num_timesteps = 0
    num_atoms = 0
    
    with open(file_path, 'r') as file:
        for line in file:
            if 'ITEM: TIMESTEP' in line:
                num_timesteps += 1
            elif 'ITEM: NUMBER OF ATOMS' in line and num_atoms == 0:
                num_atoms = int(next(file).strip())
    
    # Pre-allocate arrays
    timesteps = np.zeros(num_timesteps, dtype=int)
    box_sizes = np.zeros((num_timesteps, 3, 2), dtype=float)
    atom_positions = np.zeros((num_timesteps, num_atoms, 3), dtype=float)
    atom_types = np.zeros((num_atoms,), dtype=int)
    mol_ids = np.zeros((num_atoms,), dtype=int)
    atom_ids = np.zeros((num_atoms,), dtype=int)

    # Second pass to fill arrays
    with open(file_path, 'r') as file:
        line_iter = iter(file)
        ts_idx = -1
        
        for line in tqdm(line_iter, desc="Reading trajectory"):
            if 'ITEM: TIMESTEP' in line:
                ts_idx += 1
                timesteps[ts_idx] = int(next(line_iter).strip())
            elif 'ITEM: BOX BOUNDS' in line:
                for i in range(3):
                    box_sizes[ts_idx, i] = list(map(float, next(line_iter).split()))
            elif 'ITEM: ATOMS' in line:
                for i in range(num_atoms):
                    atom_data = next(line_iter).strip().split()
                    atom_positions[ts_idx, i, 0] = float(atom_data[4])
                    atom_positions[ts_idx, i, 1] = float(atom_data[5])
                    atom_positions[ts_idx, i, 2] = float(atom_data[6])
                    # Only store atom types and mol ids once if they don't change
                    if ts_idx == 0:
                        atom_types[i] = int(atom_data[2])
                        mol_ids[i] = int(atom_data[1])
                        atom_ids[i] = int(atom_data[0])
                        
    # Check if box size is constant
    is_constant = np.allclose(box_sizes, box_sizes[0], rtol=1e-5, atol=1e-8)
    if is_constant:
        print("Box size remains constant throughout the simulation. Great!")
        box_sizes = box_sizes[0]
    
    return timesteps, box_sizes, num_atoms, atom_positions, atom_types, mol_ids, atom_ids

def robust_sphere_optimization_3d(pos, pos_p, 
                                  particle_radius=1.0, 
                                  bounds=[(None, None), (None, None), (None, None)]):
    """
    Find an optimal sphere centered at x,y,z such that it incloses the point pos_p
    and is as large as possible without intersecting any particles in pos.

    Arguments:
    - pos: (N, 3) array of particle positions
    - pos_p: (3,) array of the point to enclose
    - particle_radius: radius of the particles (single value or array of same length as pos)

    Output:
    - result: optimization result object containing:
        - (x,y,z): optimal center coordinate of sphere
        - radius: optimal radius of the sphere
    """
    
    def objective(vars):
        x, y, z = vars
        center = np.array([x, y, z])
        
        # Compute maximum feasible radius at this center position
        distances = np.linalg.norm(pos - center, axis=1)
        surface_distances = distances - particle_radius
        max_radius = np.min(surface_distances)
        
        return -max_radius  # Maximize radius
    
    def constraint_enclose_point(vars):
        x, y, z = vars
        center = np.array([x, y, z])
        
        # Compute maximum feasible radius at this center position
        distances = np.linalg.norm(pos - center, axis=1)
        surface_distances = distances - particle_radius
        max_radius = np.min(surface_distances)
        
        # The radius must be at least the distance to pos_p
        dist_to_point = np.linalg.norm(center - pos_p)
        return max_radius - dist_to_point
    
    # Initial guess: start at pos_p
    initial_guess = [pos_p[0], pos_p[1], pos_p[2]]
    
    max_radius_at_pos_p = np.min(np.linalg.norm(pos - pos_p, axis=1)) - particle_radius
    
    constraints = [
        {'type': 'ineq', 'fun': constraint_enclose_point}
    ]
    
    
    result = opt.minimize(objective, initial_guess, method='SLSQP', 
                          constraints=constraints, bounds=bounds,
                          options={'ftol': 1e-6, 'maxiter': 1000})
    
    # Add the computed radius to the result
    if result.success:
        center = result.x
        distances = np.linalg.norm(pos - center, axis=1)
        surface_distances = distances - particle_radius
        optimal_radius = np.min(surface_distances)
        # Store radius in result for convenience
        result.radius = optimal_radius
        result.center = center
    
    return result


# Read input arguments
parser = argparse.ArgumentParser()
parser.add_argument("--traj_file", type=str, required=True) # trajectory file to read
parser.add_argument("--sampling_space", type=float, nargs=6, required=True) # Sampling space for the test particles
args = parser.parse_args()

# Read the trajectory file (NOTE: wrapped positions)
timesteps, box_sizes, num_atoms, atom_positions, atom_types, mol_ids, atom_ids = read_lammps_trajectory(args.traj_file)
sampling_space = np.array(args.sampling_space, dtype=float)


# Dictionary containing particle radii
# Read potentials.dat file to extract particle radii
particle_radii = {}
with open(f"{current_dir}/potential_60_particle_types.dat", 'r') as f:
    lines = f.readlines()
    for line in lines:
        parts = line.split()
        # Only read the lines that contain pair_coeff and wf/cut
        if "pair_coeff" in parts and "wf/cut" in parts:
            # Only read similar particle types to extract sigma_ii
            if parts[1] == parts[2]:
                particle_type = int(parts[1])
                sigma_ii = float(parts[5])
                particle_radii[particle_type] = sigma_ii / 2.0
# Create an array of the particle radii for the particles in the system
particle_radius = np.zeros(num_atoms, dtype=float)
for i in range(num_atoms):
    particle_radius[i] = particle_radii[atom_types[i]]


# Useful variables
sampling_interval = 1000 # Number of points to sample in the pore size distribution
bin_width = 1.0 # Angstroms
max_radius = 40.0 # Maximum radius to consider in the pore size distribution (in Angstroms)
num_bins = int(max_radius / bin_width) # Number of bins in the pore size histogram

histogram_data = []

# Iterate over different frames
for frame_idx in tqdm(range(0, timesteps.shape[0], 10)):
    # Analyze frame <frame_idx>
    particle_positions = atom_positions[frame_idx, :, :]  # Get positions of all particles in the frame

    # Initialize the pore size distribution histogram
    pore_size_distribution = np.zeros(num_bins, dtype=int)
    bin_centers = np.arange(0, max_radius, bin_width) + bin_width / 2.0
    
    counter = 0
    while counter < sampling_interval:
        if counter % 100 == 0:
            print(f"Sampling point {counter}/{sampling_interval} ...")
        
        # Choose a random point in the box
        x = np.random.uniform(sampling_space[0], sampling_space[1])
        y = np.random.uniform(sampling_space[2], sampling_space[3])
        z = np.random.uniform(sampling_space[4], sampling_space[5])
        pos_p = np.array([x, y, z])

        # Check if the point is inside any of the particles
        distances = np.linalg.norm(particle_positions - pos_p, axis=1)
        if np.any(distances < particle_radius):
            # If the point is inside a particle, skip this point
            continue
        
        # If the point is not inside any particle, find the optimal sphere
        optimized_sphere = robust_sphere_optimization_3d(particle_positions, 
                                                        pos_p,
                                                        particle_radius,
                                                        bounds=[(box_sizes[0,0], box_sizes[0,1]),
                                                                (box_sizes[1,0], box_sizes[1,1]),
                                                                (box_sizes[2,0], box_sizes[2,1])])
        
        if optimized_sphere.success:
            # Get the radius of the optimized sphere
            radius = optimized_sphere.radius
            
            # Find the bin for this radius
            bin_idx = int(radius / bin_width)
            
            # Increment the count in the histogram
            if bin_idx < num_bins:
                pore_size_distribution[bin_idx] += 1
            
            # Increment the counter only when a valid point is sampled
            counter += 1

    # Normalize the pore size distribution
    pore_size_distribution = pore_size_distribution / sampling_interval

    # Save histogram data
    histogram_data.append(np.column_stack((bin_centers, pore_size_distribution)))
histogram_data = np.array(histogram_data)


# Compute the mean distribution across all frames
mean_distribution = np.mean(histogram_data[:,:,1], axis=0)
error_distribution = sem(histogram_data[:,:,1], axis=0)


# Save the mean distribution
np.savetxt("pore_size_distribution.dat",
           np.column_stack((bin_centers, mean_distribution, error_distribution)),
           header="Bin_center(Angstroms) Mean_Distribution Error",
           fmt='%.6f')
