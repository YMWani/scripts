import sys
import pathlib
import argparse
import numpy as np
import matplotlib.pyplot as plt

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))
from utils import read_full_trajectory, wrap_positions

parser = argparse.ArgumentParser(description="Compute the volume fraction of chains in the active site over the trajectory.")
parser.add_argument("--traj", type=str, default=f"guest_chains.lammpstrj", help="LAMMPS trajectory file")
parser.add_argument("--zlo", type=float, default=950.0, help="Lower z bound of the active site")
parser.add_argument("--zhi", type=float, default=1050.0, help="Upper z bound of the active site")
parser.add_argument("--block", type=int, default=10, help="Number of consecutive frames to average over")
parser.add_argument("--radius", type=float, default=2.345, help="Bead radius")
parser.add_argument("--bond", type=float, default=3.8, help="Bond length between consecutive beads")
args = parser.parse_args()

box_bounds, coords, timesteps = read_full_trajectory(args.traj)

# coords[frame, atom] = (atom number, molecule ID, x, y, z) with unwrapped positions
num_particles = np.zeros(len(timesteps), dtype=int)
num_bonds = np.zeros(len(timesteps), dtype=int)
num_chains = np.zeros(len(timesteps), dtype=int)
for frame in range(len(timesteps)):
    order = np.argsort(coords[frame, :, 0])
    mol_ids = coords[frame, order, 1]
    positions = wrap_positions(coords[frame, order, 2:5], box_bounds)
    in_site = (positions[:, 2] >= args.zlo) & (positions[:, 2] <= args.zhi)
    # Consecutive atom ids in the same molecule are bonded; count bonds with both beads in the site
    bonded = mol_ids[1:] == mol_ids[:-1]
    num_particles[frame] = np.sum(in_site)
    num_bonds[frame] = np.sum(bonded & in_site[1:] & in_site[:-1])
    num_chains[frame] = len(np.unique(mol_ids[in_site]))

# Compute volume fraction of active site
# Bead volume minus the lens-shaped overlap between two bonded beads
vol_sphere = 4 / 3 * np.pi * args.radius**3
vol_overlap = np.pi * (2 * args.radius - args.bond)**2 * (args.bond + 4 * args.radius) / 12.0
vol_active_site = ((box_bounds[0, 1] - box_bounds[0, 0]) * (box_bounds[1, 1] - box_bounds[1, 0])
                   * (args.zhi - args.zlo))
vol_frac = (num_particles * vol_sphere - num_bonds * vol_overlap) / vol_active_site

print(f"Mean volume fraction in active site: {vol_frac.mean():.4f} +/- {vol_frac.std():.4f}")
print(f"Mean number of chains with at least one particle in active site: {num_chains.mean():.2f}")

np.savetxt(f"active_site_counts.dat",
           np.column_stack((timesteps, num_particles, num_bonds, num_chains, vol_frac)),
           fmt=["%d", "%d", "%d", "%d", "%.6f"], header="timestep num_particles num_bonds num_chains vol_frac")

# Average over consecutive, non-overlapping blocks of frames (leftover frames at the end are dropped)
num_blocks = len(vol_frac) // args.block
block_vol_frac = vol_frac[:num_blocks * args.block].reshape(num_blocks, args.block).mean(axis=1)
block_chains = num_chains[:num_blocks * args.block].reshape(num_blocks, args.block).mean(axis=1)

np.savetxt(f"active_site_counts_block{args.block}.dat",
           np.column_stack((np.arange(num_blocks), block_vol_frac, block_chains)),
           fmt=["%d", "%.6f", "%.2f"], header="block vol_frac num_chains")

fig, ax = plt.subplots(figsize=(3, 1.5))
ax.plot(block_vol_frac, marker=".", markersize=1, ls="-")
# Vertical lines at the frames where timestep is zero, converted to block units
for restart in np.where(timesteps == 0)[0] / args.block:
    ax.axvline(restart, color="gray", linestyle="--", linewidth=0.5, alpha=0.5)
ax.set_xlabel("Time")
ax.set_ylabel(r"$\phi_{\mathrm{active\ site}}$")
fig.tight_layout()
fig.savefig(f"active_site_vol_frac.pdf")
