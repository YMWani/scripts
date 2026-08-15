"""
Shared core for the pore size distribution (PSD) analysis.

WHY THIS MODULE EXISTS
----------------------
The PSD calculation had been copied by hand into three places
(~/scripts/RNA_flux_project/, crosslinker_fraction_sweep/, interaction_based_crosslinking/)
and the copies drifted. A wrapped-vs-unwrapped detection bug survived in all three
precisely because a fix in one never propagated to the others. The trajectory reader,
the sphere optimizer, the rejection sampler and the multiprocessing plumbing are
identical for every geometry, so they live here exactly once. The geometry-specific
part -- deciding WHICH region of space to sample -- lives in psd_windows.py and in the
two thin entry points:

    pore_size_distribution_slab.py         condensate with two liquid-vapour interfaces
    pore_size_distribution_active_site.py  condensate split by an immobile active site

All four files must sit in the same directory; Python puts a script's own directory on
sys.path, so `import psd_core` resolves regardless of the working directory the job
runs from.

METHOD
------
Pore size distribution following
    "Determining the Mesh Size of Polymer Solutions via the Pore Size Distribution"
    doi:10.1021/acs.macromol.9b02166
A random point is drawn in the sampling region, rejected if it lies inside a particle,
and otherwise grown into the largest sphere that contains it without intersecting any
particle. The distribution of those radii is the PSD.

The correlation length is NOT used, because it assumes a semi-dilute, isotropic,
homogeneous network -- none of which holds for a crosslinked protein condensate.

PERFORMANCE
-----------
Inherited from the tuned single-file version: analytic gradients are supplied to SLSQP
instead of letting scipy finite-difference them (~3.3x cheaper), and the per-sample
optimizations are distributed over a multiprocessing pool. The objective (a min over
per-particle surface distances) is non-smooth, so analytic and finite-difference
gradients can land on different nearby local optima for the same query point; that was
validated to be irrelevant in aggregate (L1 = 0.03 between normalised histograms,
two-sample KS p = 1.0 over a 300-sample comparison).

PERIODICITY
-----------
All boxes here are fully periodic (`boundary p p p`) and the condensate spans the full
x-y cross-section. Distances therefore use the MINIMUM IMAGE CONVENTION in ALL THREE
dimensions. Without it in x and y, a test point near a lateral face grows its sphere
into the empty space outside the box and reports a pore that is really just the outside
of the simulation cell -- an artifact that inflates the large-radius tail, which is the
part of the distribution carrying the information about the network. In z it is
redundant for every geometry looked at so far, since the sampling window always sits far
more than max_radius from the z faces, but it is applied anyway: it costs one rounding
operation and makes the result safe by construction instead of contingent on where the
window happened to land.

Minimum image is used rather than explicit ghost particles because it is exact (ghosts
also require widening the optimizer's centre bounds, and covering that correctly needs a
shell of 2*max_radius, roughly tripling the particle count) and because it costs no
extra particles at all.
"""

import numpy as np
from tqdm import tqdm
import multiprocessing as mp
import scipy.optimize as opt
from scipy.spatial import cKDTree
from scipy.stats import sem


# ----------------------------------------------------------------------------------
# Trajectory input
# ----------------------------------------------------------------------------------
def read_lammps_trajectory(file_path, quiet=False):
    """
    Efficiently reads a LAMMPS trajectory file.

    Assumes the dump columns are `id mol type q x y z`, i.e. coordinates in columns
    4,5,6 -- which is what every input in this project writes.

    Output:
    - timesteps:      (num_timesteps,)
    - box_sizes:      (3, 2) if the box is constant, else (num_timesteps, 3, 2)
    - num_atoms:      int
    - atom_positions: (num_timesteps, num_atoms, 3)
    - atom_types:     (num_atoms,)
    - mol_ids:        (num_atoms,)
    - atom_ids:       (num_atoms,)
    """
    num_timesteps = 0
    num_atoms = 0

    with open(file_path, 'r') as file:
        for line in file:
            if 'ITEM: TIMESTEP' in line:
                num_timesteps += 1
            elif 'ITEM: NUMBER OF ATOMS' in line and num_atoms == 0:
                num_atoms = int(next(file).strip())

    timesteps = np.zeros(num_timesteps, dtype=int)
    box_sizes = np.zeros((num_timesteps, 3, 2), dtype=float)
    atom_positions = np.zeros((num_timesteps, num_atoms, 3), dtype=float)
    atom_types = np.zeros((num_atoms,), dtype=int)
    mol_ids = np.zeros((num_atoms,), dtype=int)
    atom_ids = np.zeros((num_atoms,), dtype=int)

    with open(file_path, 'r') as file:
        line_iter = iter(file)
        ts_idx = -1

        iterator = line_iter if quiet else tqdm(line_iter, desc="Reading trajectory")
        for line in iterator:
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
                    if ts_idx == 0:
                        atom_types[i] = int(atom_data[2])
                        mol_ids[i] = int(atom_data[1])
                        atom_ids[i] = int(atom_data[0])

    is_constant = np.allclose(box_sizes, box_sizes[0], rtol=1e-5, atol=1e-8)
    if is_constant:
        if not quiet:
            print("Box size remains constant throughout the simulation. Great!")
        box_sizes = box_sizes[0]

    return timesteps, box_sizes, num_atoms, atom_positions, atom_types, mol_ids, atom_ids


def wrap_positions_inside_sim_box(positions, simBox):
    """
    Wrap positions into the simulation box, assuming the origin is at the lower vertex.

    Vectorised over frames and atoms; algebraically the same r -= floor(r/L)*L that the
    original per-atom Python loop performed.

    Arguments:
    - positions: (n_frames, n_atoms, 3)
    - simBox:    [[xmin,xmax],[ymin,ymax],[zmin,zmax]]
    """
    simBox = np.asarray(simBox, dtype=float)
    lo = simBox[:, 0]
    L = simBox[:, 1] - simBox[:, 0]
    return lo + (positions - lo) - np.floor((positions - lo) / L) * L


def trajectory_is_unwrapped(atom_positions, box_sizes):
    """
    True if any atom in any frame lies outside its own box bound.

    NOTE: the original implementation indexed atom_positions[:, 0] / [:, 1] / [:, 2],
    which selects ATOMS 0, 1 and 2 across all frames rather than the x, y and z
    columns, and compared the z candidate against the *y* box bounds. It therefore
    decided wrapped-vs-unwrapped from three atoms. It returned the right answer on the
    dumps it was used with, but would silently skip the wrap whenever those particular
    atoms happened to sit inside the box -- and for a slab spanning the full x-y
    cross-section, failing to wrap x/y smears the condensate laterally and corrupts
    every pore radius. All three coordinate columns of all atoms are tested here.
    """
    box_sizes = np.asarray(box_sizes, dtype=float)
    for dim in range(3):
        coord = atom_positions[:, :, dim]
        if np.any(coord < box_sizes[dim, 0]) or np.any(coord > box_sizes[dim, 1]):
            return True
    return False


def load_and_prepare_trajectory(traj_file, quiet=False):
    """
    Read a trajectory, wrap it if needed, and return everything the analysis needs.

    Raises if the box is not constant, since a moving box would invalidate a fixed
    sampling window. All of these runs are NVT, so it never should be.
    """
    (timesteps, box_sizes, num_atoms, atom_positions,
     atom_types, mol_ids, atom_ids) = read_lammps_trajectory(traj_file, quiet=quiet)

    if box_sizes.ndim != 2:
        raise ValueError("Box size varies between frames; a fixed sampling window is "
                         "not valid for this trajectory.")

    if trajectory_is_unwrapped(atom_positions, box_sizes):
        if not quiet:
            print("Trajectory is unwrapped. Wrapping positions back into the "
                  "simulation box for analysis.")
        atom_positions = wrap_positions_inside_sim_box(atom_positions, box_sizes)

    # cKDTree(boxsize=...) requires every coordinate in [0, L). Wrapping gets us there
    # up to floating point, where a value can land exactly on L; nudge those inside.
    lo = box_sizes[:, 0]
    L = box_sizes[:, 1] - box_sizes[:, 0]
    atom_positions = np.clip(atom_positions, lo, lo + L * (1.0 - 1e-12))

    return timesteps, box_sizes, num_atoms, atom_positions, atom_types, mol_ids, atom_ids


def select_frames(num_available, skip_frames=0, num_frames=0):
    """
    Choose frame indices: drop the first `skip_frames`, then take `num_frames` evenly
    spaced from what remains (num_frames <= 0 means take every remaining frame).
    """
    if skip_frames >= num_available:
        raise ValueError(f"skip_frames={skip_frames} discards the whole trajectory "
                         f"({num_available} frames).")
    remaining = np.arange(skip_frames, num_available)
    if 0 < num_frames < remaining.size:
        return np.linspace(remaining[0], remaining[-1], num_frames, dtype=int)
    return remaining


# ----------------------------------------------------------------------------------
# Particle radii
# ----------------------------------------------------------------------------------
def read_particle_radii(potential_file):
    """
    Read sigma_ii / 2 for every atom type from a LAMMPS pair_coeff include file.

    wf/cut is the primary source. A type whose wf/cut sigma is zero is purely repulsive
    by construction and carries its size on its lj/cut line instead, so that is used as
    a fallback -- this is what LAMMPS itself used for those contacts, so the pore
    analysis sees the bead sizes that were actually simulated.

    Both potential files in use are handled by this one reader:
      potential_with_crosslinker.dat             type 61 has sigma = 0 -> lj/cut fallback
                                                 (cavity types 29/35 have eps = 0 but
                                                  sigma != 0, so they resolve from wf/cut)
      potential_with_crosslinker_wfpotential.dat  every type resolves from wf/cut

    KNOWN DISCREPANCY, DELIBERATELY PRESERVED
    -----------------------------------------
    In potential_with_crosslinker.dat (the bond-crosslinking runs), the crosslinker bead
    type 61 was intended to be the size of a Glycine bead, type 2 (sigma 4.695110,
    radius 2.347555 A). The file's `pair_coeff 61 61 lj/cut` sigma does not say that. It
    is 5.576178 (radius 2.788089 A), which is:
      - bit-for-bit equal to the `1 61` cross term, i.e. a copy of type 1's row, and
      - 18.8% larger than Glycine in sigma, 68% larger in excluded volume.
    The intended size is visible in the cross terms instead: `2 61` is exactly sigma_2,
    which only holds if sigma_61 = sigma_2, and inverting all 60 cross terms gives an
    implied sigma_61 whose maximum is exactly Glycine's.

    This reader nevertheless returns the file's value, 2.788089 A, and that is the right
    choice: LAMMPS used it for every 61-61 contact while generating these trajectories,
    so it is the excluded volume that actually shaped the configurations being analysed.
    Substituting the intended radius would measure the pore structure of a system that
    was never simulated. The force-field slip is real but belongs to the simulations, not
    to the analysis, and correcting it means regenerating the trajectories.

    Returns (radii_by_type, source_by_type).
    """
    sigma_by_style = {"wf/cut": {}, "lj/cut": {}}
    with open(potential_file, 'r') as f:
        for line in f:
            parts = line.split()
            if len(parts) > 5 and parts[0] == "pair_coeff" and parts[3] in sigma_by_style:
                if parts[1] == parts[2]:
                    sigma_by_style[parts[3]][int(parts[1])] = float(parts[5])

    radii, source = {}, {}
    for particle_type, sigma_ii in sigma_by_style["wf/cut"].items():
        if sigma_ii == 0.0:
            sigma_ii = sigma_by_style["lj/cut"].get(particle_type, 0.0)
            source[particle_type] = "lj/cut"
        else:
            source[particle_type] = "wf/cut"
        radii[particle_type] = sigma_ii / 2.0
    return radii, source


def build_radius_array(atom_types, radii_by_type, source_by_type=None,
                       potential_file="", quiet=False):
    """
    Map per-type radii onto per-atom radii, failing loudly on a missing or zero radius
    rather than with a bare KeyError once the frame loop is under way -- or, worse, with
    a radius of zero, which would make a whole atom type invisible to the pore sampling
    and silently inflate every measured pore.
    """
    types_present = sorted(set(int(t) for t in atom_types))

    missing = [t for t in types_present if t not in radii_by_type]
    if missing:
        raise KeyError(f"No radius available for atom type(s) {missing} in "
                       f"{potential_file}.")
    zero_radius = [t for t in types_present if radii_by_type[t] == 0.0]
    if zero_radius:
        raise ValueError(f"Atom type(s) {zero_radius} resolved to a radius of zero "
                         f"under both wf/cut and lj/cut in {potential_file}, which "
                         f"would make them invisible to the pore sampling.")

    if not quiet:
        print(f"Particle radii read from {potential_file}")
        for t in types_present:
            n_t = int(np.sum(atom_types == t))
            src = f" (from {source_by_type[t]})" if source_by_type else ""
            print(f"  type {t:3d}: radius = {radii_by_type[t]:.6f} A  "
                  f"({n_t} atoms){src}")

    return np.array([radii_by_type[int(t)] for t in atom_types], dtype=float)


# ----------------------------------------------------------------------------------
# Minimum image geometry
# ----------------------------------------------------------------------------------
def minimum_image(diff, box_lengths):
    """
    Wrap displacement vectors into the primary image, in place.

    `diff` is (..., 3); `box_lengths` is (3,). A non-positive length leaves that
    dimension untouched, which is how a dimension is treated as non-periodic.
    """
    for d in range(3):
        L = box_lengths[d]
        if L > 0:
            diff[..., d] -= np.round(diff[..., d] / L) * L
    return diff


# ----------------------------------------------------------------------------------
# The sphere optimization
# ----------------------------------------------------------------------------------
def robust_sphere_optimization_3d(pos, pos_p, particle_radius=1.0,
                                  bounds=((None, None), (None, None), (None, None)),
                                  box_lengths=(0.0, 0.0, 0.0)):
    """
    Find the largest sphere that encloses the point pos_p without intersecting any
    particle in pos.

    Same problem formulation, optimizer, ftol and maxiter as the validated original,
    with analytic gradients instead of scipy's finite differencing, and with distances
    evaluated under the minimum image convention.

    Arguments:
    - pos:             (N, 3) particle positions
    - pos_p:           (3,) point that must end up inside the sphere
    - particle_radius: scalar or (N,) particle radii
    - bounds:          bounds on the sphere centre
    - box_lengths:     (3,) periodic box lengths; a non-positive entry disables
                       periodicity in that dimension

    Returns the scipy result, with `.radius` and `.center` attached on success.
    """
    pos = np.asarray(pos)
    pos_p = np.asarray(pos_p)
    particle_radius = np.asarray(particle_radius)
    box_lengths = np.asarray(box_lengths, dtype=float)
    periodic = bool(np.any(box_lengths > 0))

    # Cache so fun() and jac() at the same point (SLSQP calls both every iteration)
    # don't each redo the O(N) distance computation.
    cache = {}

    def _nearest(center):
        key = center.tobytes()
        cached = cache.get(key)
        if cached is not None:
            return cached
        diff = pos - center
        if periodic:
            diff = minimum_image(diff, box_lengths)
        distances = np.linalg.norm(diff, axis=1)
        surface_distances = distances - particle_radius
        i_star = np.argmin(surface_distances)
        # The displacement to the nearest particle is returned rather than its index,
        # so the gradients below stay consistent with the minimum-image distance the
        # objective actually used.
        result = (surface_distances[i_star], distances[i_star], diff[i_star].copy())
        cache.clear()
        cache[key] = result
        return result

    def objective(vars):
        f_val, _, _ = _nearest(np.asarray(vars, dtype=float))
        return -f_val

    def objective_grad(vars):
        _, d, diff_star = _nearest(np.asarray(vars, dtype=float))
        if d > 1e-12:
            return diff_star / d
        return np.zeros(3)

    def _point_offset(center):
        # pos_p must be reached under the same metric as everything else.
        off = center - pos_p
        if periodic:
            off = minimum_image(off.copy(), box_lengths)
        return off

    def constraint_enclose_point(vars):
        center = np.asarray(vars, dtype=float)
        f_val, _, _ = _nearest(center)
        return f_val - np.linalg.norm(_point_offset(center))

    def constraint_jac(vars):
        center = np.asarray(vars, dtype=float)
        _, d, diff_star = _nearest(center)
        grad = -diff_star / d if d > 1e-12 else np.zeros(3)
        off = _point_offset(center)
        dist_to_point = np.linalg.norm(off)
        if dist_to_point > 1e-12:
            grad = grad - off / dist_to_point
        return grad

    initial_guess = [pos_p[0], pos_p[1], pos_p[2]]
    constraints = [{'type': 'ineq', 'fun': constraint_enclose_point,
                    'jac': constraint_jac}]

    result = opt.minimize(objective, initial_guess, method='SLSQP',
                          jac=objective_grad,
                          constraints=constraints, bounds=bounds,
                          options={'ftol': 1e-6, 'maxiter': 1000})

    if result.success:
        center = result.x
        diff = pos - center
        if periodic:
            diff = minimum_image(diff, box_lengths)
        result.radius = np.min(np.linalg.norm(diff, axis=1) - particle_radius)
        result.center = center

    return result


def sample_valid_points(n_needed, sampling_space, positions, radii, rng,
                        box_lengths=None):
    """
    Rejection-sample points that do not lie inside any particle.

    Candidates are drawn in batches and screened with a KD-tree, so each candidate is
    only checked against particles that could plausibly contain it. The tree is built
    with `boxsize` when the box is periodic, so a candidate near a lateral face is
    correctly screened against particles across the boundary.
    """
    use_pbc = box_lengths is not None and np.all(np.asarray(box_lengths) > 0)
    if use_pbc:
        box_lengths = np.asarray(box_lengths, dtype=float)
        tree = cKDTree(positions, boxsize=box_lengths)
    else:
        tree = cKDTree(positions)

    max_r = radii.max()
    low = np.array([sampling_space[0], sampling_space[2], sampling_space[4]])
    high = np.array([sampling_space[1], sampling_space[3], sampling_space[5]])

    accepted = []
    batch_size = max(n_needed * 2, 256)
    while len(accepted) < n_needed:
        candidates = rng.uniform(low=low, high=high, size=(batch_size, 3))
        neighbor_lists = tree.query_ball_point(candidates, r=max_r)
        for pt, neighbors in zip(candidates, neighbor_lists):
            if neighbors:
                neighbors = np.asarray(neighbors)
                diff = positions[neighbors] - pt
                if use_pbc:
                    diff = minimum_image(diff, box_lengths)
                if np.any(np.linalg.norm(diff, axis=1) < radii[neighbors]):
                    continue
            accepted.append(pt)
            if len(accepted) >= n_needed:
                break
    return np.array(accepted[:n_needed])


# ----------------------------------------------------------------------------------
# Multiprocessing plumbing
# ----------------------------------------------------------------------------------
# NOTE: macOS and Windows default to the 'spawn' start method, under which a worker is
# a brand-new interpreter that re-imports this module and does NOT inherit globals the
# way a forked child on Linux would. Everything the workers need is therefore passed
# explicitly through Pool(initializer=..., initargs=...), which multiprocessing pickles
# and sends once at worker startup.
_mp_positions = None
_mp_radius = None
_mp_bounds = None
_mp_box_lengths = None


def _pool_initializer(positions, radius, bounds, box_lengths):
    global _mp_positions, _mp_radius, _mp_bounds, _mp_box_lengths
    _mp_positions = positions
    _mp_radius = radius
    _mp_bounds = bounds
    _mp_box_lengths = box_lengths


def _optimize_one_point(pos_p):
    result = robust_sphere_optimization_3d(_mp_positions, pos_p, _mp_radius,
                                           bounds=_mp_bounds,
                                           box_lengths=_mp_box_lengths)
    return result.radius if result.success else None


def optimize_points_parallel(points, positions, radius, bounds, num_workers,
                             box_lengths):
    if num_workers <= 1:
        _pool_initializer(positions, radius, bounds, box_lengths)
        return [_optimize_one_point(p) for p in points]
    # A fresh pool per call so each frame's updated positions reach the workers via
    # initargs; pool startup is milliseconds next to the seconds of optimization work.
    with mp.Pool(processes=num_workers, initializer=_pool_initializer,
                 initargs=(positions, radius, bounds, box_lengths)) as pool:
        return pool.map(_optimize_one_point, points,
                        chunksize=max(1, len(points) // (num_workers * 4)))


# ----------------------------------------------------------------------------------
# The PSD itself
# ----------------------------------------------------------------------------------
def compute_psd(atom_positions, particle_radius, frame_indices, sampling_space,
                box_sizes, num_workers, sampling_interval=1000, bin_width=1.0,
                max_radius=40.0, periodic_xy=True, quiet=False):
    """
    Accumulate the pore size distribution over `frame_indices`.

    Every atom in the system acts as an obstacle, including the immobile active-site
    beads: a test sphere must not be allowed to grow through the wall, or the pores
    beside it would be reported as arbitrarily large. Which REGION gets sampled is the
    caller's job (see psd_windows.py); what counts as an obstacle is everything.

    Returns (bin_centers, histogram_data, num_overflow), where histogram_data is
    (n_frames, n_bins) of per-frame normalised distributions.
    """
    num_bins = int(max_radius / bin_width)
    bin_centers = np.arange(0, max_radius, bin_width) + bin_width / 2.0

    box_lengths = np.zeros(3, dtype=float)
    if periodic_xy:
        # All three dimensions are periodic in the simulation (`boundary p p p`), so
        # minimum image is applied in all three. In practice z makes no difference for
        # any geometry looked at so far -- every sampling window sits far more than
        # max_radius from the z faces, so wrapping z cannot change a nearest-neighbour
        # distance -- but it costs one rounding operation and removes the need to trust
        # that the window is always placed that way. Safe by construction rather than
        # safe by inspection.
        box_lengths[0] = box_sizes[0, 1] - box_sizes[0, 0]
        box_lengths[1] = box_sizes[1, 1] - box_sizes[1, 0]
        box_lengths[2] = box_sizes[2, 1] - box_sizes[2, 0]

    bounds = [(box_sizes[0, 0], box_sizes[0, 1]),
              (box_sizes[1, 0], box_sizes[1, 1]),
              (box_sizes[2, 0], box_sizes[2, 1])]

    # The rejection sampler's KD-tree uses the same three lengths as the optimizer.
    tree_box = box_lengths if periodic_xy else None

    histogram_data = []
    num_overflow = 0
    rng = np.random.default_rng()

    frame_iter = frame_indices if quiet else tqdm(frame_indices)
    for frame_idx in frame_iter:
        particle_positions = atom_positions[frame_idx, :, :]
        pore_size_distribution = np.zeros(num_bins, dtype=int)

        # Keep sampling and optimizing until `sampling_interval` SUCCESSFUL
        # optimizations are collected; a failed optimization does not count towards the
        # total, matching the original semantics.
        num_success = 0
        while num_success < sampling_interval:
            remaining = sampling_interval - num_success
            candidate_points = sample_valid_points(remaining, sampling_space,
                                                   particle_positions, particle_radius,
                                                   rng, box_lengths=tree_box)
            radii = optimize_points_parallel(candidate_points, particle_positions,
                                             particle_radius, bounds, num_workers,
                                             box_lengths)
            for radius in radii:
                if radius is None:
                    continue
                bin_idx = int(radius / bin_width)
                if bin_idx < num_bins:
                    pore_size_distribution[bin_idx] += 1
                else:
                    num_overflow += 1
                num_success += 1

        histogram_data.append(pore_size_distribution / sampling_interval)

    return bin_centers, np.array(histogram_data), num_overflow


def save_psd(out_file, bin_centers, histogram_data, num_overflow, sampling_interval,
             provenance, quiet=False):
    """
    Write the frame-averaged distribution with its SEM, plus a provenance header so a
    .dat file is self-documenting months later.
    """
    mean_distribution = np.mean(histogram_data, axis=0)
    error_distribution = sem(histogram_data, axis=0)

    n_frames = histogram_data.shape[0]
    total_samples = n_frames * sampling_interval

    header_lines = [f"{k}: {v}" for k, v in provenance.items()]
    header_lines.append(f"frames: {n_frames}")
    header_lines.append(f"sampling_interval: {sampling_interval}")
    header_lines.append(f"samples_above_max_radius: {num_overflow} of {total_samples}")
    header_lines.append("Bin_center(Angstroms) Mean_Distribution Error")

    np.savetxt(out_file,
               np.column_stack((bin_centers, mean_distribution, error_distribution)),
               header="\n".join(header_lines), fmt='%.6f')

    if not quiet:
        if num_overflow:
            print(f"WARNING: {num_overflow} of {total_samples} samples "
                  f"({100.0 * num_overflow / total_samples:.2f}%) exceeded max_radius "
                  f"and are absent from the histogram; the saved distribution "
                  f"therefore sums to less than 1.")
        else:
            print("No samples exceeded max_radius.")
        print(f"Wrote {out_file}")

    return mean_distribution, error_distribution
