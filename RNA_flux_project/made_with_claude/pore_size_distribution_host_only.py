#!/usr/bin/env python3
"""
Pore size distribution of the HOST network as seen from the GUEST monomers, for a
condensate split by an immobile ACTIVE SITE.

Same trajectory and same windowing as pore_size_distribution_active_site.py:

      vapour    |  left condensate  | ACTIVE SITE |  right condensate  |   vapour
    ------------+-------------------+-------------+--------------------+-----------
                ^                   ^             ^                    ^
         90% density crossing   950 - pad      1050 + pad     90% density crossing

Two differences:

1. GUESTS ARE REMOVED. Every atom whose type id is larger than --guest_type_cutoff
   (default 40) is a guest. Guests are not obstacles and do not enter the density
   profile, so the pores reported belong to the host network alone, and the outer
   boundary of each half is the 90% crossing of the HOST number density.

2. GUEST MONOMERS ARE THE PROBES. Instead of drawing random points in the voids, the
   position of every guest bead inside a window is used as a test point and grown into
   the largest sphere that contains it without intersecting any host bead. The result
   is the pore size distribution weighted by where the guests actually are, rather than
   by void volume. Comparing it with the random-point PSD of the same system says
   whether guests sit in typical pores or seek out the larger ones.

   A guest bead whose centre lies inside a host bead (possible, since the potentials
   are soft) cannot be enclosed by any sphere that avoids that host bead. Such probes
   are skipped and counted; the count is printed and written to the output header.

THE ACTIVE SITE
---------------
The production dumps contain only the proteins, not the immobile wall beads. The wall's
z extent is therefore hard-coded below (ACTIVE_SITE_Z_LO / _HI, bead centres) and the
inner window edges are placed --active_site_padding inside it, as in the active-site
script. Because the wall beads are absent, the wall is represented as two FLAT PLANES at
z = 950 and z = 1050 that no sphere may cross. Without them, a sphere grown from a probe
near the wall face would grow into the empty, host-free gap and report a pore that is
really just the absence of the (undumped) wall. The planes sit at the wall bead centres,
so they ignore the wall beads' own radius and slightly over-allow room at the face.

Each frame's histogram is normalised by the number of successful probes in that frame,
and the saved distribution is the mean over frames with its SEM, so every frame carries
equal weight regardless of how many guests it has in the window.

Example:
    python3 pore_size_distribution_host_only.py \
        --traj_file proteins.lammpstrj \
        --potential_file /home/yw9071/scripts/RNA_flux_project/potential_60_particle_types.dat \
        --num_workers 16 --side both
"""

import argparse
import os
import pathlib
import multiprocessing as mp

import numpy as np
from tqdm import tqdm
from scipy.spatial import cKDTree
from scipy.stats import sem

import psd_core as core
import psd_windows as win

# z extent of the active-site wall (bead centres), in Angstroms. The wall is not in the
# dump, so it cannot be measured from the trajectory.
ACTIVE_SITE_Z_LO = 950.0
ACTIVE_SITE_Z_HI = 1050.0


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--traj_file", type=str, required=True)
    p.add_argument("--potential_file", type=str, required=True,
                   help="The pair_coeff include file the simulation itself used")
    p.add_argument("--guest_type_cutoff", type=int, default=40,
                   help="Atoms with type id > this value are guests (default 40)")
    p.add_argument("--num_workers", type=int,
                   default=max(1, (os.cpu_count() or 1) - 1))
    p.add_argument("--skip_frames", type=int, default=0)
    p.add_argument("--num_frames", type=int, default=0,
                   help="Evenly spaced frames to analyse after skipping; 0 means all")
    p.add_argument("--max_probes_per_frame", type=int, default=0,
                   help="Randomly subsample at most this many guest probes per frame "
                        "and side; 0 means use every guest bead in the window")
    p.add_argument("--threshold", type=float, default=0.90,
                   help="Outer boundary: host number density at least this fraction of "
                        "plateau (default 0.90)")
    p.add_argument("--profile_bin_width", type=float, default=5.0)
    p.add_argument("--min_gap", type=float, default=0.0,
                   help="Merge threshold-crossing regions separated by less than this. "
                        "OFF by default. If enabled it must stay well below the "
                        "active-site thickness so the two halves are never merged")
    p.add_argument("--min_width", type=float, default=30.0,
                   help="Discard regions narrower than this (removes density-layering "
                        "islands next to the wall)")
    p.add_argument("--active_site_padding", type=float, default=10.0,
                   help="Fixed inset from the active-site bead centres "
                        f"({ACTIVE_SITE_Z_LO} / {ACTIVE_SITE_Z_HI}); default 10.0")
    p.add_argument("--max_radius", type=float, default=40.0)
    p.add_argument("--bin_width", type=float, default=1.0)
    p.add_argument("--no_periodic_xy", action="store_true",
                   help="Disable the minimum image convention. Only for reproducing "
                        "the old, laterally-biased results")
    p.add_argument("--side", choices=("left", "right", "both"), default="both")
    p.add_argument("--out_tag", type=str, default="host_only_guest_probes",
                   help="Tag used in the output filenames, which get _left / _right")
    p.add_argument("--out_dir", type=str, default=None)
    return p.parse_args()


def probes_inside_obstacles(probes, positions, radii, box_lengths):
    """Boolean mask of probes whose centre lies inside any obstacle particle."""
    use_pbc = np.all(box_lengths > 0)
    tree = cKDTree(positions, boxsize=box_lengths if use_pbc else None)
    inside = np.zeros(len(probes), dtype=bool)
    for i, neighbors in enumerate(tree.query_ball_point(probes, r=radii.max())):
        if neighbors:
            neighbors = np.asarray(neighbors)
            diff = positions[neighbors] - probes[i]
            if use_pbc:
                diff = core.minimum_image(diff, box_lengths)
            inside[i] = np.any(np.linalg.norm(diff, axis=1) < radii[neighbors])
    return inside


def compute_guest_probe_psd(atom_positions, host_idx, guest_idx, host_radius,
                            frame_indices, z_window, box_sizes, num_workers,
                            bin_width=1.0, max_radius=40.0, periodic=True,
                            max_probes=0, wall_slab=None, rng=None):
    """
    PSD using guest bead positions inside z_window as the probe points, with host beads
    and the flat wall slab (if given) as the only obstacles.

    Returns (bin_centers, histogram_data, stats), where histogram_data is
    (n_frames_used, n_bins) of per-frame distributions normalised by that frame's
    successful probes, and stats counts what happened to every probe.
    """
    rng = rng or np.random.default_rng()
    num_bins = int(max_radius / bin_width)
    bin_centers = np.arange(0, max_radius, bin_width) + bin_width / 2.0

    box_lengths = np.zeros(3, dtype=float)
    if periodic:
        box_lengths = (box_sizes[:, 1] - box_sizes[:, 0]).astype(float)
    bounds = [(box_sizes[d, 0], box_sizes[d, 1]) for d in range(3)]

    z_lo, z_hi = z_window
    stats = {"probes_in_window": 0, "probes_used": 0, "inside_host_bead": 0,
             "optimization_failed": 0, "above_max_radius": 0, "frames_without_probes": 0}
    histogram_data = []

    for frame_idx in tqdm(frame_indices):
        host = atom_positions[frame_idx, host_idx, :]
        guests = atom_positions[frame_idx, guest_idx, :]
        probes = guests[(guests[:, 2] >= z_lo) & (guests[:, 2] <= z_hi)]
        stats["probes_in_window"] += len(probes)

        if max_probes > 0 and len(probes) > max_probes:
            probes = probes[rng.choice(len(probes), max_probes, replace=False)]

        inside = probes_inside_obstacles(probes, host, host_radius, box_lengths) \
            if len(probes) else np.zeros(0, dtype=bool)
        stats["inside_host_bead"] += int(inside.sum())
        probes = probes[~inside]

        if len(probes) == 0:
            stats["frames_without_probes"] += 1
            continue
        stats["probes_used"] += len(probes)

        radii = core.optimize_points_parallel(probes, host, host_radius, bounds,
                                              num_workers, box_lengths, wall_slab)

        counts = np.zeros(num_bins, dtype=int)
        num_success = 0
        for radius in radii:
            if radius is None:
                stats["optimization_failed"] += 1
                continue
            num_success += 1
            bin_idx = int(radius / bin_width)
            if bin_idx < num_bins:
                counts[bin_idx] += 1
            else:
                stats["above_max_radius"] += 1

        if num_success:
            histogram_data.append(counts / num_success)

    return bin_centers, np.array(histogram_data), stats


def save_probe_psd(out_file, bin_centers, histogram_data, stats, provenance):
    mean_distribution = np.mean(histogram_data, axis=0)
    error_distribution = sem(histogram_data, axis=0)

    header_lines = [f"{k}: {v}" for k, v in provenance.items()]
    header_lines.append(f"frames_with_probes: {histogram_data.shape[0]}")
    header_lines += [f"{k}: {v}" for k, v in stats.items()]
    header_lines.append("Bin_center(Angstroms) Mean_Distribution Error")

    np.savetxt(out_file,
               np.column_stack((bin_centers, mean_distribution, error_distribution)),
               header="\n".join(header_lines), fmt='%.6f')
    print(f"Wrote {out_file}")
    return mean_distribution


def main():
    args = parse_args()
    traj_path = pathlib.Path(args.traj_file).resolve()
    out_dir = pathlib.Path(args.out_dir).resolve() if args.out_dir else traj_path.parent

    # ---- trajectory -------------------------------------------------------------
    (timesteps, box_sizes, num_atoms, atom_positions,
     atom_types, mol_ids, atom_ids) = core.load_and_prepare_trajectory(str(traj_path))

    # ---- split host / guest -----------------------------------------------------
    # Index arrays rather than sliced copies: the full position array is already the
    # largest object in memory, and each frame is sliced on the fly instead.
    guest_mask = atom_types > args.guest_type_cutoff
    host_idx = np.flatnonzero(~guest_mask)
    guest_idx = np.flatnonzero(guest_mask)
    if host_idx.size == 0:
        raise ValueError(f"No atoms with type <= {args.guest_type_cutoff}; no host.")
    if guest_idx.size == 0:
        raise ValueError(f"No atoms with type > {args.guest_type_cutoff}; there are no "
                         f"guest monomers to use as probes.")
    print(f"\nHost : {host_idx.size} beads, {np.unique(mol_ids[host_idx]).size} "
          f"molecules, types {sorted(set(int(t) for t in atom_types[host_idx]))}")
    print(f"Guest: {guest_idx.size} beads, {np.unique(mol_ids[guest_idx]).size} "
          f"molecules, types {sorted(set(int(t) for t in atom_types[guest_idx]))} "
          f"(removed as obstacles, used as probes)")

    frame_indices = core.select_frames(len(timesteps), args.skip_frames,
                                       args.num_frames)
    print(f"\nAnalysing {frame_indices.size} of {len(timesteps)} frames "
          f"(timesteps {timesteps[frame_indices[0]]} to "
          f"{timesteps[frame_indices[-1]]}).")

    # ---- radii (host only: the only obstacles) ----------------------------------
    radii_by_type, source_by_type = core.read_particle_radii(args.potential_file)
    host_radius = core.build_radius_array(atom_types[host_idx], radii_by_type,
                                          source_by_type, args.potential_file)

    # ---- the wall (hard-coded) --------------------------------------------------
    z_cav_lo, z_cav_hi = ACTIVE_SITE_Z_LO, ACTIVE_SITE_Z_HI
    print(f"Active site z extent (hard-coded, bead centres): {z_cav_lo:.2f} to "
          f"{z_cav_hi:.2f}; modelled as flat walls no sphere may cross")

    # ---- host profile -----------------------------------------------------------
    z, density, plateau, regions = win.contiguous_condensate_regions(
        atom_positions, frame_indices, box_sizes,
        threshold_frac=args.threshold, bin_width=args.profile_bin_width,
        mask=~guest_mask, min_gap=args.min_gap, min_width=args.min_width,
        label="host only, number density")

    if args.min_gap >= (z_cav_hi - z_cav_lo):
        raise ValueError(
            f"--min_gap ({args.min_gap} A) is at least as large as the active-site "
            f"thickness ({z_cav_hi - z_cav_lo:.2f} A), so the two condensate halves "
            f"would be merged into one region.")

    pad = args.active_site_padding
    crosscheck = win.super_gaussian_crosscheck(
        z, density, exclude_range=(z_cav_lo - pad, z_cav_hi + pad))
    if crosscheck is not None:
        delta = 100.0 * (crosscheck["amplitude"] - plateau) / plateau
        print(f"\n  cross-check, super-Gaussian fit with the cavity excluded "
              f"({z_cav_lo - pad:.1f}-{z_cav_hi + pad:.1f}):")
        print(f"    amplitude {crosscheck['amplitude']:.6f} atoms/A^3 "
              f"({delta:+.1f}% vs the median plateau {plateau:.6f})")
        if abs(delta) > 10.0:
            print(f"    WARNING: the two plateau estimates disagree by more than 10%.")
    else:
        print("\n  cross-check: super-Gaussian fit unavailable "
              "(utils.find_interfaces did not import or did not converge)")

    if len(regions) != 2:
        raise ValueError(
            f"Expected two bulk regions (one condensate half either side of the active "
            f"site), found {len(regions)}: {regions}. The window cannot be set "
            f"automatically -- inspect the density profile before proceeding.")

    left_region, right_region = regions
    if not (left_region[1] < z_cav_hi and right_region[0] > z_cav_lo
            and left_region[0] < z_cav_lo and right_region[1] > z_cav_hi):
        raise ValueError(
            f"The hard-coded active site (z {z_cav_lo:.2f}-{z_cav_hi:.2f}) does not sit "
            f"between the two condensate regions {left_region} and {right_region}.")

    if left_region[0] <= z[0] + 1e-9 or right_region[1] >= z[-1] - 1e-9:
        print("WARNING: a condensate region reaches a z box face. The system may wrap "
              "through the periodic boundary, in which case the outer window edge is "
              "wrong.")

    windows = {
        "left": (left_region[0], z_cav_lo - pad),
        "right": (z_cav_hi + pad, right_region[1]),
    }

    # ---- audit the chosen edges -------------------------------------------------
    threshold_density = args.threshold * plateau
    print(f"\n--- chosen windows (padding {pad:.2f} A at the active-site face) ---")
    for side, (lo, hi) in windows.items():
        if hi <= lo:
            raise ValueError(f"{side} window is empty: z {lo:.2f} to {hi:.2f}. The "
                             f"active-site padding may exceed the half thickness.")
        d_lo = win.density_at(z, density, lo)
        d_hi = win.density_at(z, density, hi)
        print(f"  {side:5s}: z {lo:8.2f} to {hi:8.2f}  (width {hi - lo:7.2f} A)")
        print(f"         host density at edges: {d_lo / plateau * 100:5.1f}% and "
              f"{d_hi / plateau * 100:5.1f}% of plateau")
        wall_edge_density = d_hi if side == "left" else d_lo
        if wall_edge_density < threshold_density:
            print(f"         NOTE: the active-site-side edge sits at "
                  f"{wall_edge_density / plateau * 100:.1f}% of plateau, below the "
                  f"{args.threshold * 100:.0f}% criterion used on the outer side; "
                  f"consider increasing --active_site_padding.")

    sides = ("left", "right") if args.side == "both" else (args.side,)
    periodic = not args.no_periodic_xy
    rng = np.random.default_rng()

    for side in sides:
        z_lo, z_hi = windows[side]
        print(f"\n=== {side.upper()} : z [{z_lo:.2f}, {z_hi:.2f}] ===")
        print(f"Minimum image: {periodic}")

        bin_centers, histogram_data, stats = compute_guest_probe_psd(
            atom_positions, host_idx, guest_idx, host_radius, frame_indices,
            (z_lo, z_hi), box_sizes, args.num_workers, bin_width=args.bin_width,
            max_radius=args.max_radius, periodic=periodic,
            max_probes=args.max_probes_per_frame, wall_slab=(z_cav_lo, z_cav_hi),
            rng=rng)

        print(f"guest probes in window : {stats['probes_in_window']} over "
              f"{frame_indices.size} frames")
        print(f"  inside a host bead   : {stats['inside_host_bead']} (skipped)")
        print(f"  optimization failed  : {stats['optimization_failed']}")
        print(f"  above max_radius     : {stats['above_max_radius']}")
        if histogram_data.size == 0:
            print(f"No usable guest probes in the {side} window; nothing written.")
            continue

        provenance = {
            "geometry": f"active site, {side} condensate half, HOST ONLY, "
                        f"guest monomers as probes",
            "trajectory": str(traj_path),
            "potential_file": args.potential_file,
            "guests": f"type > {args.guest_type_cutoff} ({guest_idx.size} beads), "
                      f"removed as obstacles and used as probe points",
            "outer_boundary": f"{args.threshold * 100:.0f}% of plateau HOST number "
                              f"density",
            "inner_boundary": f"hard-coded active site {z_cav_lo}-{z_cav_hi} +/- "
                              f"{pad} A padding",
            "wall_obstacle": f"flat planes at z = {z_cav_lo} and {z_cav_hi}",
            "plateau_number_density_per_A3": f"{plateau:.6f} (host only)",
            "profile_bin_width_A": f"{args.profile_bin_width}",
            "z_window": f"{z_lo:.4f} {z_hi:.4f}",
            "minimum_image": str(periodic),
            "max_probes_per_frame": str(args.max_probes_per_frame or "all"),
            "max_radius_A": f"{args.max_radius}",
            "normalisation": "per frame by successful probes, then mean over frames",
            "timestep_range": f"{timesteps[frame_indices[0]]} to "
                              f"{timesteps[frame_indices[-1]]}",
        }

        out_file = out_dir / f"psd_{args.out_tag}_{side}.dat"
        mean = save_probe_psd(str(out_file), bin_centers, histogram_data, stats,
                              provenance)

        norm = mean.sum()
        cdf = np.cumsum(mean) / norm
        print(f"mean pore radius   = {(bin_centers * mean).sum() / norm:.3f} A")
        print(f"median pore radius = {np.interp(0.5, cdf, bin_centers):.3f} A")
        print(f"P(R > 10 A)        = {mean[bin_centers > 10].sum() / norm:.4f}")


if __name__ == "__main__":
    mp.freeze_support()
    main()
