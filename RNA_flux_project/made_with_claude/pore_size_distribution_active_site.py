#!/usr/bin/env python3
"""
Pore size distribution for a condensate split by an immobile ACTIVE SITE.

Geometry (cf0.1 as the reference case, box z 0-2000):

      vapour    |  left condensate  | ACTIVE SITE |  right condensate  |   vapour
    ------------+-------------------+-------------+--------------------+-----------
                ^                   ^             ^                    ^
         90% density crossing   950 - pad      1050 + pad     90% density crossing

The active site is a slab-shaped wall of immobile beads spanning the full x-y
cross-section, so it cuts the condensate into two disconnected halves that must be
analysed separately -- hence one output file per side.

The two boundary types are treated differently, deliberately:

- The OUTER boundary of each half is a liquid-vapour interface, located by the 90%
  number-density criterion. A fixed padding there was shown to reach into partly-dilute
  material and inflate the large-pore tail (see psd_windows.py).
- The INNER boundary is the active-site face. Its position is already known exactly from
  the wall beads, and because the wall is immobile the density drops abruptly and can
  oscillate next to it -- so a density threshold is the wrong tool. A FIXED PADDING is
  used instead, measured from the outermost wall bead CENTRE, which reproduces the
  earlier convention on this side (950 - 10 = 940, 1050 + 10 = 1060) and keeps the new
  results comparable to the old ones there.

The wall beads are excluded from the density profile -- they would otherwise put a large
spike in the middle of it and corrupt the plateau estimate -- but they are KEPT as
obstacles in the sphere optimization, since a test sphere must not grow through the wall.

Example:
    python3 pore_size_distribution_active_site.py \
        --traj_file equilibrate_crosslinked_condensate.lammpstrj \
        --potential_file /home/yw9071/scripts/RNA_flux_project/potential_with_crosslinker.dat \
        --num_workers 16 --side both
"""

import argparse
import os
import pathlib
import multiprocessing as mp

import numpy as np

import psd_core as core
import psd_windows as win


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--traj_file", type=str, required=True)
    p.add_argument("--potential_file", type=str, required=True,
                   help="The pair_coeff include file the simulation itself used")
    p.add_argument("--num_workers", type=int,
                   default=max(1, (os.cpu_count() or 1) - 1))
    p.add_argument("--skip_frames", type=int, default=0)
    p.add_argument("--num_frames", type=int, default=0,
                   help="Evenly spaced frames to analyse after skipping; 0 means all")
    p.add_argument("--sampling_interval", type=int, default=1000)
    p.add_argument("--threshold", type=float, default=0.90,
                   help="Outer boundary: number density at least this fraction of "
                        "plateau (default 0.90)")
    p.add_argument("--profile_bin_width", type=float, default=5.0)
    p.add_argument("--min_gap", type=float, default=0.0,
                   help="Merge threshold-crossing regions separated by less than this. "
                        "OFF by default. If enabled it must stay well below the "
                        "active-site thickness (~100 A) so the two condensate halves "
                        "are never merged into one")
    p.add_argument("--min_width", type=float, default=30.0,
                   help="Discard regions narrower than this. Needed here: density "
                        "layering next to the immobile wall produces isolated islands a "
                        "few Angstroms wide (cf0.1 has them at z~941 and z~1058) which "
                        "would otherwise make this script see four regions, not two")
    p.add_argument("--active_site_padding", type=float, default=10.0,
                   help="Fixed inset from the outermost active-site bead CENTRE "
                        "(default 10.0, matching the earlier convention)")
    p.add_argument("--cavity_mol", type=int, default=1,
                   help="Molecule id of the active site (the inputs use `group cavity "
                        "molecule == 1`)")
    p.add_argument("--cavity_type_range", type=int, nargs=2, default=(21, 40),
                   help="Inclusive atom type range the active site is expected to use; "
                        "used only to cross-check the molecule-id selection")
    p.add_argument("--max_radius", type=float, default=40.0)
    p.add_argument("--bin_width", type=float, default=1.0)
    p.add_argument("--no_periodic_xy", action="store_true",
                   help="Disable the minimum image convention in x and y. Only for "
                        "reproducing the old, laterally-biased results")
    p.add_argument("--xy_margin", type=float, default=0.0,
                   help="Inset the x/y sampling range. Should stay 0 with minimum image "
                        "enabled; the old convention used 10.0")
    p.add_argument("--side", choices=("left", "right", "both"), default="both")
    p.add_argument("--out_tag", type=str, default="active_site",
                   help="Tag used in the output filenames, which get _left / _right")
    p.add_argument("--out_dir", type=str, default=None)
    return p.parse_args()


def identify_cavity(mol_ids, atom_types, cavity_mol, type_range):
    """
    Select the active site by molecule id, then cross-check it against the expected
    atom type range.

    Two independent facts should agree: the inputs define the wall as
    `group cavity molecule == 1`, and the wall is built from purely-repulsive types in
    21-40. If they disagree, the convention has drifted and a silent wrong answer is
    the worst outcome -- so this raises instead.
    """
    cavity_mask = (mol_ids == cavity_mol)
    if not np.any(cavity_mask):
        raise ValueError(f"No atoms found with molecule id {cavity_mol}; this "
                         f"trajectory may not have an active site. For a plain slab, "
                         f"use pore_size_distribution_slab.py.")

    lo, hi = type_range
    cavity_types = sorted(set(int(t) for t in atom_types[cavity_mask]))
    other_types = sorted(set(int(t) for t in atom_types[~cavity_mask]))

    stray = [t for t in cavity_types if not (lo <= t <= hi)]
    if stray:
        raise ValueError(f"Atoms in molecule {cavity_mol} have type(s) {stray} outside "
                         f"the expected active-site range {lo}-{hi}. Check which "
                         f"molecule id the active site actually is.")
    leaked = [t for t in other_types if lo <= t <= hi]
    if leaked:
        raise ValueError(f"Non-cavity atoms carry active-site type(s) {leaked} in range "
                         f"{lo}-{hi}, so type and molecule id disagree about what the "
                         f"wall is. Resolve before trusting the window.")

    print(f"Active site: molecule {cavity_mol}, {int(cavity_mask.sum())} beads, "
          f"types {cavity_types}")
    print(f"Condensate : {int((~cavity_mask).sum())} beads, types {other_types}")
    return cavity_mask


def main():
    args = parse_args()
    traj_path = pathlib.Path(args.traj_file).resolve()
    out_dir = pathlib.Path(args.out_dir).resolve() if args.out_dir else traj_path.parent

    # ---- trajectory -------------------------------------------------------------
    (timesteps, box_sizes, num_atoms, atom_positions,
     atom_types, mol_ids, atom_ids) = core.load_and_prepare_trajectory(str(traj_path))

    frame_indices = core.select_frames(len(timesteps), args.skip_frames,
                                       args.num_frames)
    print(f"\nAnalysing {frame_indices.size} of {len(timesteps)} frames "
          f"(timesteps {timesteps[frame_indices[0]]} to "
          f"{timesteps[frame_indices[-1]]}), {args.sampling_interval} samples each.")

    # ---- radii (all atoms, wall included: it is an obstacle) --------------------
    radii_by_type, source_by_type = core.read_particle_radii(args.potential_file)
    particle_radius = core.build_radius_array(atom_types, radii_by_type,
                                              source_by_type, args.potential_file)

    # ---- the wall ---------------------------------------------------------------
    cavity_mask = identify_cavity(mol_ids, atom_types, args.cavity_mol,
                                  tuple(args.cavity_type_range))
    z_cav_lo, z_cav_hi = win.active_site_extent(atom_positions, frame_indices,
                                                cavity_mask)
    print(f"Active site z extent (bead centres): {z_cav_lo:.2f} to {z_cav_hi:.2f} "
          f"(thickness {z_cav_hi - z_cav_lo:.2f} A)")

    # ---- condensate profile, wall excluded --------------------------------------
    z, density, plateau, regions = win.contiguous_condensate_regions(
        atom_positions, frame_indices, box_sizes,
        threshold_frac=args.threshold, bin_width=args.profile_bin_width,
        mask=~cavity_mask, min_gap=args.min_gap, min_width=args.min_width,
        label="condensate only, number density")

    if args.min_gap >= (z_cav_hi - z_cav_lo):
        raise ValueError(
            f"--min_gap ({args.min_gap} A) is at least as large as the active-site "
            f"thickness ({z_cav_hi - z_cav_lo:.2f} A), so the two condensate halves "
            f"would be merged into one region.")

    # Independent plateau estimate from the super-Gaussian fit, with the cavity and its
    # layering zone excluded from the fit. Reported only -- it never sets a window.
    # The exclusion margin reuses --active_site_padding rather than adding another knob.
    pad = args.active_site_padding
    crosscheck = win.super_gaussian_crosscheck(
        z, density, exclude_range=(z_cav_lo - pad, z_cav_hi + pad))
    if crosscheck is not None:
        delta = 100.0 * (crosscheck["amplitude"] - plateau) / plateau
        print(f"\n  cross-check, super-Gaussian fit with the cavity excluded "
              f"({z_cav_lo - pad:.1f}-{z_cav_hi + pad:.1f}):")
        print(f"    amplitude {crosscheck['amplitude']:.6f} atoms/A^3 "
              f"({delta:+.1f}% vs the median plateau {plateau:.6f})")
        print(f"    outer interfaces {crosscheck['left']:.2f} / "
              f"{crosscheck['right']:.2f} (steepest-gradient points, not the "
              f"{args.threshold * 100:.0f}% crossings used for the window)")
        if abs(delta) > 10.0:
            print(f"    WARNING: the two plateau estimates disagree by more than 10%. "
                  f"Check that the exclusion range actually covers the cavity -- an "
                  f"unmasked fit on this geometry reads ~25% low.")
    else:
        print("\n  cross-check: super-Gaussian fit unavailable "
              "(utils.find_interfaces did not import or did not converge)")

    if len(regions) != 2:
        raise ValueError(
            f"Expected two bulk regions (one condensate half either side of the active "
            f"site), found {len(regions)}: {regions}. The window cannot be set "
            f"automatically -- inspect the density profile before proceeding.")

    left_region, right_region = regions
    if not (left_region[1] < z_cav_lo < z_cav_hi < right_region[0]):
        raise ValueError(
            f"The active site (z {z_cav_lo:.2f}-{z_cav_hi:.2f}) does not sit between "
            f"the two condensate regions {left_region} and {right_region}. The geometry "
            f"is not what this script assumes.")

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
        print(f"         density at edges: {d_lo / plateau * 100:5.1f}% and "
              f"{d_hi / plateau * 100:5.1f}% of plateau")
        wall_edge_density = d_hi if side == "left" else d_lo
        if wall_edge_density < threshold_density:
            print(f"         NOTE: the active-site-side edge sits at "
                  f"{wall_edge_density / plateau * 100:.1f}% of plateau, below the "
                  f"{args.threshold * 100:.0f}% criterion used on the outer side. The "
                  f"fixed padding of {pad:.2f} A does not clear the wall's depletion "
                  f"layer here; consider increasing --active_site_padding.")

    sides = ("left", "right") if args.side == "both" else (args.side,)
    periodic_xy = not args.no_periodic_xy

    for side in sides:
        z_lo, z_hi = windows[side]
        sampling_space = np.array([
            box_sizes[0, 0] + args.xy_margin, box_sizes[0, 1] - args.xy_margin,
            box_sizes[1, 0] + args.xy_margin, box_sizes[1, 1] - args.xy_margin,
            z_lo, z_hi,
        ], dtype=float)

        print(f"\n=== {side.upper()} : z [{z_lo:.2f}, {z_hi:.2f}] ===")
        print(f"Minimum image in x,y: {periodic_xy}")

        bin_centers, histogram_data, num_overflow = core.compute_psd(
            atom_positions, particle_radius, frame_indices, sampling_space, box_sizes,
            args.num_workers, sampling_interval=args.sampling_interval,
            bin_width=args.bin_width, max_radius=args.max_radius,
            periodic_xy=periodic_xy)

        provenance = {
            "geometry": f"active site, {side} condensate half",
            "trajectory": str(traj_path),
            "potential_file": args.potential_file,
            "outer_boundary": f"{args.threshold * 100:.0f}% of plateau number density",
            "inner_boundary": f"active-site bead centre +/- {pad} A fixed padding",
            "active_site_mol": str(args.cavity_mol),
            "active_site_z_extent": f"{z_cav_lo:.4f} to {z_cav_hi:.4f}",
            "plateau_number_density_per_A3": f"{plateau:.6f} (median of in-slab bins)",
            "plateau_crosscheck_super_gaussian": (
                f"{crosscheck['amplitude']:.6f} (cavity excluded "
                f"{z_cav_lo - pad:.1f}-{z_cav_hi + pad:.1f})"
                if crosscheck else "unavailable"),
            "profile_bin_width_A": f"{args.profile_bin_width}",
            "sampling_space": " ".join(f"{v:.4f}" for v in sampling_space),
            "minimum_image_xy": str(periodic_xy),
            "max_radius_A": f"{args.max_radius}",
            "timestep_range": f"{timesteps[frame_indices[0]]} to "
                              f"{timesteps[frame_indices[-1]]}",
        }

        out_file = out_dir / f"psd_{args.out_tag}_{side}.dat"
        core.save_psd(str(out_file), bin_centers, histogram_data, num_overflow,
                      args.sampling_interval, provenance)

        mean = np.mean(histogram_data, axis=0)
        norm = mean.sum()
        mean_R = (bin_centers * mean).sum() / norm
        cdf = np.cumsum(mean) / norm
        print(f"mean pore radius   = {mean_R:.3f} A")
        print(f"median pore radius = {np.interp(0.5, cdf, bin_centers):.3f} A")
        print(f"P(R > 10 A)        = {mean[bin_centers > 10].sum() / norm:.4f}")


if __name__ == "__main__":
    mp.freeze_support()
    main()
