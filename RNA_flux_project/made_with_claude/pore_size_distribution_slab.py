#!/usr/bin/env python3
"""
Pore size distribution for a condensate in SLAB geometry with no active site.

The condensate is bounded by two liquid-vapour interfaces and nothing else, so there is
exactly one region of bulk material and both of its boundaries are located by the
density criterion (see psd_windows.py for why a fixed padding was abandoned).

For a condensate split by an immobile active site, use
pore_size_distribution_active_site.py instead -- that geometry has two bulk regions and
one boundary per region whose position is known from the wall beads rather than from the
density profile.

Example:
    python3 pore_size_distribution_slab.py \
        --traj_file result.lammpstrj \
        --potential_file /home/yw9071/scripts/RNA_flux_project/potential_with_crosslinker_wfpotential.dat \
        --num_workers 16
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
                   help="The pair_coeff include file the simulation itself used, so the "
                        "analysis sees the bead sizes that were actually simulated")
    p.add_argument("--num_workers", type=int,
                   default=max(1, (os.cpu_count() or 1) - 1))
    p.add_argument("--skip_frames", type=int, default=0,
                   help="Leading frames to discard (default 0: these runs equilibrate "
                        "in a separate job, so the production dump is usable from "
                        "frame 0)")
    p.add_argument("--num_frames", type=int, default=0,
                   help="Evenly spaced frames to analyse after skipping; 0 means all")
    p.add_argument("--sampling_interval", type=int, default=1000,
                   help="Successful sphere optimizations per frame")
    p.add_argument("--threshold", type=float, default=0.90,
                   help="Sample where the number density is at least this fraction of "
                        "the plateau (default 0.90)")
    p.add_argument("--profile_bin_width", type=float, default=5.0,
                   help="z bin width for the density profile, in Angstroms")
    p.add_argument("--min_gap", type=float, default=0.0,
                   help="Merge threshold-crossing regions separated by less than this. "
                        "OFF by default: it existed to repair fragmentation caused by a "
                        "biased plateau estimator, which is now fixed at source, and "
                        "merging is the operation that could hide a genuine void")
    p.add_argument("--min_width", type=float, default=30.0,
                   help="Discard regions narrower than this. Removes sub-bead-scale "
                        "slivers from density layering; cannot conceal a real break")
    p.add_argument("--max_radius", type=float, default=40.0,
                   help="Largest pore radius the histogram can represent")
    p.add_argument("--bin_width", type=float, default=1.0,
                   help="Pore radius bin width, in Angstroms")
    p.add_argument("--no_periodic_xy", action="store_true",
                   help="Disable the minimum image convention in x and y. Only for "
                        "reproducing the old, laterally-biased results")
    p.add_argument("--xy_margin", type=float, default=0.0,
                   help="Inset the x/y sampling range by this much. Should stay 0 with "
                        "minimum image enabled; the old convention used 10.0 because it "
                        "had no periodic images and had to keep test points away from "
                        "the faces")
    p.add_argument("--z_range", type=float, nargs=2, default=None,
                   help="Override the automatic z window with an explicit LO HI. The "
                        "density profile is still computed and reported, so the chosen "
                        "window can be judged against it. Intended for reproducing an "
                        "earlier run's window exactly, not for production")
    p.add_argument("--out_tag", type=str, default="slab",
                   help="Tag used in the output filename")
    p.add_argument("--out_dir", type=str, default=None,
                   help="Where to write the .dat (default: alongside the trajectory)")
    return p.parse_args()


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

    # ---- radii ------------------------------------------------------------------
    radii_by_type, source_by_type = core.read_particle_radii(args.potential_file)
    particle_radius = core.build_radius_array(atom_types, radii_by_type,
                                              source_by_type, args.potential_file)

    # ---- sampling window --------------------------------------------------------
    # Every atom counts towards the density here: a slab run has no immobile wall to
    # exclude, so the profile is simply the condensate profile.
    z, density, plateau, regions = win.contiguous_condensate_regions(
        atom_positions, frame_indices, box_sizes,
        threshold_frac=args.threshold, bin_width=args.profile_bin_width,
        min_gap=args.min_gap, min_width=args.min_width,
        label="all atoms, number density")

    # Independent estimate from the project's existing super-Gaussian fit
    # (utils.find_interfaces). Reported only -- it never sets the window. A slab is the
    # one geometry that fit is valid for, since it assumes a single centred dense phase.
    crosscheck = win.super_gaussian_crosscheck(z, density)
    if crosscheck is not None:
        delta = 100.0 * (crosscheck["amplitude"] - plateau) / plateau
        print(f"\n  cross-check, super-Gaussian fit (utils.find_interfaces):")
        print(f"    amplitude {crosscheck['amplitude']:.6f} atoms/A^3 "
              f"({delta:+.1f}% vs the median plateau {plateau:.6f})")
        print(f"    interfaces {crosscheck['left']:.2f} / {crosscheck['right']:.2f} "
              f"(steepest-gradient points, not the 90% crossings used for the window)")
        if abs(delta) > 10.0:
            print(f"    WARNING: the two plateau estimates disagree by more than 10%. "
                  f"That usually means the profile is not a single centred slab, in "
                  f"which case the fit is the unreliable one -- inspect the profile.")
    else:
        print("\n  cross-check: super-Gaussian fit unavailable "
              "(utils.find_interfaces did not import or did not converge)")

    if args.z_range is not None:
        z_lo, z_hi = float(args.z_range[0]), float(args.z_range[1])
        if z_hi <= z_lo:
            raise ValueError(f"--z_range must be increasing, got {z_lo} {z_hi}.")
        d_lo = win.density_at(z, density, z_lo)
        d_hi = win.density_at(z, density, z_hi)
        print(f"\nOVERRIDE: using --z_range {z_lo:.2f} to {z_hi:.2f} instead of the "
              f"automatic window.")
        print(f"  density at edges: {d_lo / plateau * 100:5.1f}% and "
              f"{d_hi / plateau * 100:5.1f}% of plateau "
              f"(criterion would be {args.threshold * 100:.0f}%)")
        if len(regions) == 1:
            print(f"  automatic window would have been "
                  f"{regions[0][0]:.2f} to {regions[0][1]:.2f}")
    else:
        if len(regions) != 1:
            raise ValueError(
                f"Expected exactly one bulk region for a slab geometry, found "
                f"{len(regions)}: {regions}. If this system has an active site, use "
                f"pore_size_distribution_active_site.py. If the condensate has broken "
                f"up or the slab straddles the periodic z boundary, the window cannot "
                f"be set automatically -- inspect the profile first, or pass "
                f"--z_range explicitly.")

        z_lo, z_hi = regions[0]
        if z_lo <= z[0] + 1e-9 or z_hi >= z[-1] - 1e-9:
            print("WARNING: the bulk region reaches a z box face. The slab may wrap "
                  "through the periodic boundary, in which case this window is wrong.")

    sampling_space = np.array([
        box_sizes[0, 0] + args.xy_margin, box_sizes[0, 1] - args.xy_margin,
        box_sizes[1, 0] + args.xy_margin, box_sizes[1, 1] - args.xy_margin,
        z_lo, z_hi,
    ], dtype=float)

    periodic_xy = not args.no_periodic_xy
    print(f"\nSampling space: x [{sampling_space[0]:.2f}, {sampling_space[1]:.2f}], "
          f"y [{sampling_space[2]:.2f}, {sampling_space[3]:.2f}], "
          f"z [{sampling_space[4]:.2f}, {sampling_space[5]:.2f}]")
    print(f"Slab thickness sampled: {z_hi - z_lo:.2f} A")
    print(f"Minimum image in x,y: {periodic_xy}")

    # ---- the PSD ----------------------------------------------------------------
    bin_centers, histogram_data, num_overflow = core.compute_psd(
        atom_positions, particle_radius, frame_indices, sampling_space, box_sizes,
        args.num_workers, sampling_interval=args.sampling_interval,
        bin_width=args.bin_width, max_radius=args.max_radius,
        periodic_xy=periodic_xy)

    provenance = {
        "geometry": "slab (two liquid-vapour interfaces, no active site)",
        "trajectory": str(traj_path),
        "potential_file": args.potential_file,
        "window_source": ("explicit --z_range override" if args.z_range is not None
                          else "automatic, from the density criterion"),
        "density_criterion": f"{args.threshold * 100:.0f}% of plateau number density",
        "plateau_number_density_per_A3": f"{plateau:.6f} (median of in-slab bins)",
        "plateau_crosscheck_super_gaussian": (f"{crosscheck['amplitude']:.6f}"
                                              if crosscheck else "unavailable"),
        "profile_bin_width_A": f"{args.profile_bin_width}",
        "sampling_space": " ".join(f"{v:.4f}" for v in sampling_space),
        "minimum_image_xy": str(periodic_xy),
        "max_radius_A": f"{args.max_radius}",
        "timestep_range": f"{timesteps[frame_indices[0]]} to "
                          f"{timesteps[frame_indices[-1]]}",
    }

    out_file = out_dir / f"psd_{args.out_tag}.dat"
    core.save_psd(str(out_file), bin_centers, histogram_data, num_overflow,
                  args.sampling_interval, provenance)

    mean = np.mean(histogram_data, axis=0)
    norm = mean.sum()
    mean_R = (bin_centers * mean).sum() / norm
    cdf = np.cumsum(mean) / norm
    print(f"\nmean pore radius   = {mean_R:.3f} A")
    print(f"median pore radius = {np.interp(0.5, cdf, bin_centers):.3f} A")
    print(f"P(R > 10 A)        = {mean[bin_centers > 10].sum() / norm:.4f}")


if __name__ == "__main__":
    mp.freeze_support()
    main()
