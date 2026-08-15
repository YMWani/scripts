"""
Sampling-window determination for the pore size distribution analysis.

WHY THIS EXISTS
---------------
The PSD must be measured in the BULK of the condensate. The earlier convention placed
the sampling boundary a fixed 10 A inside each half-density interface, and that turned
out to be far too shallow: for the interaction-crosslinked slab, the local density 10 A
inside the half-density point is only 64% of the plateau, so those test points sit in
partly-dilute material and grow into large spurious pores.

Measured on that system (41 frames, 1000 samples each), moving from the 10 A convention
to a bulk-only window left the core of the distribution alone -- identical 4.5 A modal
bin, median shifting only 4.9 -> 4.7 A -- but cut the tail out from under it:

    99th percentile   22.36 -> 14.36 A
    P(R > 15 A)       0.0374 -> 0.0092
    samples > 40 A    120/41000 -> 0
    mean R            6.30 -> 5.64 A  (18 sigma)

A 22 A-radius sphere has room for roughly 540 beads at these bead sizes and cannot
exist in material at 0.515 g/cm^3, which is what identifies that tail as dilute phase
rather than network porosity.

The fix is a criterion that ADAPTS to each system instead of a fixed distance.
Interfacial width varies with crosslinker fraction, so a fixed 10 A padding admits a
different amount of dilute phase in every run -- meaning the artifact is not a constant
offset and can manufacture or mask a trend in pore size versus crosslinker fraction.
A "within X% of this system's own plateau" criterion cannot do that.

CONVENTIONS SETTLED FOR THIS PROJECT
------------------------------------
- Threshold: 90% of the plateau.
- Plateau: the MEDIAN of the bins inside the condensate, not the mean of the densest
  bins (which selects on noise and biases high) and not a fit amplitude (which assumes
  a single centred dense phase and so breaks on the active-site geometry). See
  estimate_plateau. For slabs a super-Gaussian fit is reported alongside as a
  cross-check.
- Density: NUMBER density, not mass density. Pore geometry is about bead count and
  excluded volume, and the two criteria only agree if composition is uniform across the
  interface -- which is not guaranteed if the crosslinker partitions to the surface.
- The profile is RECOMPUTED from the exact frames that enter the histogram, rather than
  read from the run's density_all.dat. That keeps the window self-consistent with the
  frames analysed, does not care whether the ave/chunk window matched the dump window,
  and works for runs where no density file was written or its bins were too coarse.
- Liquid-vapour interfaces get the 90% criterion. The ACTIVE-SITE FACE gets a fixed
  padding instead: the active site is immobile, so the density drops abruptly there and
  can oscillate next to it, which makes a density threshold an unreliable way to locate
  a boundary whose position is already known exactly from the wall beads themselves.
"""

import numpy as np


def number_density_profile(atom_positions, frame_indices, box_sizes, bin_width=5.0,
                           mask=None):
    """
    Number density profile along z, averaged over `frame_indices`.

    Arguments:
    - atom_positions: (n_frames, n_atoms, 3), already wrapped
    - frame_indices:  which frames to average over
    - box_sizes:      (3, 2)
    - bin_width:      z bin width in Angstroms
    - mask:           optional (n_atoms,) boolean selecting which atoms count towards
                      the density. Used to exclude the immobile active-site beads, which
                      would otherwise put a large spike in the middle of the profile and
                      corrupt the plateau estimate.

    Returns (z_centers, density) with density in atoms / A^3.
    """
    zlo, zhi = box_sizes[2, 0], box_sizes[2, 1]
    Lx = box_sizes[0, 1] - box_sizes[0, 0]
    Ly = box_sizes[1, 1] - box_sizes[1, 0]

    num_bins = int(round((zhi - zlo) / bin_width))
    edges = np.linspace(zlo, zhi, num_bins + 1)
    bin_volume = Lx * Ly * (edges[1] - edges[0])

    counts = np.zeros(num_bins, dtype=float)
    for frame_idx in frame_indices:
        z = atom_positions[frame_idx, :, 2]
        if mask is not None:
            z = z[mask]
        counts += np.histogram(z, bins=edges)[0]

    density = counts / (len(frame_indices) * bin_volume)
    z_centers = 0.5 * (edges[:-1] + edges[1:])
    return z_centers, density


def estimate_plateau(density, inside_frac=0.8, smooth_bins=3):
    """
    Plateau density, taken as the MEDIAN of the bins inside the condensate.

    WHY NOT THE MEAN OF THE N DENSEST BINS (the previous implementation): selecting the
    densest bins selects on noise, so the estimate is biased upward, and the bias grows
    as statistics shrink. Measured on the interaction slab against the 41-frame answer:

        frames   mean of top 8      median of in-slab bins
             2   0.003197 (+7.8%)   0.002999 (+4.4%)
             5   0.003037 (+5.9%)   0.002934 (+2.1%)
            10   0.003093 (+7.8%)   0.002907 (+1.2%)
            41   0.002966 (ref)     0.002873 (ref)

    That upward bias is not harmless: it raises the 90% threshold above ordinary bulk
    density fluctuations, so perfectly normal interior bins fall below it and one
    condensate fragments into several apparent regions. At 2 frames the top-8 mean
    produced SIX regions where the in-slab median produced one.

    WHY NOT A FIT: a super-Gaussian fit is available and is a good estimator for a plain
    slab -- its amplitude agrees with an independent tanh fit to ~0.2% -- but it assumes
    a single dense phase centred in the box. With an active site there are two dense
    phases and a hole between them, and the fit spans the lot: on cf0.1 it returned an
    amplitude 28% BELOW the true plateau because it averaged the empty cavity into the
    flat top. The median below is geometry-agnostic, so one estimator serves both
    scripts. See super_gaussian_crosscheck() for the fit, reported alongside for slabs.

    "Inside" is defined against a lightly smoothed maximum rather than against the
    plateau itself, which would be circular and can ratchet upward on iteration. The
    result is insensitive to the exact fraction: 0.7 and 0.8 agree to better than 0.1%,
    while 0.9 starts picking up noise.
    """
    density = np.asarray(density, dtype=float)
    if density.size == 0:
        raise ValueError("Empty density profile.")

    kernel = np.ones(smooth_bins) / smooth_bins
    smoothed_max = float(np.convolve(density, kernel, mode='same').max())
    inside = density > inside_frac * smoothed_max

    # Degenerate profile (e.g. a condensate only a couple of bins wide): fall back to
    # the densest handful rather than returning a median of nothing.
    if int(inside.sum()) < 3:
        n_top = min(8, density.size)
        return float(np.sort(density)[-n_top:].mean())

    return float(np.median(density[inside]))


def super_gaussian_crosscheck(z, density, exclude_range=None,
                              scripts_dir="/home/yw9071/scripts/RNA_flux_project"):
    """
    Independent plateau and interface estimate from the project's existing super-Gaussian
    fit, utils.find_interfaces. Reported alongside the median estimator, never used to
    set the window.

    exclude_range: (low, high) z range dropped before fitting. REQUIRED for the
    active-site geometry, where an empty cavity sits in the middle of the box. Fitted
    without it, the single flat-topped super-Gaussian averages the hole into its
    amplitude and lands far below the true plateau. Measured on cf0.1 at 31 frames,
    against an in-slab median of 0.003615:

        no exclusion            A = 0.002720   -24.8%
        cavity exactly          A = 0.003512    -2.8%
        cavity +/- 10 A         A = 0.003630    +0.4%
        cavity +/- 20 A         A = 0.003657    +1.2%

    A margin around the wall is what makes it work: the density oscillates for ~10 A
    either side of the active site, so excluding only the cavity leaves those perturbed
    bins in the fit. The callers pass cavity +/- active_site_padding, reusing the padding
    already chosen for the sampling window rather than introducing another knob.

    Note the margin IS a free parameter worth roughly 3% in the answer, which is the
    reason this stays a cross-check and the median stays primary: the median has no such
    choice to make, and cannot fail to converge.

    This masks the arrays before calling utils.find_interfaces, which reproduces exactly
    what the `exclude_range` variant in crosslinker_fraction_sweep.ipynb does -- that
    function computes its weighting from the surviving points, and the fitted curve still
    spans the full coordinate range because only interior bins are removed.

    Returns a dict with 'amplitude', 'left', 'right', or None if unavailable.
    """
    try:
        import sys
        if scripts_dir not in sys.path:
            sys.path.insert(0, scripts_dir)
        from utils import find_interfaces
    except Exception:
        return None

    z = np.asarray(z, dtype=float)
    density = np.asarray(density, dtype=float)

    if exclude_range is not None:
        low, high = exclude_range
        keep = ~((z >= low) & (z <= high))
        if int(keep.sum()) < 8:
            return None
        z, density = z[keep], density[keep]

    try:
        left, right, _fine, fitted = find_interfaces(z, density)
    except Exception:
        return None

    return {"amplitude": float(np.max(fitted)),
            "left": float(left),
            "right": float(right)}


def find_regions(z, density, threshold):
    """
    All contiguous z-regions where density >= threshold.

    Edges are placed by linear interpolation between the bracketing bin centres, so the
    returned boundary is where the profile crosses the threshold rather than the centre
    of the first bin above it.

    Returns a list of (z_lo, z_hi) tuples, ordered by increasing z.
    """
    above = density >= threshold
    if not np.any(above):
        return []

    regions = []
    # Indices where the boolean mask switches, giving inclusive [start, end] bin runs.
    padded = np.concatenate(([False], above, [False]))
    switches = np.flatnonzero(padded[1:] != padded[:-1])
    for start, stop in zip(switches[0::2], switches[1::2]):
        i0, i1 = start, stop - 1  # first and last bin in this run

        # Lower edge: interpolate between bin i0-1 (below threshold) and i0 (above).
        if i0 == 0:
            z_lo = z[0]
        else:
            z_lo = np.interp(threshold, [density[i0 - 1], density[i0]],
                             [z[i0 - 1], z[i0]])
        # Upper edge: interpolate between bin i1 (above) and i1+1 (below).
        if i1 == len(z) - 1:
            z_hi = z[-1]
        else:
            z_hi = np.interp(threshold, [density[i1 + 1], density[i1]],
                             [z[i1 + 1], z[i1]])
        regions.append((float(z_lo), float(z_hi)))
    return regions


def consolidate_regions(regions, min_gap=0.0, min_width=30.0, quiet=False):
    """
    Merge regions separated by a gap narrower than `min_gap`, then drop regions narrower
    than `min_width`.

    THE TWO OPERATIONS HAVE DIFFERENT STANDING, hence the asymmetric defaults.

    min_gap = 0 (OFF by default). Merging existed to repair a single condensate that had
    fragmented into several apparent regions, which happened because the old plateau
    estimator was biased high and pushed the threshold above ordinary bulk fluctuations.
    That is fixed at source in estimate_plateau, so merging is no longer needed: the
    2-frame case that produced six fragments now produces one region with no merging at
    all. It is left available but off, because merging is the dangerous direction -- it
    is the operation that could glue across a genuine void and hide real structure.

    min_width = 30 A (ON by default). Dropping slivers does a different and still
    necessary job. Next to the immobile active site the density oscillates: on cf0.1
    there is a dip at z = 937.5 and a peak at 942.5, and again a peak at 1057.5 with a
    dip at 1062.5. Those produce isolated islands 3.3 A and 2.6 A wide, which would
    otherwise make the active-site script see four regions instead of two. A 30 A floor
    is well below any meaningful condensate fragment and well above these, and unlike
    merging it cannot conceal a break -- it can only discard fragments too small to
    support pore analysis at all.

    Note the slivers do not change the active-site windows either way, since the inner
    edge is set by the fixed padding at the wall rather than by the density region. They
    only affect the region COUNT, which the entry points assert on.

    Both operations are reported, because a window that was silently altered is worse
    than one that was never checked.
    """
    if not regions:
        return []

    merged = [list(regions[0])]
    merges = []
    for lo, hi in regions[1:]:
        gap = lo - merged[-1][1]
        if gap < min_gap:
            merges.append((merged[-1][1], lo, gap))
            merged[-1][1] = hi
        else:
            merged.append([lo, hi])

    kept, dropped = [], []
    for lo, hi in merged:
        if (hi - lo) >= min_width:
            kept.append((float(lo), float(hi)))
        else:
            dropped.append((float(lo), float(hi)))

    if not quiet and (merges or dropped):
        print(f"  consolidation (min_gap {min_gap:.1f} A, min_width {min_width:.1f} A):")
        for a, b, gap in merges:
            print(f"    merged across a {gap:.2f} A gap at z {a:.2f}-{b:.2f}")
        for lo, hi in dropped:
            print(f"    dropped sliver z {lo:.2f}-{hi:.2f} "
                  f"(width {hi - lo:.2f} A)")
        print(f"  -> {len(kept)} region(s) after consolidation:")
        for i, (lo, hi) in enumerate(kept):
            print(f"       region {i}: z = {lo:8.2f} to {hi:8.2f}   "
                  f"(width {hi - lo:7.2f} A)")

    return kept


def density_at(z, density, z_query):
    """Linearly interpolated density at an arbitrary z. Used to audit a chosen edge."""
    return float(np.interp(z_query, z, density))


def report_profile(z, density, plateau, threshold_frac, regions, label="condensate",
                   inside_frac=0.8):
    """
    Print an auditable summary of how the window was chosen, so a log file records the
    reasoning and not just the final numbers.
    """
    threshold = threshold_frac * plateau
    n_inside = int(np.sum(np.asarray(density) > inside_frac * float(
        np.convolve(np.asarray(density), np.ones(3) / 3, mode='same').max())))
    print(f"\n--- z density profile ({label}) ---")
    print(f"plateau number density : {plateau:.6f} atoms/A^3 "
          f"(median of the {n_inside} bins inside the condensate)")
    print(f"threshold              : {threshold_frac * 100:.0f}% of plateau = "
          f"{threshold:.6f} atoms/A^3")
    print(f"regions above threshold: {len(regions)}")
    for i, (lo, hi) in enumerate(regions):
        print(f"  region {i}: z = {lo:8.2f} to {hi:8.2f}   (width {hi - lo:7.2f} A)")


def contiguous_condensate_regions(atom_positions, frame_indices, box_sizes,
                                  threshold_frac=0.90, bin_width=5.0, mask=None,
                                  min_gap=0.0, min_width=30.0,
                                  label="condensate", quiet=False):
    """
    Convenience wrapper: profile -> plateau -> regions -> consolidation, with the audit
    report.

    Returns (z, density, plateau, regions).
    """
    if len(frame_indices) < 10:
        print(f"WARNING: only {len(frame_indices)} frames are being averaged into the "
              f"density profile. The plateau and the interface crossings will be noisy, "
              f"so the window may be off by a bin or two. Fine for a smoke test, not "
              f"for a production number.")

    z, density = number_density_profile(atom_positions, frame_indices, box_sizes,
                                       bin_width=bin_width, mask=mask)
    plateau = estimate_plateau(density)
    regions = find_regions(z, density, threshold_frac * plateau)
    if not quiet:
        report_profile(z, density, plateau, threshold_frac, regions, label=label)
    regions = consolidate_regions(regions, min_gap=min_gap, min_width=min_width,
                                 quiet=quiet)
    return z, density, plateau, regions


def active_site_extent(atom_positions, frame_indices, cavity_mask):
    """
    z extent of the immobile active site, as (z_min, z_max) over bead CENTRES.

    The active site is not integrated, so this is constant in time; it is measured over
    all analysed frames anyway and checked for drift, because a moving "immobile" wall
    would mean the input did not do what it was assumed to do.
    """
    z_min = np.inf
    z_max = -np.inf
    for frame_idx in frame_indices:
        z = atom_positions[frame_idx, cavity_mask, 2]
        z_min = min(z_min, float(z.min()))
        z_max = max(z_max, float(z.max()))

    z_min_first = float(atom_positions[frame_indices[0], cavity_mask, 2].min())
    z_max_first = float(atom_positions[frame_indices[0], cavity_mask, 2].max())
    drift = max(abs(z_min - z_min_first), abs(z_max - z_max_first))
    if drift > 1e-6:
        print(f"WARNING: the active site moved by {drift:.4f} A over the analysed "
              f"frames. It is supposed to be immobile -- check that the input excluded "
              f"it from the integrators.")
    return z_min, z_max
