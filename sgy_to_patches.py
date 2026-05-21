"""
Convert SEG-Y file to 2D .npy patches for Flow Matching interpolation training.

Organization: group traces into sections by crossline (or inline), keep original
trace order within each section, then sliding-window patches are cut top-to-bottom,
left-to-right.

Usage:
    python sgy_to_patches.py \
        --sgy_path /path/to/file.sgy \
        --out_dir /path/to/patches/label_256 \
        --group_by inline \
        --patch_size 256 --stride 128
"""
import argparse
import os
import numpy as np
import segyio
from collections import defaultdict


def parse_args():
    p = argparse.ArgumentParser(description="Convert SEG-Y to training patches")
    p.add_argument("--sgy_path", type=str, required=True)
    p.add_argument("--out_dir", type=str, required=True)
    p.add_argument("--group_by", type=str, default="inline",
                   choices=["crossline", "inline"],
                   help="Group traces by crossline or inline to form 2D sections")
    p.add_argument("--patch_size", type=int, nargs='+', default=[256],
                   help="Patch size: single int for square, or 'H W' for rectangle (time x trace)")
    p.add_argument("--stride", type=int, nargs='+', default=None,
                   help="Stride: single int or 'H W'. Default: half of patch_size")
    p.add_argument("--min_traces_per_section", type=int, default=256,
                   help="Skip sections with fewer traces than this")
    p.add_argument("--clip_low", type=float, default=1.0,
                   help="Low percentile for clipping outliers (default 1)")
    p.add_argument("--clip_high", type=float, default=99.0,
                   help="High percentile for clipping outliers (default 99)")
    p.add_argument("--stats_sample", type=int, default=2000,
                   help="Number of random traces to sample for global statistics "
                        "(0 = use all traces)")
    return p.parse_args()


def extract_sections(sgy_path, group_by="crossline", min_traces_per_section=256,
                     clip_low=1.0, clip_high=99.0, stats_sample=2000):
    """
    Read SEG-Y, group traces into 2D sections, apply global normalization.

    Normalization steps (applied to raw traces before cutting patches):
      1. Clip amplitudes to [percentile(clip_low), percentile(clip_high)]
      2. Divide by global max absolute value

    Returns:
        sections: dict section_key -> (num_time, num_traces) float32  (normalized)
        norm_stats: dict with clip_low_val, clip_high_val, global_max_abs
    """
    TraceField = segyio.TraceField
    primary_field = TraceField.CROSSLINE_3D if group_by == "crossline" else TraceField.INLINE_3D
    primary_name = group_by

    print(f"Reading {sgy_path} (group_by={group_by}) ...")
    with segyio.open(sgy_path, "r", ignore_geometry=True) as f:
        n_samples = f.samples.size
        n_traces = f.tracecount
        print(f"  Traces: {n_traces}, Samples per trace: {n_samples}")

        # ── Step 1: count traces per section + sample for global stats ──
        grp_counts = defaultdict(int)
        amp_samples = []

        sample_size = min(n_traces, max(stats_sample, 1)) if stats_sample > 0 else n_traces
        sample_indices = set(np.random.choice(n_traces, sample_size, replace=False))

        for i in range(n_traces):
            grp_key = f.header[i][primary_field]
            grp_counts[grp_key] += 1
            if i in sample_indices:
                amp_samples.append(f.trace[i])

        print(f"  Unique {primary_name}s: {len(grp_counts)}")
        vals = list(grp_counts.values())
        print(f"  Trace count per {primary_name}: min={min(vals)}, max={max(vals)}")

        # ── Step 2: compute global clip thresholds ──
        all_amps = np.concatenate([t for t in amp_samples])
        clip_low_val = float(np.percentile(all_amps, clip_low))
        clip_high_val = float(np.percentile(all_amps, clip_high))
        print(f"  Global amplitude stats (from {sample_size:,} sampled traces):")
        print(f"    raw min={all_amps.min():.4f}, raw max={all_amps.max():.4f}")
        print(f"    clip [{clip_low}%, {clip_high}%] = [{clip_low_val:.4f}, {clip_high_val:.4f}]")

        # ── Step 3: compute global max_abs after clipping ──
        global_max_abs = max(abs(clip_low_val), abs(clip_high_val))
        if global_max_abs < 1e-8:
            global_max_abs = 1.0
        print(f"    global max abs = {global_max_abs:.4f}")
        del all_amps, amp_samples  # free memory

        # ── Step 4: filter sections ──
        valid_keys = sorted([k for k, cnt in grp_counts.items() if cnt >= min_traces_per_section])
        print(f"  {primary_name}s with >= {min_traces_per_section} traces: {len(valid_keys)}")

        # Second pass: collect trace indices
        grp_trace_indices = defaultdict(list)
        for i in range(n_traces):
            grp_key = f.header[i][primary_field]
            if grp_key in valid_keys:
                grp_trace_indices[grp_key].append(i)

        # ── Step 5: build sections with normalized traces ──
        sections = {}
        for grp_key in valid_keys:
            indices = grp_trace_indices[grp_key]

            section = np.zeros((n_samples, len(indices)), dtype=np.float32)
            for j, idx in enumerate(indices):
                trace = f.trace[idx].astype(np.float32)
                trace = np.clip(trace, clip_low_val, clip_high_val)
                trace = trace / global_max_abs
                section[:, j] = trace

            sections[grp_key] = section

    print(f"  Built {len(sections)} normalized sections.")
    norm_stats = {
        "clip_low_val": clip_low_val,
        "clip_high_val": clip_high_val,
        "global_max_abs": global_max_abs,
    }
    return sections, norm_stats


def cut_patches(section, patch_h, patch_w, stride_h, stride_w):
    """Yield patches from a 2D section (time × trace)."""
    rows, cols = section.shape
    for i in range(0, rows - patch_h + 1, stride_h):
        for j in range(0, cols - patch_w + 1, stride_w):
            yield section[i:i + patch_h, j:j + patch_w]


def main():
    args = parse_args()

    # Unpack patch_size: single int → square, two ints → (H, W) = (time, trace)
    ps = args.patch_size
    patch_h = ps[0]
    patch_w = ps[1] if len(ps) > 1 else ps[0]

    if args.stride is not None:
        st = args.stride
        stride_h = st[0]
        stride_w = st[1] if len(st) > 1 else st[0]
    else:
        stride_h = patch_h // 2
        stride_w = patch_w // 2

    print(f"Patch size: {patch_h} (time) × {patch_w} (trace), stride: {stride_h}×{stride_w}")

    os.makedirs(args.out_dir, exist_ok=True)

    sections, norm_stats = extract_sections(
        args.sgy_path, args.group_by, args.min_traces_per_section,
        args.clip_low, args.clip_high, args.stats_sample,
    )
    prefix = "xl" if args.group_by == "crossline" else "il"

    # Save normalization stats alongside patches for reproducibility
    np.savez(os.path.join(args.out_dir, "_normalization_stats.npz"), **norm_stats)

    total_patches = 0
    total_skipped = 0

    for grp_key, section in sections.items():
        rows, cols = section.shape
        if cols < patch_w or rows < patch_h:
            print(f"  Skipping {prefix}={grp_key}: shape=({rows},{cols}) < patch ({patch_h},{patch_w})")
            total_skipped += 1
            continue

        col_nz = np.any(section != 0, axis=0)
        if np.sum(col_nz) < patch_w:
            print(f"  Skipping {prefix}={grp_key}: only {np.sum(col_nz)} non-zero columns, need {patch_w}")
            total_skipped += 1
            continue

        count = 0
        for patch in cut_patches(section, patch_h, patch_w, stride_h, stride_w):
            if np.max(np.abs(patch)) < 1e-8:
                continue
            fname = f"{prefix}_{grp_key}_{count:05d}.npy"
            np.save(os.path.join(args.out_dir, fname), patch)
            count += 1
            total_patches += 1

        if count > 0:
            print(f"  {prefix}={grp_key}: shape=({rows},{cols}), patches={count}")

    print(f"\nDone: {total_patches} patches saved to {args.out_dir}")
    print(f"  Skipped {total_skipped} sections (too narrow/small)")


if __name__ == "__main__":
    main()