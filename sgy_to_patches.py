"""
Convert SEG-Y file to 2D .npy patches for Flow Matching interpolation training.

Organization: for each section (grouped by crossline or inline), collect traces,
sort by the other spatial dimension + offset, then sliding-window patches are cut.

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
    p.add_argument("--group_by", type=str, default="crossline",
                   choices=["crossline", "inline"],
                   help="Group traces by crossline or inline to form 2D sections")
    p.add_argument("--patch_size", type=int, default=256)
    p.add_argument("--stride", type=int, default=None)
    p.add_argument("--min_traces_per_section", type=int, default=256,
                   help="Skip sections with fewer traces than this")
    return p.parse_args()


def extract_sections(sgy_path, group_by="crossline", min_traces_per_section=256):
    """
    Read SEG-Y and group traces into 2D sections.
    Returns dict: section_key -> (traces_array, meta_list)
      traces_array: (num_time, num_traces) float32
    """
    TraceField = segyio.TraceField
    primary_field = TraceField.CROSSLINE_3D if group_by == "crossline" else TraceField.INLINE_3D
    secondary_field = TraceField.INLINE_3D if group_by == "crossline" else TraceField.CROSSLINE_3D
    primary_name = group_by

    print(f"Reading {sgy_path} (group_by={group_by}) ...")
    with segyio.open(sgy_path, "r", ignore_geometry=True) as f:
        n_samples = f.samples.size
        n_traces = f.tracecount
        print(f"  Traces: {n_traces}, Samples per trace: {n_samples}")

        # First pass: count traces per section
        grp_counts = defaultdict(int)
        for i in range(n_traces):
            grp_key = f.header[i][primary_field]
            grp_counts[grp_key] += 1

        print(f"  Unique {primary_name}s: {len(grp_counts)}")
        vals = list(grp_counts.values())
        print(f"  Trace count per {primary_name}: min={min(vals)}, max={max(vals)}")

        # Filter sections that meet minimum trace requirement
        valid_keys = sorted([k for k, cnt in grp_counts.items() if cnt >= min_traces_per_section])
        print(f"  {primary_name}s with >= {min_traces_per_section} traces: {len(valid_keys)}")

        # Second pass: collect trace indices per section
        grp_trace_indices = defaultdict(list)
        for i in range(n_traces):
            grp_key = f.header[i][primary_field]
            if grp_key in valid_keys:
                grp_trace_indices[grp_key].append(i)

        # Build sections
        sections = {}
        for grp_key in valid_keys:
            indices = grp_trace_indices[grp_key]
            meta = [(i,
                     f.header[i][secondary_field],
                     f.header[i][TraceField.offset])
                    for i in indices]
            meta.sort(key=lambda x: (x[1], x[2]))
            sorted_indices = [m[0] for m in meta]

            section = np.zeros((n_samples, len(sorted_indices)), dtype=np.float32)
            for j, idx in enumerate(sorted_indices):
                section[:, j] = f.trace[idx]

            sections[grp_key] = section

    print(f"  Built {len(sections)} sections.")
    return sections


def cut_patches(section, patch_size, stride):
    """Yield patches from a 2D section (time × trace)."""
    rows, cols = section.shape
    for i in range(0, rows - patch_size + 1, stride):
        for j in range(0, cols - patch_size + 1, stride):
            yield section[i:i + patch_size, j:j + patch_size]


def main():
    args = parse_args()
    stride = args.stride if args.stride is not None else args.patch_size // 2

    os.makedirs(args.out_dir, exist_ok=True)

    sections = extract_sections(args.sgy_path, args.group_by, args.min_traces_per_section)
    prefix = "xl" if args.group_by == "crossline" else "il"

    total_patches = 0
    total_skipped = 0

    for grp_key, section in sections.items():
        rows, cols = section.shape
        if cols < args.patch_size or rows < args.patch_size:
            print(f"  Skipping {prefix}={grp_key}: shape=({rows},{cols}) < patch_size")
            total_skipped += 1
            continue

        col_nz = np.any(section != 0, axis=0)
        if np.sum(col_nz) < args.patch_size:
            print(f"  Skipping {prefix}={grp_key}: only {np.sum(col_nz)} non-zero columns")
            total_skipped += 1
            continue

        count = 0
        for patch in cut_patches(section, args.patch_size, stride):
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
