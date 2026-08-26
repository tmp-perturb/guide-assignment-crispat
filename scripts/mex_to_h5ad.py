#!/usr/bin/env python3
"""
MEX trio → AnnData .h5ad conversion for crispat input.

Reads the standard merged MEX trio from 04_extraction_mex/{tool}/,
extracts lane/batch info from barcode suffix (-L{NN}),
writes gRNA_counts.h5ad alongside the MEX files.

Usage:
  python mex_to_h5ad.py --input 04_extraction_mex/cellranger/
  python mex_to_h5ad.py --input 04_extraction_mex/ham/
  python mex_to_h5ad.py --input 04_extraction_mex/simpleaf_k15/
  python mex_to_h5ad.py --all   # process all 3 tools at once
"""

import argparse, os, sys, gzip, time
import numpy as np
import scipy.io, scipy.sparse
from scipy.sparse import csr_matrix
import anndata as ad


def load_mex_trio(mex_dir):
    """Load merged MEX trio, return (mtx, barcodes, features)."""
    t0 = time.time()
    with gzip.open(f"{mex_dir}/merged_matrix.mtx.gz", "rt") as f:
        mtx = scipy.io.mmread(f).tocsr()
    with gzip.open(f"{mex_dir}/merged_barcodes.tsv.gz", "rt") as f:
        barcodes = [line.strip() for line in f]
    with gzip.open(f"{mex_dir}/merged_features.tsv.gz", "rt") as f:
        features = [line.strip().split("\t")[0] for line in f]
    elapsed = time.time() - t0
    return mtx, barcodes, features, elapsed


def extract_batch(barcode):
    """Extract lane number from barcode suffix -L{NN}. Returns int."""
    # Format: NNNNNNNNNNNNNNNN-LNN
    try:
        lane = int(barcode.rsplit("-L", 1)[-1])
    except (ValueError, IndexError):
        lane = 0
    return lane


def build_anndata(mtx, barcodes, features, tool_name):
    """Build AnnData from MEX components, adding obs['batch']. """
    t0 = time.time()

    # Ensure CSR for row access
    if not isinstance(mtx, csr_matrix):
        mtx = mtx.tocsr()

    adata = ad.AnnData(mtx, dtype=np.int32)
    adata.obs_names = barcodes
    adata.var_names = features

    # Extract batch from barcode suffix
    batches = [extract_batch(bc) for bc in barcodes]
    adata.obs["batch"] = batches
    adata.obs["batch"] = adata.obs["batch"].astype("category")

    n_batches = len(set(batches))

    elapsed = time.time() - t0
    print(f"  AnnData built: {adata.shape[0]:,} cells × {adata.shape[1]:,} guides, "
          f"{n_batches} batches [{elapsed:.1f}s]")
    return adata


def convert_one(mex_dir, force=False):
    """Convert one MEX directory to h5ad."""
    tool = os.path.basename(mex_dir.rstrip("/"))
    out_path = os.path.join(mex_dir, "gRNA_counts.h5ad")

    if os.path.exists(out_path) and not force:
        print(f"[{tool}] SKIP — {out_path} already exists (use --force to overwrite)")
        return out_path

    print(f"[{tool}] Loading MEX from {mex_dir} …")
    mtx, barcodes, features, load_t = load_mex_trio(mex_dir)
    nnz = mtx.nnz
    sparsity = (1 - nnz / (mtx.shape[0] * mtx.shape[1])) * 100
    print(f"  Loaded: {mtx.shape[0]:,} cells × {mtx.shape[1]:,} guides  "
          f"nnz={nnz:,}  sparsity={sparsity:.2f}%  [{load_t:.1f}s]")

    adata = build_anndata(mtx, barcodes, features, tool)

    # Add total_counts for methods that need it (2-beta)
    adata.obs["total_counts"] = np.array(adata.X.sum(axis=1)).flatten()

    t0 = time.time()
    adata.write(out_path)
    write_t = time.time() - t0
    size_mb = os.path.getsize(out_path) / (1024**2)
    print(f"  Saved: {out_path}  ({size_mb:.0f} MB) [{write_t:.1f}s]")
    return out_path


def main():
    parser = argparse.ArgumentParser(
        description="MEX trio → AnnData .h5ad for crispat"
    )
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--input", help="Path to a single MEX directory")
    group.add_argument("--all", action="store_true", help="Process all 3 tools")
    parser.add_argument("--base", default="04_extraction_mex",
                        help="Base directory for --all mode (default: 04_extraction_mex)")
    parser.add_argument("--force", action="store_true",
                        help="Overwrite existing .h5ad files")
    args = parser.parse_args()

    T0 = time.time()

    if args.all:
        base = args.base.rstrip("/")
        tools = ["cellranger", "ham", "simpleaf_k15"]
        for tool in tools:
            mex_dir = os.path.join(base, tool)
            if not os.path.isdir(mex_dir):
                print(f"WARNING: {mex_dir} not found, skipping")
                continue
            convert_one(mex_dir, force=args.force)
            print()
    else:
        convert_one(args.input, force=args.force)

    print(f"Total time: {time.time()-T0:.0f}s")


if __name__ == "__main__":
    main()
