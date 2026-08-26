#!/usr/bin/env python3
"""
crispat ga_2beta — batch-level 2-Beta mixture model runner with monitoring.

Usage:
  python run_crispat_2beta.py \
      --h5ad 04_extraction_mex/cellranger/gRNA_counts.h5ad \
      --output 07_2beta_crispat/cellranger/ \
      --tool cellranger \
      --n-iter 500
"""

import os, sys, time, json, platform, argparse
import numpy as np
import pandas as pd

# ── Monitoring helpers ──
try:
    import psutil
    _HAS_PSUTIL = True
except ImportError:
    _HAS_PSUTIL = False

def get_memory_rss_mb():
    if _HAS_PSUTIL:
        return psutil.Process().memory_info().rss / (1024 * 1024)
    return -1.0

def get_system_info():
    info = {
        "hostname": platform.node(),
        "platform": platform.platform(),
        "python_version": sys.version,
    }
    import torch; info["torch_version"] = torch.__version__
    import pyro;  info["pyro_version"] = pyro.__version__
    info["cpu_count_logical"] = os.cpu_count()
    try:
        import subprocess
        out = subprocess.check_output(["lscpu", "-p=cpu,core"], text=True, timeout=5)
        cores = set()
        for line in out.strip().split("\n"):
            if line.startswith("#"): continue
            parts = line.split(",")
            if len(parts) == 2: cores.add(parts[1])
        info["cpu_count_physical"] = len(cores)
    except Exception:
        info["cpu_count_physical"] = None
    if _HAS_PSUTIL:
        mem = psutil.virtual_memory()
        info["total_ram_gb"] = round(mem.total / (1024**3), 1)
    return info


def main():
    parser = argparse.ArgumentParser(
        description="crispat ga_2beta — batch-level mixture model runner"
    )
    parser.add_argument("--h5ad", required=True, help="Path to gRNA_counts.h5ad")
    parser.add_argument("--output", required=True, help="Output directory")
    parser.add_argument("--tool", required=True, help="Tool label")
    parser.add_argument("--n-iter", type=int, default=500)
    parser.add_argument("--umi-threshold", type=int, default=0)
    args = parser.parse_args()

    monitor = {
        "method": "ga_2beta",
        "tool": args.tool,
        "parameters": {
            "n_iter": args.n_iter,
            "batch_list": None,
            "add_UMI_counts": True,
            "UMI_threshold": args.umi_threshold,
        },
        "stages": {},
        "system": get_system_info(),
    }

    T0 = time.time()
    mem_start = get_memory_rss_mb()
    # Do NOT pre-create output dir — ga_2beta creates it along with
    # fitted_model_plots/ subdir only when the dir doesn't exist yet.
    print(f"{'='*70}")
    print(f"crispat ga_2beta — BATCH-LEVEL 2-Beta Mixture Model")
    print(f"  Tool:           {args.tool}")
    print(f"  H5AD:           {args.h5ad}")
    print(f"  Output:         {args.output}")
    print(f"  n_iter:         {args.n_iter}")
    print(f"{'='*70}\n")

    # ── Stage 1: Load h5ad ──
    print(f"[{time.time()-T0:.0f}s] Stage 1/2: Loading h5ad …")
    t0 = time.time()
    import scanpy as sc
    adata = sc.read_h5ad(args.h5ad)
    n_cells = adata.shape[0]
    n_guides = adata.shape[1]
    n_batches = adata.obs["batch"].nunique() if "batch" in adata.obs else 1
    load_t = time.time() - t0
    print(f"  Cells: {n_cells:,}  Guides: {n_guides:,}  Batches: {n_batches}  [{load_t:.1f}s]")

    monitor["stages"]["load_h5ad"] = {
        "wall_s": round(load_t, 2),
        "ncells": n_cells,
        "nguides": n_guides,
        "n_batches": int(n_batches),
    }

    # ── Stage 2: Fit 2-Beta per batch ──
    print(f"\n[{time.time()-T0:.0f}s] Stage 2/2: Fit 2-Beta Mixture Model "
          f"({n_batches} batches, ~8s/batch estimated)")

    t0 = time.time()
    mem_before = get_memory_rss_mb()

    from crispat import ga_2beta
    ga_2beta(
        input_file=args.h5ad,
        output_dir=args.output.rstrip("/") + "/",
        n_iter=args.n_iter,
        batch_list=None,
        add_UMI_counts=True,
        UMI_threshold=args.umi_threshold,
    )
    fit_t = time.time() - t0
    mem_after = get_memory_rss_mb()

    # ── Read results ──
    out_csv = os.path.join(args.output, "assignments.csv")
    na, nca, ng2 = 0, 0, 0
    gpc_stats = {}
    if os.path.exists(out_csv):
        df = pd.read_csv(out_csv)
        na = len(df)
        nca = df["cell"].nunique()
        ng2 = df["gRNA"].nunique()
        gpc = df.groupby("cell").size()
        gpc_stats = {
            "median": float(gpc.median()),
            "mean": round(float(gpc.mean()), 2),
            "max": int(gpc.max()),
            "n_1_guide": int((gpc == 1).sum()),
            "n_2_guides": int((gpc == 2).sum()),
            "n_ge3_guides": int((gpc >= 3).sum()),
        }
        um = df["UMI_counts"].median() if "UMI_counts" in df.columns else 0
    else:
        um = 0
        df = None

    monitor["stages"]["fit_2beta"] = {
        "wall_s": round(fit_t, 2),
        "wall_min": round(fit_t / 60, 2),
        "mem_before_mb": round(mem_before, 1),
        "mem_after_mb": round(mem_after, 1),
    }

    monitor["summary"] = {
        "total_wall_s": round(time.time() - T0, 2),
        "total_wall_min": round((time.time() - T0) / 60, 2),
        "peak_rss_mb": round(get_memory_rss_mb(), 1),
        "mem_start_mb": round(mem_start, 1),
        "mem_end_mb": round(get_memory_rss_mb(), 1),
        "total_assignments": int(na),
        "cells_assigned": int(nca),
        "cells_total": n_cells,
        "cell_recovery_pct": round(nca / max(n_cells, 1) * 100, 2),
        "guides_detected": int(ng2),
        "guides_per_cell": gpc_stats,
        "umi_median": float(um),
    }

    mon_json = os.path.join(args.output, "monitoring.json")
    with open(mon_json, "w") as f:
        json.dump(monitor, f, indent=2, default=str)

    print(f"\n{'='*70}")
    print(f"crispat ga_2beta DONE — {args.tool}")
    print(f"{'='*70}")
    print(f"  Assignments:        {na:>12,}")
    print(f"  Cells assigned:     {nca:>12,}  ({nca/max(n_cells,1)*100:.1f}%)")
    print(f"  Guides detected:    {ng2:>12,}")
    if gpc_stats:
        print(f"  Guides/cell:        median={gpc_stats['median']:.0f}  "
              f"mean={gpc_stats['mean']:.2f}  max={gpc_stats['max']}")
    print(f"  Total wall time:    {time.time()-T0:.0f}s ({(time.time()-T0)/60:.1f} min)")
    print(f"  Output:             {out_csv}")
    print(f"  Monitoring:         {mon_json}")
    print(f"{'='*70}\n")


if __name__ == "__main__":
    main()
