#!/usr/bin/env python3
"""
crispat ga_poisson_gauss — parallel runner with monitoring.

Runs ga_poisson_gauss across 16 workers (split by start_gRNA/step).
Each tool × UMI_threshold combo produces a merged assignments.csv + monitoring.json.

Usage:
  python run_crispat_pgmm.py \
      --h5ad 04_extraction_mex/cellranger/gRNA_counts.h5ad \
      --output 06_pgmm_crispat/cellranger/UMI_0/ \
      --tool cellranger \
      --umi-threshold 0 \
      --n-iter 500 \
      --workers 16
"""

import os, sys, time, json, platform, shutil, argparse
from pathlib import Path
from multiprocessing import Pool
import numpy as np
import pandas as pd

# ── Monitoring helpers ──
try:
    import psutil
    _HAS_PSUTIL = True
except ImportError:
    _HAS_PSUTIL = False

try:
    import resource
    _HAS_RESOURCE = True
except ImportError:
    _HAS_RESOURCE = False


def get_memory_rss_mb():
    if _HAS_PSUTIL:
        return psutil.Process().memory_info().rss / (1024 * 1024)
    elif _HAS_RESOURCE:
        return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024
    return -1.0


def get_system_info():
    info = {
        "hostname": platform.node(),
        "platform": platform.platform(),
        "python_version": sys.version,
    }
    import torch; info["torch_version"] = torch.__version__
    import pyro;  info["pyro_version"] = pyro.__version__
    try:
        import anndata; info["anndata_version"] = anndata.__version__
    except Exception: pass
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


# ── Worker function (runs in subprocess, imports crispat inside) ──
def _worker(args):
    """Run ga_poisson_gauss on a chunk of guides."""
    chunk_id, start_g, step, h5ad_path, chunk_dir, n_iter, umi_th = args

    # Per-worker OMP control
    omp_threads = 2
    os.environ["OMP_NUM_THREADS"] = str(omp_threads)
    os.environ["MKL_NUM_THREADS"] = str(omp_threads)
    import torch; torch.set_num_threads(omp_threads)
    from crispat import ga_poisson_gauss

    # Do NOT pre-create chunk_dir — ga_poisson_gauss creates its own
    # output dir + fitted_model_plots/ + loss_plots/ subdirs, but only
    # when the output dir itself doesn't exist yet.

    t0 = time.time()
    ga_poisson_gauss(
        input_file=h5ad_path,
        output_dir=chunk_dir.rstrip("/") + "/",
        start_gRNA=start_g,
        step=step,
        n_iter=n_iter,
        n_counts=None,
        UMI_threshold=umi_th,
    )
    elapsed = time.time() - t0

    # Find the output CSV
    csvs = list(Path(chunk_dir).glob("assignments_*.csv"))
    n_assign = 0
    if csvs:
        df = pd.read_csv(csvs[0])
        n_assign = len(df)
    return {"chunk": chunk_id, "time_s": round(elapsed, 1), "assignments": n_assign}


# ═══════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════
def main():
    parser = argparse.ArgumentParser(
        description="crispat ga_poisson_gauss — parallel runner with monitoring"
    )
    parser.add_argument("--h5ad", required=True, help="Path to gRNA_counts.h5ad")
    parser.add_argument("--output", required=True, help="Output directory")
    parser.add_argument("--tool", required=True, help="Tool label")
    parser.add_argument("--umi-threshold", type=int, default=0)
    parser.add_argument("--n-iter", type=int, default=500)
    parser.add_argument("--workers", type=int, default=16)
    parser.add_argument("--omp-threads", type=int, default=2)
    args = parser.parse_args()

    monitor = {
        "method": "ga_poisson_gauss",
        "tool": args.tool,
        "parameters": {
            "n_iter": args.n_iter,
            "n_counts": None,
            "UMI_threshold": args.umi_threshold,
            "workers": args.workers,
            "omp_threads_per_worker": args.omp_threads,
        },
        "stages": {},
        "system": get_system_info(),
    }

    T0 = time.time()
    mem_start = get_memory_rss_mb()
    os.makedirs(args.output, exist_ok=True)

    print(f"{'='*70}")
    print(f"crispat ga_poisson_gauss — PARALLEL ({args.workers} workers)")
    print(f"  Tool:           {args.tool}")
    print(f"  H5AD:           {args.h5ad}")
    print(f"  Output:         {args.output}")
    print(f"  n_iter:         {args.n_iter}")
    print(f"  UMI_threshold:  {args.umi_threshold}")
    print(f"{'='*70}\n")

    # ── Stage 1: Load h5ad to get n_guides ──
    print(f"[{time.time()-T0:.0f}s] Stage 1/3: Loading h5ad to count guides …")
    t0 = time.time()
    import scanpy as sc
    adata = sc.read_h5ad(args.h5ad)
    n_guides = adata.shape[1]
    n_cells = adata.shape[0]
    guide_names = adata.var_names.tolist()
    load_t = time.time() - t0
    print(f"  Cells: {n_cells:,}  Guides: {n_guides:,}  [{load_t:.1f}s]")

    monitor["stages"]["load_h5ad"] = {
        "wall_s": round(load_t, 2),
        "ncells": n_cells,
        "nguides": n_guides,
        "h5ad_path": args.h5ad,
    }

    # ── Stage 2: Parallel SVI ──
    print(f"\n[{time.time()-T0:.0f}s] Stage 2/3: Parallel SVI fitting "
          f"({args.workers} workers × ~{n_guides // args.workers} guides each) …")

    chunk_dir = os.path.join(args.output, "chunks")
    if os.path.exists(chunk_dir):
        shutil.rmtree(chunk_dir)
    os.makedirs(chunk_dir, exist_ok=True)

    chunk_size = n_guides // args.workers
    remainder = n_guides % args.workers
    tasks, start = [], 0
    for i in range(args.workers):
        step = chunk_size + (1 if i < remainder else 0)
        if step == 0:
            break
        tasks.append((i, start, step, args.h5ad,
                      os.path.join(chunk_dir, f"chunk_{i:02d}") + "/",
                      args.n_iter, args.umi_threshold))
        start += step

    # Estimated: ~4s/guide SVI → ~283 guides/worker → ~19 min per tool
    est_per_worker = (chunk_size * 4) / 60
    print(f"  Estimated worker time: ~{est_per_worker:.0f} min/worker")

    t0 = time.time()
    mem_before = get_memory_rss_mb()
    with Pool(processes=len(tasks)) as pool:
        worker_results = pool.map(_worker, tasks)
    em_wall = time.time() - t0
    mem_after = get_memory_rss_mb()

    worker_times = [r["time_s"] for r in worker_results]
    total_assign = sum(r["assignments"] for r in worker_results)

    monitor["stages"]["svi_fitting"] = {
        "wall_s": round(em_wall, 2),
        "wall_min": round(em_wall / 60, 2),
        "mem_before_mb": round(mem_before, 1),
        "mem_after_mb": round(mem_after, 1),
        "n_workers": len(tasks),
        "chunk_size": chunk_size,
        "worker_wall_min": round(min(worker_times), 1),
        "worker_wall_max": round(max(worker_times), 1),
        "worker_wall_mean": round(np.mean(worker_times), 1),
        "worker_wall_median": round(np.median(worker_times), 1),
        "raw_assignments_from_chunks": total_assign,
    }
    print(f"  SVI wall time: {em_wall:.0f}s ({em_wall/60:.1f} min)")
    print(f"  Worker times: min={min(worker_times):.0f}s  max={max(worker_times):.0f}s  "
          f"mean={np.mean(worker_times):.0f}s")

    # ── Stage 3: Merge chunks ──
    print(f"\n[{time.time()-T0:.0f}s] Stage 3/3: Merge chunks …")
    t0 = time.time()
    mem_before = get_memory_rss_mb()

    all_dfs = []
    for chunk_subdir in sorted(Path(chunk_dir).iterdir()):
        csvs = list(chunk_subdir.glob("assignments_*.csv"))
        if csvs:
            all_dfs.append(pd.read_csv(csvs[0]))

    if all_dfs:
        merged = pd.concat(all_dfs, ignore_index=True)
    else:
        merged = pd.DataFrame(columns=["cell", "gRNA", "UMI_counts"])

    out_csv = os.path.join(args.output, "assignments.csv")
    merged.to_csv(out_csv, index=False)

    merge_t = time.time() - t0
    mem_after = get_memory_rss_mb()
    out_size_mb = os.path.getsize(out_csv) / (1024**2)

    monitor["stages"]["merge"] = {
        "wall_s": round(merge_t, 2),
        "mem_before_mb": round(mem_before, 1),
        "mem_after_mb": round(mem_after, 1),
        "output_csv": out_csv,
        "output_size_mb": round(out_size_mb, 1),
        "final_assignments": len(merged),
    }

    # ── Summary ──
    tt = time.time() - T0
    na = len(merged)
    nca = merged["cell"].nunique() if na > 0 else 0
    ng2 = merged["gRNA"].nunique() if na > 0 else 0
    gpc = merged.groupby("cell").size() if na > 0 else pd.Series(dtype=int)
    um = merged["UMI_counts"].median() if na > 0 and "UMI_counts" in merged.columns else 0

    monitor["summary"] = {
        "total_wall_s": round(tt, 2),
        "total_wall_min": round(tt / 60, 2),
        "peak_rss_mb": round(get_memory_rss_mb(), 1),
        "mem_start_mb": round(mem_start, 1),
        "mem_end_mb": round(get_memory_rss_mb(), 1),
        "total_assignments": int(na),
        "cells_assigned": int(nca),
        "cells_total": n_cells,
        "cell_recovery_pct": round(nca / max(n_cells, 1) * 100, 2),
        "guides_detected": int(ng2),
        "guides_per_cell_median": float(gpc.median()) if na > 0 else 0,
        "guides_per_cell_mean": round(float(gpc.mean()), 2) if na > 0 else 0,
        "guides_per_cell_max": int(gpc.max()) if na > 0 else 0,
        "cells_1_guide": int((gpc == 1).sum()) if na > 0 else 0,
        "cells_2_guides": int((gpc == 2).sum()) if na > 0 else 0,
        "cells_ge3_guides": int((gpc >= 3).sum()) if na > 0 else 0,
        "umi_median": float(um),
    }

    mon_json = os.path.join(args.output, "monitoring.json")
    with open(mon_json, "w") as f:
        json.dump(monitor, f, indent=2, default=str)

    print(f"\n{'='*70}")
    print(f"crispat ga_poisson_gauss DONE — {args.tool}  UMI>={args.umi_threshold}")
    print(f"{'='*70}")
    print(f"  Assignments:        {na:>12,}")
    print(f"  Cells assigned:     {nca:>12,}  ({nca/max(n_cells,1)*100:.1f}%)")
    print(f"  Guides detected:    {ng2:>12,}")
    if na > 0:
        print(f"  Guides/cell:        median={gpc.median():.0f}  "
              f"mean={gpc.mean():.2f}  max={gpc.max()}")
    print(f"  Total wall time:    {tt:.0f}s ({tt/60:.1f} min)")
    print(f"  Output:             {out_csv}  ({out_size_mb:.1f} MB)")
    print(f"  Monitoring:         {mon_json}")
    print(f"{'='*70}\n")


if __name__ == "__main__":
    main()
