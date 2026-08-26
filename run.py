#!/usr/bin/env python3
"""Omnibenchmark module: guide_assignment_crispat.

crispat guide assignment (SVI). Re-orchestration only: vendored scripts
`mex_to_h5ad.py` + `run_crispat_pgmm.py` / `run_crispat_2beta.py` called
UNCHANGED (patched crispat pkg supplied by the assignment_crispat env). The
injected MEX trio is presented as merged_{matrix,barcodes,features}, converted
to gRNA_counts.h5ad, then scored.

NOTE (parity): crispat uses Pyro SVI (stochastic). Determinism relies on the
pinned assignment_crispat env + fixed seed; byte-md5 parity is attempted and, if
it drifts across an env rebuild, the tolerance gate applies (see CONSISTENCY_VALIDATION).

Omnibenchmark CLI contract:
    --output_dir <dir> --name <node_id>
    --data.matrix / --data.barcodes / --data.features
    --crispat_method <pgmm|2beta>  [--umi_threshold <int>] [--n_iter <int>] [--seed <int>] [--workers <int>]

Output: <output_dir>/assignments.csv
"""
import argparse
import os
import shutil
import subprocess
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
SCRIPTS = os.path.join(HERE, "scripts")


def _mex_dir(matrix, barcodes, features, workdir):
    d = os.path.join(workdir, "mex")
    os.makedirs(d, exist_ok=True)
    for src, name in ((matrix, "merged_matrix.mtx.gz"),
                      (barcodes, "merged_barcodes.tsv.gz"),
                      (features, "merged_features.tsv.gz")):
        dst = os.path.join(d, name)
        if os.path.lexists(dst):
            os.remove(dst)
        os.symlink(os.path.abspath(src), dst)
    return d


def main():
    p = argparse.ArgumentParser(description="Omnibenchmark module: guide_assignment_crispat")
    p.add_argument("--output_dir", required=True)
    p.add_argument("--name", default="crispat")
    p.add_argument("--data.matrix", required=True)
    p.add_argument("--data.barcodes", required=True)
    p.add_argument("--data.features", required=True)
    p.add_argument("--crispat_method", default="pgmm", choices=["pgmm", "2beta"])
    p.add_argument("--umi_threshold", default="0")
    p.add_argument("--n_iter", default="1000")
    p.add_argument("--seed", default="0")
    p.add_argument("--workers", default="8")
    args = p.parse_args()

    matrix = getattr(args, "data.matrix")
    barcodes = getattr(args, "data.barcodes")
    features = getattr(args, "data.features")
    out = os.path.abspath(args.output_dir)
    os.makedirs(out, exist_ok=True)
    out_csv = os.path.join(out, "assignments.csv")

    with tempfile.TemporaryDirectory() as work:
        mex = _mex_dir(matrix, barcodes, features, work)
        # MEX -> gRNA_counts.h5ad (written alongside the MEX files)
        subprocess.run([sys.executable, os.path.join(SCRIPTS, "mex_to_h5ad.py"),
                        "--input", mex], check=True)
        h5ad = os.path.join(mex, "gRNA_counts.h5ad")

        crun = os.path.join(work, "crispat_out")  # 2beta requires a non-preexisting dir
        if args.crispat_method == "pgmm":
            cmd = [sys.executable, os.path.join(SCRIPTS, "run_crispat_pgmm.py"),
                   "--h5ad", h5ad, "--output", crun, "--tool", args.name,
                   "--umi-threshold", str(args.umi_threshold),
                   "--n-iter", str(args.n_iter), "--workers", str(args.workers)]
        else:
            cmd = [sys.executable, os.path.join(SCRIPTS, "run_crispat_2beta.py"),
                   "--h5ad", h5ad, "--output", crun, "--tool", args.name,
                   "--n-iter", str(args.n_iter), "--umi-threshold", str(args.umi_threshold)]
        print("+ " + " ".join(cmd), flush=True)
        subprocess.run(cmd, check=True)

        produced = os.path.join(crun, "assignments.csv")
        if not os.path.exists(produced):
            sys.exit(f"ERROR: crispat did not produce {produced}")
        shutil.copyfile(produced, out_csv)

    print("guide_assignment_crispat: wrote assignments.csv")


if __name__ == "__main__":
    main()
