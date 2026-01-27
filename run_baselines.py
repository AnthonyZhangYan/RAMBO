import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import sys
import argparse
import subprocess
from datetime import datetime

from utils import (
    BASELINE_SCRIPTS,
    ensure_dir,
    format_banner,
)

def parse_supported_args(script_path):
    try:
        out = subprocess.check_output(
            [sys.executable, script_path, "--help"],
            stderr=subprocess.STDOUT,
            universal_newlines=True,
        )
    except subprocess.CalledProcessError as e:
        out = e.output

    supported = []
    for line in out.splitlines():
        line = line.strip()
        if line.startswith("--"):
            parts = line.split()
            flag = parts[0]
            flag = flag.rstrip(",:")
            supported.append(flag)
    return supported

def build_cmd_for_baseline(
    baseline_name,
    script_path,
    args,
    outdir_root,
):
    supported = parse_supported_args(script_path)
    baseline_dir = os.path.join(outdir_root, baseline_name.lower())
    ensure_dir(baseline_dir)

    cmd = [sys.executable, script_path]

    common_args = {
        "--fn": args.fn,
        "--T": args.T,
        "--N_INIT": args.N_INIT,
        "--R": args.R,
        "--outdir": baseline_dir,
    }

    for flag, value in common_args.items():
        if flag in supported:
            cmd += [flag, str(value)]

    special_possible = vars(args)
    for key, val in special_possible.items():
        flag = f"--{key}"
        if flag in common_args:
            continue
        if (flag in supported) and (val is not None):
            cmd += [flag, str(val)]

    return cmd

def build_argparser():
    p = argparse.ArgumentParser()
    p.add_argument(
        "--fn",
        choices=[
            "levy6d", "levy10d", 
            "schwefel6d", "schwefel10d", 
            "torsion_energy",
            "cancer_6t2w",
            "constellaration",
        ],
        default="levy6d",
    )
    p.add_argument("--T", type=int, default=200)
    p.add_argument("--N_INIT", type=int, default=20)
    p.add_argument("--R", type=int, default=5)
    p.add_argument("--outdir_root", type=str, default="results")

    p.add_argument(
        "--baselines",
        nargs="+",
        default=["RAMBO", "SGP"],
        choices=["RAMBO", "SGP", "ALL"],
    )

    return p

def main():
    parser = build_argparser()
    args, unknown = parser.parse_known_args()

    i = 0
    while i < len(unknown):
        if unknown[i].startswith("--"):
            key = unknown[i][2:]
            if i + 1 < len(unknown) and not unknown[i + 1].startswith("--"):
                val = unknown[i + 1]
                setattr(args, key, val)
                i += 2
            else:
                setattr(args, key, True)
                i += 1
        else:
            i += 1

    if "ALL" in args.baselines:
        baselines_to_run = ["RAMBO", "SGP"]
    else:
        baselines_to_run = args.baselines

    outdir_root = os.path.abspath(args.outdir_root)
    ensure_dir(outdir_root)

    print(format_banner("Unified Baseline Runner"))
    print(f"[INFO] fn={args.fn}, T={args.T}, N_INIT={args.N_INIT}, R={args.R}")
    print(f"[INFO] Outdir root = {outdir_root}")
    print(f"[INFO] Baselines = {baselines_to_run}")
    print(f"[INFO] Unknown extra args passed = {unknown}\n")

    for name in baselines_to_run:
        script_rel = BASELINE_SCRIPTS[name]
        script_path = os.path.join(
            os.path.dirname(os.path.abspath(__file__)),
            script_rel,
        )

        if not os.path.isfile(script_path):
            print(f"[WARN] Missing script for {name}: {script_path}")
            continue

        cmd = build_cmd_for_baseline(
            baseline_name=name,
            script_path=script_path,
            args=args,
            outdir_root=outdir_root,
        )

        print(format_banner(f"Running baseline {name}"))
        print("[CMD]", " ".join(cmd))

        ret = subprocess.call(cmd)
        print(f"[{name}] finished with code {ret}\n")
        if ret != 0:
            print(f"[ERROR] Baseline {name} failed.\n")

    print(format_banner("Baseline finished"))

if __name__ == "__main__":
    main()
