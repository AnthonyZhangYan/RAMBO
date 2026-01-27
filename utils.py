import os
import sys
import io
import argparse
import torch
import matplotlib.pyplot as plt
from contextlib import redirect_stdout
from datetime import datetime
from typing import Dict

from benchmarks import make_f_obj, get_bounds

BASELINE_SCRIPTS: Dict[str, str] = {
    "SGP":     "run_sgp.py",
    "RAMBO":    "run_rambo.py",
}

def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)

def format_banner(text: str, width: int = 60, char: str = "=") -> str:
    line = char * width
    return f"{line}\n{text}\n{line}"

class Tee(io.TextIOBase):
    def __init__(self, *streams):
        self.streams = streams

    def write(self, s: str) -> int:
        for st in self.streams:
            st.write(s)
            st.flush()
        return len(s)

    def flush(self) -> None:
        for st in self.streams:
            st.flush()

def make_sobol_X0(
    bounds: torch.Tensor,
    n_init: int,
    seed: int = 42,
    device: str = "cpu",
) -> torch.Tensor:
    D = bounds.shape[0]
    engine = torch.quasirandom.SobolEngine(
        dimension=D,
        scramble=True,
        seed=seed,
    )
    U0 = engine.draw(n_init).to(device=device)
    X0 = bounds[:, 0] + (bounds[:, 1] - bounds[:, 0]) * U0
    return X0

def get_common_parser():
    p = argparse.ArgumentParser(add_help=False)
    p.add_argument(
        "--fn",
        default="levy6d",
        choices=[
            "levy6d", "levy10d", 
            "schwefel6d", "schwefel10d", 
            "torsion_energy",
            "cancer_6t2w",
            "constellaration",
        ],
    )
    p.add_argument("--outdir", type=str, default="results")
    p.add_argument("--T", type=int, default=200)
    p.add_argument("--N_INIT", type=int, default=20)
    p.add_argument("--xi", type=float, default=0.1)
    p.add_argument("--R", type=int, default=5, help="Number of independent runs")
    p.add_argument("--start_R", type=int, default=1, help="First round to run")
    p.add_argument("--end_R", type=int, default=None, help="Last round to run")
    p.add_argument("--resume", type=int, default=0, help="Set to 1 to resume")
    return p

def run_common_experiment(
    algo_name: str,
    run_once_fn: callable,
    parser: argparse.ArgumentParser = None,
):
    if parser is None:
        parser = argparse.ArgumentParser(parents=[get_common_parser()])
    args = parser.parse_args()
    
    start_r = args.start_R
    end_r = args.end_R if args.end_R is not None else args.R

    device = "cpu"
    dir_prefix = f"{args.fn}_T{args.T}_N{args.N_INIT}_{algo_name}"
    
    outdir = None
    if args.resume == 1:
        candidates = []
        if os.path.exists(args.outdir):
            for d in os.listdir(args.outdir):
                full_p = os.path.join(args.outdir, d)
                if os.path.isdir(full_p) and d.startswith(dir_prefix):
                    candidates.append(d)
        
        if candidates:
            candidates.sort()
            latest_dir = candidates[-1]
            outdir = os.path.join(args.outdir, latest_dir)
            print(format_banner(f"RESUMING EXPERIMENT"))
            print(f"[INFO] Found existing directory: {outdir}")
        else:
            print(f"[WARN] --resume requested but no folder found. Creating new.")

    if outdir is None:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        outdir = os.path.join(
            args.outdir,
            f"{dir_prefix}_{ts}",
        )
        ensure_dir(outdir)

    f_obj = make_f_obj(args.fn, negate=True)
    bounds = get_bounds(args.fn, as_torch=True, device=device)

    X0_cont = make_sobol_X0(bounds, args.N_INIT, seed=42, device=device)

    pool_X = None
    pool_Y = None
    if args.fn in ("cancer_6t2w", "constellaration"):
        if args.fn == "cancer_6t2w":
            from benchmarks import get_cancer_data
            pool_X, pool_Y, _ = get_cancer_data()
            pool_Y = pool_Y.view(-1)
        else:
            from benchmarks import get_constellar_data
            target = "metrics.qi"
            print(f"[utils] Loading discrete pool for {args.fn} with target: {target}")
            pool_X, pool_Y, _ = get_constellar_data(target_col=target)
            pool_Y = pool_Y.view(-1)

        pool_X_t = pool_X.to(device=device)
        dists = torch.cdist(X0_cont, pool_X_t)
        nn_idx = dists.argmin(dim=1)
        X0 = pool_X_t[nn_idx]
        y0 = f_obj(X0).reshape(-1)
    else:
        X0 = X0_cont
        y0 = f_obj(X0).reshape(-1)

    context = {
        "args": args,
        "f_obj": f_obj,
        "bounds": bounds,
        "X0": X0,
        "y0": y0,
        "device": device,
        "outdir": outdir,
        "pool_X": pool_X.to(device=device) if pool_X is not None else None,
        "pool_Y": pool_Y.to(device=device) if pool_Y is not None else None,
    }

    log_path = os.path.join(outdir, f"{args.fn}_{algo_name}.log")
    curves_list = []

    with open(log_path, "a", buffering=1) as f:
        tee = Tee(sys.stdout, f)
        with redirect_stdout(tee):
            print(format_banner(f"Running {algo_name} on {args.fn}"))
            print(f"[Config] T={args.T}, N_INIT={args.N_INIT}, R={args.R}")

            for s in range(start_r, end_r + 1):
                out_csv = os.path.join(outdir, f"round{s}_best.csv")
                
                if os.path.exists(out_csv):
                    print(f"===== Round ({s}/{args.R}) — {algo_name} [SKIPPING: COMPLETED] =====")
                    try:
                        vals = []
                        with open(out_csv, 'r') as cf:
                            lines = cf.readlines()
                            for line in lines[1:]:
                                parts = line.strip().split(',')
                                if len(parts) >= 2:
                                    vals.append(float(parts[1]))
                        if len(vals) > 0:
                            curve = torch.tensor(vals, dtype=torch.float32)
                            curves_list.append(curve)
                            print(f"[Loaded] {len(vals)} iters from {out_csv}")
                            continue 
                    except Exception as e:
                        print(f"[WARN] Failed to load {out_csv}, rerunning. Error: {e}")
                
                print(f"===== Round ({s}/{args.R}) — {algo_name} =====")
                try:
                    curve = run_once_fn(seed=s, context=context)
                    if not isinstance(curve, torch.Tensor):
                        curve = torch.tensor(curve, dtype=torch.float32)
                    curves_list.append(curve)

                    with open(out_csv, "w") as cf:
                        cf.write("iter,best\n")
                        for i, b in enumerate(curve.cpu().tolist()):
                            cf.write(f"{i},{b}\n")
                    print(f"[Saved] {out_csv}\n")
                except Exception as e:
                    print(f"[ERROR] Round {s} failed: {e}")
                    import traceback
                    traceback.print_exc()

            if curves_list:
                curves = torch.stack(curves_list)
                mean_curve = curves.mean(0)
                std_curve = curves.std(0)
                iters = torch.arange(len(mean_curve))

                plt.figure(figsize=(7.2, 4.2))
                plt.plot(iters, mean_curve, label=f"{algo_name}")
                plt.fill_between(
                    iters,
                    mean_curve - std_curve,
                    mean_curve + std_curve,
                    alpha=0.15,
                )
                plt.xlabel("Iteration")
                plt.ylabel("Best-so-far objective")
                plt.title(f"{args.fn} — {algo_name} (mean±std)")
                plt.legend()
                plt.grid(True)

                fig_path = os.path.join(outdir, "summary_plot.png")
                plt.savefig(fig_path, bbox_inches="tight")
                print(f"[Saved figure] {fig_path}")

                csv_path = os.path.join(outdir, "summary_mean_std.csv")
                with open(csv_path, "w") as cf:
                    cf.write("iter,mean,std\n")
                    for i, (m, st) in enumerate(zip(mean_curve.tolist(), std_curve.tolist())):
                        cf.write(f"{i},{m},{st}\n")
                print(f"[Saved summary CSV] {csv_path}")

    print(f"\n[Finished] Log: {log_path}")
