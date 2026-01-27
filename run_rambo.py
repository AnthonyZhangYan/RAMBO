import torch
import argparse
import math
from bo import bayes_opt
from models.rambo import DPMGPR
from utils import run_common_experiment, get_common_parser

def run_dpmm_once(seed: int, context: dict) -> torch.Tensor:
    args = context["args"]
    torch.manual_seed(seed)
    T = int(args.T)

    if args.alpha_sched == "linear":
        if args.alpha_start is None and args.alpha_end is None:
            args.alpha_start = args.alpha_end = args.alpha
        elif args.alpha_start is None:
            args.alpha_start = args.alpha
        elif args.alpha_end is None:
            args.alpha_end = args.alpha

        a0 = float(args.alpha_start)
        a1 = float(args.alpha_end)
        model = DPMGPR(alpha=a0, device=context["device"])
        alpha_schedule = lambda t: a0 + (a1 - a0) * ((t - 1) / max(1, T - 1))

    elif args.alpha_sched == "sqrtlog":
        alpha0 = float(args.alpha)
        model = DPMGPR(alpha=alpha0, device=context["device"])
        alpha_schedule = lambda t: alpha0 * (math.sqrt(t) / math.log(t + math.e))

    else:
        raise ValueError(f"Unknown alpha_sched: {args.alpha_sched}")

    _, _, _, _, hist = bayes_opt(
        f_obj=context["f_obj"],
        bounds=context["bounds"],
        mode="dpmm",
        model=model,
        n_init=args.N_INIT,
        n_iter=args.T,
        acq="ei",
        xi=args.xi,
        dpmm_fit_kwargs={
            "n_iterations": 1000,
            "burn_in": 500,
            "optim_every": 5,
        },
        maximize=True,
        X0=context["X0"],
        y0=context["y0"],
        device=context["device"],
        alpha_schedule=alpha_schedule,
        pool_X=context.get("pool_X", None),
        pool_Y=context.get("pool_Y", None),
    )

    return torch.tensor([e["best"] for e in hist], dtype=torch.float32)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(parents=[get_common_parser()])
    parser.add_argument("--alpha", type=float, default=2.0)
    parser.add_argument("--alpha_start", type=float, default=None)
    parser.add_argument("--alpha_end", type=float, default=None)
    parser.add_argument(
        "--alpha_sched",
        type=str,
        default="linear",
        choices=["linear", "sqrtlog"],
    )

    args, _ = parser.parse_known_args()
    sched_tag = args.alpha_sched

    if sched_tag == "sqrtlog":
        algo_name = f"DPMM-Sqrtlog-α{args.alpha}"
    else:
        a0 = args.alpha_start if args.alpha_start is not None else args.alpha
        a1 = args.alpha_end   if args.alpha_end   is not None else args.alpha
        if float(a0) != float(a1):
            algo_name = f"DPMM-Sched-{a0}-{a1}"
        else:
            algo_name = f"DPMM-α{a0}"

    run_common_experiment(algo_name, run_dpmm_once, parser)
