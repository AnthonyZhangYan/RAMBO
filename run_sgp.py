import torch
import argparse
from bo import bayes_opt
from utils import run_common_experiment, get_common_parser

def run_sgp_once(seed: int, context: dict) -> torch.Tensor:
    args = context["args"]
    torch.manual_seed(seed)

    _, _, _, _, hist = bayes_opt(
        f_obj   = context["f_obj"],
        bounds  = context["bounds"],
        mode    = "sgp",
        n_init  = args.N_INIT,
        n_iter  = args.T,
        acq     = "ei",
        xi      = args.xi,
        gp_kwargs = dict(steps=220, lr=0.035),
        maximize = True,
        X0 = context["X0"], 
        y0 = context["y0"],
        device = context["device"],
        pool_X = context.get("pool_X", None),
        pool_Y = context.get("pool_Y", None),
    )
    return torch.tensor([e["best"] for e in hist], dtype=torch.float32)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(parents=[get_common_parser()])
    run_common_experiment("SGP", run_sgp_once, parser)
