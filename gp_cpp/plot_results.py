"""
plot_results.py  --  visualize the GP/SPGP tutorial output.

Plots predictive mean +/- 2 std for both, overlays the learned pseudo-inputs,
and (optionally) the true sin(x) and the noisy training points if present.

usage:
    python plot_results.py (defaults to SPGP)
    python plot_results.py --dir path/to/csvs --save results.png
    python plot_results.py --gp        # plot GP
    python plot_results.py --spgp      # plot SPGP
    python plot_results.py --compare   # plot GP vs SPGP

requires: numpy, matplotlib   ->   pip install numpy matplotlib
"""

import argparse
import os
import numpy as np
import matplotlib.pyplot as plt

def load_pred(path):
    """Load an x,mean,var CSV. Returns (x, mean, std) sorted by x."""
    d = np.loadtxt(path, delimiter=",")
    x, mean, var = d[:, 0], d[:, 1], d[:, 2]
    order = np.argsort(x)
    x, mean, var = x[order], mean[order], var[order]
    std = np.sqrt(np.clip(var, 0, None))   # guard against tiny negatives
    return x, mean, std

def maybe_load_pseudo(path):
    if not os.path.exists(path):
        return None
    d = np.loadtxt(path, delimiter=",")
    d = np.atleast_2d(d)
    return d[:, 0]

def maybe_load_training(path):
    if not os.path.exists(path):
        return None, None
    d = np.loadtxt(path, delimiter=",")
    d = np.atleast_2d(d)
    return d[:, 0], d[:, 1]

def plot_gp(args):
    """plot full GP predictions before and after optimization."""
    before = os.path.join(args.dir, "gp_pred_before.csv")
    after = os.path.join(args.dir, "gp_pred_after.csv")
  
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), sharex=True, sharey=True)

    panel(axes[0], before, "C0", "before optimization")
    panel(axes[1], after,  "C1", "after optimization")

    tx, ty = maybe_load_training(os.path.join(args.dir, "gp_training.csv"))
    if tx is not None:
        axes[1].plot(tx, ty, ".", color="0.6", ms=3, alpha=0.5, label="training data")  

    fig.suptitle("GP predictive distribution", fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, 0.95])

    if args.save:
        fig.savefig(args.save, dpi=150)
        print("saved", args.save)
    else:
        plt.show()

def plot_spgp(args):
    """plot SPGP predictions before and after optimization."""
    before = os.path.join(args.dir, "spgp_pred_before.csv")
    after = os.path.join(args.dir, "spgp_pred_after.csv")
    pseudo_ini = os.path.join(args.dir, "spgp_pseudo_inputs_ini.csv")
    pseudo = os.path.join(args.dir, "spgp_pseudo_inputs.csv")

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), sharex=True, sharey=True)

    panel(axes[0], before, "C0", "before optimization")
    panel(axes[1], after,  "C1", "after optimization")

    # overlay learned pseudo-input locations on the "before" panel
    pi = maybe_load_pseudo(pseudo_ini)
    if pi is not None:
        y0 = axes[0].get_ylim()[0]
        axes[0].plot(pi, np.full_like(pi, y0), "k|", ms=12,
                     label="pseudo-inputs (ini)")
        axes[0].legend(loc="upper right", fontsize=8)

    # overlay learned pseudo-input locations on the "after" panel
    pi = maybe_load_pseudo(pseudo)
    if pi is not None:
        y0 = axes[1].get_ylim()[0]
        axes[1].plot(pi, np.full_like(pi, y0), "k|", ms=12,
                     label="pseudo-inputs (opt)")
        axes[1].legend(loc="upper right", fontsize=8)

    fig.suptitle("SPGP / FITC predictive distribution", fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, 0.95])

    if args.save:
        fig.savefig(args.save, dpi=150)
        print("saved", args.save)
    else:
        plt.show()

def plot_gp_spgp(args):
    """overlay full-GP and SPGP 'after' predictions on one axis."""
    gp_x, gp_m, gp_s = load_pred(os.path.join(args.dir, "gp_pred_after.csv"))
    sp_x, sp_m, sp_s = load_pred(os.path.join(args.dir, "spgp_pred_after.csv"))

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.fill_between(gp_x, gp_m - 2*gp_s, gp_m + 2*gp_s, color="C2", alpha=0.15)
    ax.plot(gp_x, gp_m, "C2", lw=2, label="full GP")
    ax.fill_between(sp_x, sp_m - 2*sp_s, sp_m + 2*sp_s, color="C1", alpha=0.15)
    ax.plot(sp_x, sp_m, "C1", lw=2, ls="--", label="SPGP (M pseudo-inputs)")
    ax.plot(gp_x, np.sin(gp_x + gp_x**2/2), "k:", lw=1, alpha=0.7, label="true sin(x+x^2/2)")

    tx, ty = maybe_load_training(os.path.join(args.dir, "gp_training.csv"))
    if tx is not None:
        ax.plot(tx, ty, ".", color="0.6", ms=3, alpha=0.5, label="training data")

    pi = maybe_load_pseudo(os.path.join(args.dir, "spgp_pseudo_inputs.csv"))
    if pi is not None:
        ax.plot(pi, np.full_like(pi, ax.get_ylim()[0]), "k|", ms=12,
                label="pseudo-inputs")

    ax.set_title("full GP vs SPGP (after optimization)")
    ax.set_xlabel("x"); ax.set_ylabel("y")
    ax.legend(loc="upper right", fontsize=8)
    fig.tight_layout()
    if args.save:
        fig.savefig(args.save, dpi=150); print("saved", args.save)
    else:
        plt.show()

def panel(ax, path, color, title, show_truth=True):
    x, mean, std = load_pred(path)
    ax.fill_between(x, mean - 2 * std, mean + 2 * std,
                    color=color, alpha=0.20, label="mean $\\pm$ 2$\\sigma$")
    ax.plot(x, mean, color=color, lw=2, label="mean")
    if show_truth:
        ax.plot(x, np.sin(x+x**2/2), "k--", lw=1, alpha=0.7, label="true sin(x+x^2/2)")
    ax.set_title(title)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.legend(loc="upper right", fontsize=8)
    return x

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default=".", help="folder containing the CSVs")
    ap.add_argument("--save", default=None, help="save figure to this path instead of showing")
    ap.add_argument("--gp", action="store_true", help="plot full GP predictions (requires gp_pred_after.csv)")
    ap.add_argument("--spgp", action="store_true", help="plot SPGP predictions (requires spgp_pred_after.csv)")
    ap.add_argument("--compare", action="store_true", help="overlay full GP vs SPGP (needs both sets of CSVs)")
    args = ap.parse_args()

    if args.gp:
        plot_gp(args)
        return

    if args.spgp:
        plot_spgp(args)
        return

    if args.compare:
        plot_gp_spgp(args)
        return
    
    print("no plotting option selected. use --gp, --spgp, or --compare.")
    print("defaulting to --spgp.")
    plot_spgp(args)

if __name__ == "__main__":
    main()
