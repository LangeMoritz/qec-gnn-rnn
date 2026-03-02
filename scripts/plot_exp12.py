"""
Plot Exp 12 test results: Frozen vs. Trainable vs. Random GNN (d=5, multi-p).

Usage:
    python scripts/plot_exp12.py
    python scripts/plot_exp12.py --out results/exp12_test_results.pdf
"""
import os, json, argparse
import numpy as np
import matplotlib.pyplot as plt
import scienceplots

plt.style.use('science')
plt.rcParams['figure.dpi'] = 300
plt.rcParams['axes.prop_cycle'] = plt.cycler(color=[
    "#66c2a5", "#fc8d62", "#8da0cb", "#e78ac3",
    "#a6d854", "#ffd92f", "#e5c494", "#b3b3b3"])

LOG_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "logs")

CONDITIONS = [
    ("iterative_d5_p0.001_t50_dt2_260227_6005309_ctrl_frozen",   "Frozen GNN",    "D", "--"),
    ("iterative_d5_p0.001_t50_dt2_260227_6005310_trainable_gnn", "Trainable GNN", "o", "-"),
    ("iterative_d5_p0.001_t50_dt2_260227_6005311_random_gnn",    "Random GNN",    "s", "-"),
]
COLORS = ["#8da0cb", "#fc8d62", "#66c2a5"]   # frozen=blue, trainable=orange, random=teal
MWPM_COLOR = "#b3b3b3"


def load(name):
    path = os.path.join(LOG_DIR, name + ".json")
    with open(path) as f:
        return json.load(f)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default="results/exp12_test_results.pdf")
    cli = parser.parse_args()

    data = [(label, marker, ls, load(name)["test_results"])
            for name, label, marker, ls in CONDITIONS]

    # use MWPM from the first condition (all share the same baseline)
    p_values = sorted(float(p) for p in data[0][3].keys())
    n = len(p_values)

    fig, axes = plt.subplots(1, n, figsize=(3.2 * n, 3.8), sharey=False)

    for ax, p in zip(axes, p_values):
        p_str = str(p)
        # MWPM from first condition
        rounds_ref = sorted(int(t) for t in data[0][3][p_str].keys())
        mwpm_pl  = np.array([data[0][3][p_str][str(t)]["mwpm"]["P_L"]  for t in rounds_ref])
        mwpm_std = np.array([data[0][3][p_str][str(t)]["mwpm"]["std"]  for t in rounds_ref])
        ax.errorbar(rounds_ref, mwpm_pl, yerr=mwpm_std, label="MWPM",
                    color=MWPM_COLOR, marker="^", capsize=3, linewidth=1.5,
                    markersize=4, linestyle=":")

        for (label, marker, ls, tr), color in zip(data, COLORS):
            rounds = sorted(int(t) for t in tr[p_str].keys())
            nn_pl  = np.array([tr[p_str][str(t)]["nn"]["P_L"]  for t in rounds])
            nn_std = np.array([tr[p_str][str(t)]["nn"]["std"]  for t in rounds])
            ax.errorbar(rounds, nn_pl, yerr=nn_std, label=label,
                        color=color, marker=marker, capsize=3, linewidth=1.5,
                        markersize=4, linestyle=ls)

        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel("Rounds $t$")
        if ax is axes[0]:
            ax.set_ylabel(r"$P_L$")
        ax.set_title(f"$p = {p}$")
        ax.legend(fontsize=7)

    fig.suptitle("Exp 12: d=5 Hierarchical Decoder — GNN Ablation", fontsize=9)
    fig.tight_layout()

    os.makedirs(os.path.dirname(cli.out) or ".", exist_ok=True)
    fig.savefig(cli.out, bbox_inches="tight")
    print(f"Saved: {cli.out}")


if __name__ == "__main__":
    main()
