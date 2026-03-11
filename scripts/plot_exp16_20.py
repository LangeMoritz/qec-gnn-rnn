"""
Plot Exp 16 (d=9 cont3) and Exp 20 (d=7 cont3) test results.

Usage:
    python scripts/plot_exp16_20.py
    python scripts/plot_exp16_20.py --out results/exp16_20_d7_d9_test_results.pdf
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

MODELS = [
    ("iterative_d7_p0.001_t50_dt2_260309_6079402_lr1e-5_cont3_load_6064033", "d=7"),
    ("iterative_d9_p0.001_t50_dt2_260309_6080257_uniform_lr_d9_cont3_load_6021817", "d=9"),
]
P_COLORS = ["#66c2a5", "#fc8d62", "#8da0cb", "#e78ac3", "#a6d854"]
MWPM_COLOR = "#b3b3b3"


def load(name):
    path = os.path.join(LOG_DIR, name + ".json")
    with open(path) as f:
        return json.load(f)["test_results"]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default="results/exp16_20_d7_d9_test_results.pdf")
    cli = parser.parse_args()

    datasets = [(label, load(name)) for name, label in MODELS]
    p_values = sorted(datasets[0][1].keys(), key=float)

    fig, axes = plt.subplots(1, 2, figsize=(8, 3.8))

    for ax, (d_label, tr) in zip(axes, datasets):
        for p, color in zip(p_values, P_COLORS):
            rounds = sorted(int(t) for t in tr[p].keys())
            mwpm_pl  = np.array([tr[p][str(t)]["mwpm"]["P_L"]  for t in rounds])
            mwpm_std = np.array([tr[p][str(t)]["mwpm"]["std"]  for t in rounds])
            nn_pl    = np.array([tr[p][str(t)]["nn"]["P_L"]    for t in rounds])
            nn_std   = np.array([tr[p][str(t)]["nn"]["std"]    for t in rounds])

            ax.errorbar(rounds, mwpm_pl, yerr=mwpm_std,
                        color=color, marker="^", capsize=2, linewidth=1.2,
                        markersize=3, linestyle=":", alpha=0.7)
            ax.errorbar(rounds, nn_pl, yerr=nn_std, label=f"p={p}",
                        color=color, marker="o", capsize=2, linewidth=1.5,
                        markersize=3, linestyle="-")

        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel("Rounds $t$")
        ax.set_title(d_label)
        ax.legend(fontsize=7, title="NN (solid) / MWPM (dotted)", title_fontsize=6)

    axes[0].set_ylabel(r"$P_L$")
    fig.suptitle("Exp 16 / 20: Hierarchical decoder d=7 and d=9 (10M shots)", fontsize=9)
    fig.tight_layout()

    os.makedirs(os.path.dirname(cli.out) or ".", exist_ok=True)
    fig.savefig(cli.out, bbox_inches="tight")
    print(f"Saved: {cli.out}")


if __name__ == "__main__":
    main()
