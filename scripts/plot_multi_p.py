"""
Plot NN vs MWPM logical error rate across all p values from a multi-p test run.

Usage:
    python examples/plot_multi_p.py logs/d3_p0.001_t50_dt2_260224_5980183_load_5978671.json
    python examples/plot_multi_p.py logs/d3_p0.001_t50_dt2_260224_5980183_load_5978671.json --out results/multi_p_d3
"""
import sys, os, json, argparse
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))

import numpy as np
import matplotlib.pyplot as plt
import scienceplots
plt.style.use('science')
plt.rcParams['figure.dpi'] = 300
import matplotlib.colors as colors
plt.rcParams['axes.prop_cycle'] = plt.cycler(color=[
    "#66c2a5", "#fc8d62", "#8da0cb", "#e78ac3",
    "#a6d854", "#ffd92f", "#e5c494", "#b3b3b3"])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("json_path", help="Path to the test JSON log file")
    parser.add_argument("--out", default=None, help="Output path prefix (default: next to JSON)")
    cli = parser.parse_args()

    with open(cli.json_path) as f:
        summary = json.load(f)

    test_results = summary.get("test_results")
    if not test_results:
        print("No test_results found in JSON.")
        sys.exit(1)

    model_name = summary.get("model_name", os.path.splitext(os.path.basename(cli.json_path))[0])
    out_prefix = cli.out or os.path.splitext(cli.json_path)[0]
    os.makedirs(os.path.dirname(out_prefix) or ".", exist_ok=True)

    p_values = sorted(float(p) for p in test_results.keys())
    n = len(p_values)

    fig, axes = plt.subplots(1, n, figsize=(3.5 * n, 4), sharey=False)
    if n == 1:
        axes = [axes]

    for ax, p in zip(axes, p_values):
        data = test_results[str(p)]
        rounds = sorted(int(t) for t in data.keys())

        mwpm_pl  = np.array([data[str(t)]["mwpm"]["P_L"]  for t in rounds])
        mwpm_std = np.array([data[str(t)]["mwpm"]["std"]  for t in rounds])
        nn_pl    = np.array([data[str(t)]["nn"]["P_L"]    for t in rounds])
        nn_std   = np.array([data[str(t)]["nn"]["std"]    for t in rounds])

        ax.errorbar(rounds, mwpm_pl, yerr=mwpm_std, label="MWPM",
                    marker="D", capsize=3, linewidth=1.5, markersize=4)
        ax.errorbar(rounds, nn_pl,   yerr=nn_std,   label="NN",
                    marker="o", capsize=3, linewidth=1.5, markersize=4)

        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel("Rounds")
        ax.set_ylabel(r"$P_L$")
        ax.set_title(f"$p = {p}$")
        ax.legend(fontsize=8)

    args = summary.get("args", {})
    d = args.get("distance", "?")
    emb = args.get("embedding_features", "?")
    h   = args.get("hidden_size", "?")
    ng  = args.get("n_gru_layers", "?")
    p_train = args.get("error_rates") or [args.get("error_rate", "?")]
    best_ep = summary.get("best_epoch", "?")

    p_str  = "{" + ", ".join(str(p) for p in p_train) + "}"
    emb_str = "--".join(str(e) for e in emb) if isinstance(emb, list) else str(emb)
    arch_str = (f"GNN [{emb_str}], GRU h={h}x{ng}  |  "
                f"train: t={args.get('t','?')}, p in {p_str}, {best_ep} epochs")

    fig.suptitle(f"d={d}  multi-p test  ({model_name})", fontsize=9)
    fig.text(0.5, 0.01, arch_str, ha="center", fontsize=7,
             color="#555555", style="italic")
    fig.tight_layout(rect=[0, 0.04, 1, 1])

    out_pdf = out_prefix + "_multi_p.pdf"
    fig.savefig(out_pdf, bbox_inches="tight")
    print(f"Saved: {out_pdf}")


if __name__ == "__main__":
    main()
