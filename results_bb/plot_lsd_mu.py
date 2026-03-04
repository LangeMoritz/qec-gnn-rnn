"""
BB code logical failure rates — three plots for reference comparison.

All log-log plots use the per-cycle logical error rate:
  p_L = 1 - (1 - P_L)^(1/t)   (≈ P_L/t for small P_L)
where t = number of syndrome rounds simulated.

Outputs (saved in results/):
  lsd_mu_72_12_6.pdf          — linear-x log-y, [[72,12,6]] all µ (P_L total)
  lsd_mu_72_12_6_loglog.pdf   — log-log, [[72,12,6]] all µ, per-cycle p_L
  lsd_mu_multicode_loglog.pdf — log-log, multi-code at µ=0, per-cycle p_L
                                (cf. IBM Fig. 2a and LSD-paper Fig. 6)
"""

import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scienceplots

HERE = os.path.dirname(os.path.abspath(__file__))

plt.style.use('science')
plt.rcParams['figure.dpi'] = 300
COLORS = ["#66c2a5", "#fc8d62", "#8da0cb", "#e78ac3",
          "#a6d854", "#e5c494", "#b3b3b3", "#ffd92f"]
plt.rcParams['axes.prop_cycle'] = plt.cycler(color=COLORS)


def per_cycle(p_l, t):
    return 1.0 - (1.0 - p_l) ** (1.0 / t)


def per_cycle_std(p_l, std, t):
    return std * (1.0 / t) * (1.0 - p_l) ** (1.0 / t - 1.0)


# ── Load ────────────────────────────────────────────────────────────────────
df = pd.read_csv(os.path.join(HERE, "lsd_mu_ibm_72_12_6.csv"), skipinitialspace=True)
df["json_metadata"] = df["json_metadata"].apply(json.loads)
df["code"]      = df["json_metadata"].apply(lambda m: m["code"])
df["p"]         = df["json_metadata"].apply(lambda m: m["p"])
df["rounds"]    = df["json_metadata"].apply(lambda m: m["rounds"])
df["lsd_order"] = df["json_metadata"].apply(lambda m: m["decoder_args"]["lsd_order"])
df["p_l"]       = df["errors"] / df["shots"]
df["std"]       = np.sqrt(df["p_l"] * (1 - df["p_l"]) / df["shots"])
df["p_l_pc"]    = df.apply(lambda r: per_cycle(r["p_l"], r["rounds"]), axis=1)
df["std_pc"]    = df.apply(lambda r: per_cycle_std(r["p_l"], r["std"], r["rounds"]), axis=1)

df72   = df[df["code"] == "BB_n72_k12"]
orders = sorted(df72["lsd_order"].unique())
p_fit  = np.logspace(np.log10(1e-4), np.log10(9e-3), 300)


def add_fit(ax, ps, pls, color, ls="-"):
    mask = pls > 0
    c = np.polyfit(np.log10(ps[mask]), np.log10(pls[mask]), 1)
    ax.plot(p_fit, 10 ** np.polyval(c, np.log10(p_fit)),
            color=color, linewidth=0.9, alpha=0.65, linestyle=ls, zorder=1)


# ── Plot 1: linear-x, log-y (P_L total) ─────────────────────────────────
fig, ax = plt.subplots(figsize=(4.5, 3.5))
for i, order in enumerate(orders):
    sub = df72[df72["lsd_order"] == order].sort_values("p")
    ax.errorbar(sub["p"], sub["p_l"], yerr=sub["std"],
                marker="o", markersize=3, linewidth=1,
                capsize=2, capthick=0.8, elinewidth=0.8,
                color=COLORS[i], label=rf"$\mu={order}$")
ax.set_xlabel(r"Physical error rate $p$")
ax.set_ylabel(r"Logical failure rate $P_L$")
ax.set_title(r"BP-LSD on BB $[[72,12,6]]$")
ax.set_yscale("log")
ax.legend(title="LSD order", fontsize=7, title_fontsize=7, loc="upper left")
fig.tight_layout()
fig.savefig(os.path.join(HERE, "lsd_mu_72_12_6.pdf"))
print("Saved lsd_mu_72_12_6.pdf")


# ── Plot 2: log-log, [[72,12,6]] vs µ, per-cycle ─────────────────────────
fig, ax = plt.subplots(figsize=(4.5, 4.0))
for i, order in enumerate(orders):
    color = COLORS[i]
    sub = df72[df72["lsd_order"] == order].sort_values("p")
    ps, pls, stds = sub["p"].values, sub["p_l_pc"].values, sub["std_pc"].values
    add_fit(ax, ps, pls, color)
    ax.errorbar(ps, pls, yerr=stds,
                fmt='D', markersize=4, color=color,
                capsize=2, capthick=0.8, elinewidth=0.8,
                zorder=2, label=rf"$\mu={order}$")
ax.set_xscale("log"); ax.set_yscale("log")
ax.set_xlim(1e-4, 1e-2)
ax.set_xlabel(r"Physical error rate $p$")
ax.set_ylabel(r"Logical error rate per cycle $p_L$")
ax.set_title(r"BP-LSD on BB $[[72,12,6]]$ — IBM data")
ax.legend(title="LSD order", fontsize=7, title_fontsize=7, loc="upper left")
fig.tight_layout()
fig.savefig(os.path.join(HERE, "lsd_mu_72_12_6_loglog.pdf"))
print("Saved lsd_mu_72_12_6_loglog.pdf")


# ── Plot 3: log-log, multi-code at µ=0, per-cycle ────────────────────────
CODE_META = {
    "BB_n72_k12":  dict(label=r"$[[72,12,6]]$",   color="#d73027", marker="o"),
    "BB_n108_k8":  dict(label=r"$[[108,8,10]]$",  color="#1a9850", marker="s"),
    "BB_n144_k12": dict(label=r"$[[144,12,12]]$", color="#ff7f00", marker="D"),
}
fig, ax = plt.subplots(figsize=(4.5, 4.0))
for code, meta in CODE_META.items():
    sub = df[(df["code"] == code) & (df["lsd_order"] == 0)].sort_values("p")
    ps  = sub["p"].values
    pls = sub["p_l_pc"].values
    stds = sub["std_pc"].values
    add_fit(ax, ps, pls, meta["color"], ls="--")
    ax.errorbar(ps, pls, yerr=stds,
                fmt=meta["marker"], markersize=5, color=meta["color"],
                capsize=2, capthick=0.8, elinewidth=0.8,
                zorder=2, label=meta["label"])
ax.set_xscale("log"); ax.set_yscale("log")
ax.set_xlim(1e-4, 1e-2)
ax.set_xlabel(r"Physical error rate $p$")
ax.set_ylabel(r"Logical error rate per cycle $p_L$")
ax.set_title(r"BP-LSD ($\mu=0$) on BB codes")
ax.legend(fontsize=9, loc="upper left")
fig.tight_layout()
fig.savefig(os.path.join(HERE, "lsd_mu_multicode_loglog.pdf"))
print("Saved lsd_mu_multicode_loglog.pdf")

# ── Key numbers at p=0.001 ───────────────────────────────────────────────
print("\nPer-cycle p_L at p=0.001, µ=0:")
for code in CODE_META:
    row = df[(df["code"] == code) & (df["lsd_order"] == 0) & (df["p"] == 0.001)]
    if len(row):
        print(f"  {code}: p_L = {row['p_l_pc'].values[0]:.3e} "
              f"± {row['std_pc'].values[0]:.1e}  ({row['rounds'].values[0]} rounds)")
