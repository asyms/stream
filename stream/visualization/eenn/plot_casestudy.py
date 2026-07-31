"""Heterogeneous multi-core case study: the Figure 8 Pareto EENNs on nine iso-budget systems.

Nine 4-compute-core accelerators built from three core styles, all with the same compute and
memory budget and all sharing the pooling, SIMD and off-chip cores, mesh and mapping that
produced Figure 8. sys_EEEE *is* the Figure 8 accelerator, so it anchors the numbers. The six
workloads are the Pareto-optimal architectures on the gold line of Figure 8.

Core styles (what differs is how the register files are wired to the array, `served_dimensions`):
    E  edge_tpu_like   weight RF shared across [D3,D4] (one read feeds 16 MACs)
    M  meta_like       same sharing, wider D1, split weight/activation banks
    Y  eyeriss_like    every RF private to one multiplier, extra output SRAM level

Figures:
  casestudy_main.pdf        accuracy vs ET per system, and the mean-ET ranking of the systems.
                            This is the headline figure.
  casestudy_allocation.pdf  which core style the allocator actually used, per system. The
                            mechanism behind any heterogeneous gain.
  casestudy_breakdown.pdf   per-exit-stage backbone vs classifier latency, for sanity checking.

Run from this directory:  cd stream/visualization/eenn && python plot_casestudy.py
"""

import argparse
import glob
import os
import pickle
import sys

import pandas as pd
import plotly.graph_objects as go
import plotly.io as pio
from plotly.subplots import make_subplots
from utils import save_fig, style_figure, style_legend

if pio.kaleido.scope is not None:
    pio.kaleido.scope.mathjax = None

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

DEFAULT_OUT_ROOT = "../../../outputs-eenn/hw_casestudy"
NAS_DIR = "../../../stream/inputs/eenn/workload/nas"
PRECISION_DIR = "pb8_pc8"
ANCHOR = "sys_EEEE"

# The six Pareto-optimal architectures of Figure 8: exit-id string -> (nas dir, Figure 8 label).
WORKLOADS = {
    "6_11": ("iter_0/net_3", "6_11_iter_0"),
    "2_8_11": ("iter_0/net_8", "2_8_11_iter_0"),
    "0_4_5_7_11": ("iter_0/net_1", "0_4_5_7_11_iter_0"),
    "2_3_4_5_9_11": ("iter_3/net_3", "2_3_4_5_9_11_iter_3"),
    "0_1_5_8_9_11": ("iter_2/net_7", "0_1_5_8_9_11_iter_2"),
    "2_3_4_5_6_9_11": ("iter_6/net_0", "2_3_4_5_6_9_11_iter_6"),
}
SYS_ORDER = ["sys_EEEE", "sys_MMMM", "sys_YYYY", "sys_EEYY", "sys_EEMM", "sys_MMYY", "sys_EEEY", "sys_EYYY", "sys_EEMY"]
STYLE_COLOR = {"E": "#1f77b4", "M": "#2ca02c", "Y": "#d62728"}
BEST_COLOR = "#e6b800"


def composition(system):
    """'sys_EEMY' -> 'EEMY'."""
    return system.replace("sys_", "")


def pretty(system):
    comp = composition(system)
    label = " ".join(comp)
    return f"{label}  (as in Fig. 8)" if system == ANCHOR else label


def is_homogeneous(system):
    return len(set(composition(system))) == 1


def system_color(system):
    """Homogeneous systems take their style colour; mixes blend their constituents."""
    comp = composition(system)
    rgb = [0, 0, 0]
    for letter in comp:
        base = STYLE_COLOR[letter]
        for i in range(3):
            rgb[i] += int(base[1 + 2 * i : 3 + 2 * i], 16)
    return "#" + "".join(f"{v // len(comp):02x}" for v in rgb)


def load_stats(rel):
    path = os.path.join(NAS_DIR, rel)
    stats_file = next(f for f in os.listdir(path) if "stats" in f)
    with open(os.path.join(path, stats_file)) as handle:
        stats = eval(handle.read())
    return 100 * stats["top1_accuracy"], stats["exits_ratios"]


def best_per_stage(stage_data):
    frame = pd.DataFrame(stage_data)
    frame["energy_j"] = frame["energy"] / 1e12
    frame["edp"] = frame["latency"] * frame["energy_j"]
    return frame.loc[frame.groupby("stage_id")["edp"].idxmin()].sort_values("stage_id").reset_index(drop=True)


def collect(out_root):
    runs, stages, alloc = [], [], []
    for system in sorted(os.listdir(out_root)):
        stage_dir = os.path.join(out_root, system, PRECISION_DIR, "stage_data")
        if not os.path.isdir(stage_dir):
            continue
        comp = composition(system)
        for ids, (rel, label) in WORKLOADS.items():
            path = os.path.join(stage_dir, f"model_{ids}.pickle")
            if not os.path.exists(path):
                print(f"  warning: missing {system}/{ids}")
                continue
            with open(path, "rb") as handle:
                data = pickle.load(handle)
            if not data:
                continue
            accuracy, exit_ratios = load_stats(rel)
            best = best_per_stage(data)
            if len(best) != len(exit_ratios):
                print(f"  warning: {system}/{ids}: {len(best)} stages vs {len(exit_ratios)} ratios")
                continue
            cum_e = best["energy_j"].cumsum()
            cum_l = best["latency"].cumsum()
            cum_et = cum_e * cum_l
            static_et = float((cum_e * best["block_latency"].cumsum()).iloc[-1])
            avg_et = float((cum_et * pd.Series(exit_ratios)).sum())
            runs.append(
                {
                    "system": system,
                    "composition": comp,
                    "homogeneous": is_homogeneous(system),
                    "workload": ids,
                    "label": label,
                    "n_exits": len(exit_ratios),
                    "accuracy": accuracy,
                    "avg_et": avg_et,
                    "avg_latency": float((cum_l * pd.Series(exit_ratios)).sum()),
                    "avg_energy": float((cum_e * pd.Series(exit_ratios)).sum()),
                    "et_reduction": static_et / avg_et,
                    "classifier_share": float(best["classifier_latency"].sum() / best["latency"].sum()),
                }
            )
            for _, row in best.iterrows():
                stages.append(
                    {
                        "system": system,
                        "workload": ids,
                        "stage": int(row["stage_id"]),
                        "block_latency": row["block_latency"],
                        "classifier_latency": row["classifier_latency"],
                    }
                )
            alloc += core_usage(out_root, system, comp, ids)
    return pd.DataFrame(runs), pd.DataFrame(stages), pd.DataFrame(alloc)


def core_usage(out_root, system, comp, ids):
    path = os.path.join(out_root, system, PRECISION_DIR, "runs", f"model_{ids}", "scme.pickle")
    if not os.path.exists(path):
        return []
    with open(path, "rb") as handle:
        scme = pickle.load(handle)
    busy = {}
    for node in scme.workload.node_list:
        core_id = getattr(node, "chosen_core_allocation", None)
        if core_id is None or core_id > 3:
            continue
        style = comp[core_id]
        busy[style] = busy.get(style, 0.0) + float(node.end - node.start)
    total = sum(busy.values()) or 1.0
    # Emit a row for every style *present in this system*, including ones the allocator did not
    # use. Without the zeros, averaging over workloads only averages the runs where a style was
    # used at all, and the per-system shares no longer sum to 100.
    return [
        {"system": system, "workload": ids, "style": s, "share": 100 * busy.get(s, 0.0) / total}
        for s in sorted(set(comp))
    ]


def has_raw_outputs(out_root):
    """True when the full sweep outputs are present, as opposed to only the committed CSVs."""
    return bool(glob.glob(os.path.join(out_root, "*", PRECISION_DIR, "stage_data", "*.pickle")))


def read_csvs(out_root):
    """Rebuild the three frames from the committed CSVs.

    The raw sweep writes ~170 MB of per-stage GA traces, of which the figures use one row per
    exit stage. Only the aggregated CSVs are kept in the repository, so the figures regenerate
    from a clean checkout without the raw outputs.
    """
    runs = pd.read_csv(os.path.join(out_root, "casestudy_runs.csv"))
    stages_path = os.path.join(out_root, "casestudy_stages.csv")
    alloc_path = os.path.join(out_root, "casestudy_allocation.csv")
    stages = pd.read_csv(stages_path) if os.path.exists(stages_path) else pd.DataFrame()
    alloc = pd.read_csv(alloc_path) if os.path.exists(alloc_path) else pd.DataFrame()
    return runs, stages, alloc


def plot_main(runs, fig_path, exclude=()):
    """Headline figure: the six Figure 8 Pareto EENNs run on every system.

    One line per accelerator, one marker per EENN architecture. Sized and styled for direct
    inclusion in the paper: single panel, no annotations outside the axes.
    """
    present = [s for s in SYS_ORDER if s in set(runs["system"]) and composition(s) not in exclude]
    # Draw the anchor last so it sits on top: several mixes land within 1% of it and would
    # otherwise hide it. No "winner" is highlighted, because the best mix only matches the
    # anchor and colouring it as a winner would overstate the result.
    present = [s for s in present if s != ANCHOR] + [ANCHOR]

    fig = make_subplots()
    for system in present:
        sub = runs[runs["system"] == system].sort_values("accuracy")
        fig.add_trace(
            go.Scatter(
                x=sub["accuracy"],
                y=sub["avg_et"],
                mode="lines+markers",
                name=pretty(system),
                line=dict(
                    color=system_color(system),
                    width=2.0,
                    dash="solid" if is_homogeneous(system) else "dot",
                ),
                marker=dict(size=9, symbol="circle", line=dict(width=0.8, color="black")),
                customdata=sub["n_exits"],
                hovertemplate="%{fullData.name}<br>%{customdata} exits<br>ET %{y:.0f}<extra></extra>",
            )
        )

    font = dict(family="Arial", size=19, color="black")
    fig.update_layout(
        width=980,
        height=620,
        plot_bgcolor="white",
        font=font,
        margin=dict(l=10, r=25, t=110, b=10),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="center",
            x=0.5,
            entrywidth=97,
            bgcolor="aliceblue",
            bordercolor="black",
            borderwidth=1.5,
            font=dict(family="Arial", size=16, color="black"),
            tracegroupgap=4,
        ),
    )
    axis = dict(
        showline=True,
        linecolor="black",
        linewidth=1.2,
        mirror=True,
        showgrid=True,
        gridcolor="lightgrey",
        gridwidth=0.6,
        ticks="outside",
        tickfont=dict(family="Arial", size=17, color="black"),
        title=dict(font=dict(family="Arial", size=21, color="black")),
    )
    fig.update_xaxes(**axis, title_text="Average accuracy (%)")
    fig.update_yaxes(**axis, title_text="Average ET (J x cycles)", type="log")
    save_fig(fig, fig_path)


def plot_allocation(alloc, fig_path):
    if alloc.empty:
        print("  no allocation data; skipping")
        return
    present = [s for s in SYS_ORDER if s in set(alloc["system"])]
    fig = make_subplots()
    for style in ["E", "M", "Y"]:
        vals = []
        for system in present:
            sub = alloc[(alloc["system"] == system) & (alloc["style"] == style)]
            vals.append(sub["share"].mean() if len(sub) else 0.0)
        fig.add_trace(
            go.Bar(x=[pretty(s) for s in present], y=vals, name=f"{style} cores", marker_color=STYLE_COLOR[style])
        )
    style_figure(fig, "System (4 compute cores)", "Share of compute runtime (%)")
    style_legend(fig, title="Core style the allocator used")
    fig.update_layout(width=1000, height=500, barmode="stack", margin=dict(r=20, t=110, b=5))
    save_fig(fig, fig_path)


def plot_breakdown(stages, fig_path):
    present = [s for s in SYS_ORDER if s in set(stages["system"])]
    fig = make_subplots()
    for part, color in (("block_latency", "#4c78a8"), ("classifier_latency", "#f58518")):
        vals = [stages[stages["system"] == s][part].sum() for s in present]
        fig.add_trace(go.Bar(x=[pretty(s) for s in present], y=vals, name=part.replace("_", " "), marker_color=color))
    style_figure(fig, "System (4 compute cores)", "Summed latency over all stages (cycles)")
    style_legend(fig, title="Where the cycles go")
    fig.update_layout(width=1000, height=500, barmode="stack", margin=dict(r=20, t=110, b=5))
    save_fig(fig, fig_path)


def report(runs, alloc):
    pd.set_option("display.width", 240)
    present = [s for s in SYS_ORDER if s in set(runs["system"])]
    print("\n" + "=" * 84)
    print("HETEROGENEOUS CASE STUDY  (9 iso-budget systems x 6 Figure 8 Pareto EENNs)")
    print("=" * 84)

    piv = runs.pivot_table(index="workload", columns="system", values="avg_et")[present]
    order = runs.groupby("workload")["accuracy"].first().sort_values().index
    print("\nAverage ET (J x cycles), workloads ordered by accuracy:")
    print(piv.reindex(order).round(0).to_string())

    mean_et = runs.groupby("system")["avg_et"].mean().reindex(present).sort_values()
    print("\nMean ET per system:")
    for s, v in mean_et.items():
        tag = "  <- Figure 8 anchor" if s == ANCHOR else ""
        print(f"  {pretty(s):<20}{v:>9.1f}   {mean_et[ANCHOR]/v:>5.2f}x vs anchor{tag}")
    best = mean_et.index[0]
    print(f"\nBest system: {pretty(best)}  ({mean_et[ANCHOR]/mean_et[best]:.2f}x lower ET than the anchor)")

    rk = {s: tuple(runs[runs["system"] == s].sort_values("avg_et")["workload"]) for s in present}
    print(f"\nDistinct EENN rankings across the nine systems: {len(set(rk.values()))}")

    print("\nET reduction from early exiting (mean per system):")
    for s, v in runs.groupby("system")["et_reduction"].mean().reindex(present).items():
        print(f"  {pretty(s):<20}{v:>6.2f}x")

    if not alloc.empty:
        print("\nShare of compute runtime by core style (mean over workloads):")
        tab = alloc.pivot_table(index="system", columns="style", values="share", aggfunc="mean")
        print(tab.reindex(present).round(1).to_string())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-root", default=DEFAULT_OUT_ROOT)
    parser.add_argument(
        "--exclude",
        default="",
        help="Comma-separated compositions to leave out of the main figure, e.g. --exclude MMMM. "
        "MMMM sits far above the rest and compresses the y-axis; dropping it makes the "
        "differences between the remaining systems readable.",
    )
    args = parser.parse_args()
    exclude = tuple(x.strip() for x in args.exclude.split(",") if x.strip())

    if has_raw_outputs(args.out_root):
        runs, stages, alloc = collect(args.out_root)
        if runs.empty:
            raise SystemExit(f"No results under {args.out_root}")
        runs.to_csv(os.path.join(args.out_root, "casestudy_runs.csv"), index=False)
        stages.to_csv(os.path.join(args.out_root, "casestudy_stages.csv"), index=False)
        if not alloc.empty:
            alloc.to_csv(os.path.join(args.out_root, "casestudy_allocation.csv"), index=False)
        print(f"Derived from the raw sweep outputs under {args.out_root}; CSVs refreshed.")
    else:
        runs, stages, alloc = read_csvs(args.out_root)
        if runs.empty:
            raise SystemExit(f"No results and no CSVs under {args.out_root}")
        print(f"Raw sweep outputs not present; plotting from the committed CSVs in {args.out_root}.")
    report(runs, alloc)
    plot_main(runs, os.path.join(args.out_root, "casestudy_main.pdf"), exclude=exclude)
    if not exclude:
        # Also emit the zoomed variant, since MMMM dominates the y-range.
        plot_main(runs, os.path.join(args.out_root, "casestudy_main_no_MMMM.pdf"), exclude=("MMMM",))
    plot_allocation(alloc, os.path.join(args.out_root, "casestudy_allocation.pdf"))
    plot_breakdown(stages, os.path.join(args.out_root, "casestudy_breakdown.pdf"))
    print(f"\nFigures written to {args.out_root}")


if __name__ == "__main__":
    main()
