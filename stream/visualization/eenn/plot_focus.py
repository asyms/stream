import os
import pickle
import re

import pandas as pd
import plotly.graph_objects as go
import plotly.io as pio
from plotly.subplots import make_subplots

pio.kaleido.scope.mathjax = None

DATA_PATH = "outputs-eenn/focus_data/"
FIG_PATH = "outputs-eenn/focus_data/mounting_points_accuracy.pdf"
PRECISION = (8, 8)

# Define the colors for each precision combination
PRECISION_COLORS = {
    (32, 32): "blue",
    (8, 8): "green",
    (8, 4): "indianred",
    (4, 8): "purple",
    (4, 4): "orange",
}


def get_legend_name_from_label(label):
    params = get_params_from_label(label)
    ids = params[:-2]
    # Convert the ids to letters where 0 is A, 1 is B, etc.
    letters = [chr(65 + i) for i in ids]
    letters_str = ", ".join(letters)
    return f"[{letters_str}, K]"


def get_params_from_label(label):
    """
    Extract i, j, k, pb, and pc from the string and return them as a tuple
    """
    pattern = r"Model (\d+), (\d+), (\d+) pb(\d+) pc(\d+)"
    match = re.match(pattern, label)
    if match:
        i, j, k, pb, pc = match.groups()
        return int(i), int(j), int(k), int(pb), int(pc)
    else:
        raise ValueError(f"Label {label} does not match the pattern 'Model i, j, k pbpb pcpc'.")


def get_legend_name_from_pb_pc(pb, pc):
    """
    Return the legend name which is FP32+32 if pb and pc are 32, 32 and else it's INTpb+pc
    """
    if pb == 32 and pc == 32:
        return "FP32+32"
    else:
        return f"INT{pb}+{pc}"


def style_legend(fig):
    # Update layout to place the legend on top and style the box
    fig.update_layout(
        legend=dict(
            yanchor="bottom",
            y=1.1,  # Position the legend above the plot
            xanchor="center",
            x=0.5,  # Center the legend horizontally
            bgcolor="aliceblue",  # Light blue background with transparency
            bordercolor="Black",
            borderwidth=2,
            font=dict(family="Arial", size=14, color="black"),
            orientation="h",
            entrywidth=0,
        )
    )


def style_profesionally(fig):
    # figure size
    fig.update_layout(width=600, height=350)
    # choose the figure font
    font_dict = dict(family="Arial", size=14, color="black")
    fig.update_layout(
        font=font_dict,  # font formatting
        plot_bgcolor="white",  # background color
        # width=2000,  # figure width
        # height=700,  # figure height
        margin=dict(r=20, t=5, b=5),
    )
    # x and y-axis formatting
    fig.update_yaxes(
        showline=True,
        linecolor="black",
        linewidth=1,
        ticks="outside",
        tickfont=font_dict,
        tickwidth=1,
        tickcolor="black",
        showgrid=True,
        gridcolor="lightgray",
        gridwidth=0.5,
        mirror=True,
        title={"font": {"size": 20}},
    )
    fig.update_xaxes(
        showline=True,  # Show the axis line
        showticklabels=True,  # Show the tick labels
        linecolor="black",  # Axis line color
        linewidth=1,  # Axis line width
        showgrid=True,  # Show grid lines
        gridcolor="lightgrey",  # Color of the grid lines
        gridwidth=0.5,  # Width of the grid lines (thin)
        mirror=True,
        title={"font": {"size": 20}},
    )
    # Set axis titles
    fig.update_xaxes(title_text="Average Accuracy (%)")
    fig.update_yaxes(title_text="Average ET (J x cycles)")

    style_legend(fig)


def get_annotation_text_from_label(label):
    """
    Return a list of form [i,j,k] from the label of form Model i, j, k pbpb pcpc
    """
    i, j, k, _, _ = get_params_from_label(label)
    return f"[{i},{j},{k}]"


def scatter_plot_edp_accuracy(dfs, labels, exit_ratios, fig_path):
    """
    Plot a scatter plot of the EDP across stages for different precision combinations
    """
    # Add label column to each df
    for df, label in zip(dfs, labels):
        df["label"] = label

    # Create one big df containing all entries
    df = pd.concat(dfs)

    # Create a subplot with 1 row and 2 columns, sharing the y-axis
    fig = make_subplots()

    # Convert energy from pJ to J
    df["energy"] = df["energy"] / 1e12

    # Add column that computes the EDP for each stage
    df["edp"] = df["latency"] * df["energy"]

    # Create new dataframe containing for each label and each stage the minimum EDP
    min_data = []
    for label in labels:
        for stage_id in df["stage_id"].unique():
            df_stage = df[(df["label"] == label) & (df["stage_id"] == stage_id)]
            min_edp = df_stage.loc[df_stage["edp"].idxmin()]
            min_data.append(min_edp)
    df_min = pd.DataFrame(min_data)

    # Compute for each label the cumulative EDP across stages by summing the energies and latencies and then multiplying them
    for label in labels:
        df_min_label = df_min[df_min["label"] == label]
        df_min_label["cum_energy"] = df_min_label["energy"].cumsum()
        df_min_label["cum_latency"] = df_min_label["latency"].cumsum()
        df_min_label["cum_edp"] = df_min_label["cum_energy"] * df_min_label["cum_latency"]
        df_min.loc[df_min["label"] == label, "cum_edp"] = df_min_label["cum_edp"]

    # # Plot cumulative EDP across stages for each label
    # for label in labels:
    #     df_min_label = df_min[df_min["label"] == label]
    #     _, _, _, pb, pc = get_params_from_label(label)
    #     legend_name = get_legend_name_from_pb_pc(pb, pc)
    #     color = PRECISION_COLORS.get((pb, pc), "black")
    #     trace = go.Scatter(x=df_min_label["stage_id"], y=df_min_label["cum_edp"], mode="lines+markers", name=legend_name, line=dict(color=color))
    #     fig.add_trace(trace, row=1, col=1)

    # Compute average EDP based on exit ratios
    data_avg = []
    for label, exit_ratio in zip(labels, exit_ratios):
        df_min_label = df_min[df_min["label"] == label]
        avg_edp = sum(df_min_label["cum_edp"] * exit_ratio)
        data_avg.append({"label": label, "avg_edp": avg_edp, "accuracy": 100 * df_min_label["accuracy"].iloc[0]})

    # Create dataframe from data_avg
    df_avg = pd.DataFrame(data_avg)
    # Plot the average EDP for each label
    for label in labels:
        df_avg_label = df_avg[df_avg["label"] == label]
        # color = PRECISION_COLORS.get((pb, pc), "black")
        color = "black"
        name = get_legend_name_from_label(label)
        trace = go.Scatter(
            x=df_avg_label["accuracy"], y=df_avg_label["avg_edp"], mode="markers", name=name, marker=dict(color=color)
        )
        trace.showlegend = False
        fig.add_trace(trace)
        # Annotate the label next to the point
        annotation_text = get_legend_name_from_label(label)
        if annotation_text == "[A, C, G, K]":
            yshift = -15
        else:
            yshift = 15
        fig.add_annotation(
            x=df_avg_label["accuracy"].iloc[0],
            y=df_avg_label["avg_edp"].iloc[0],
            text=annotation_text,
            showarrow=False,
            font=dict(size=14),
            xshift=0,
            yshift=yshift,
        )

    # Style the figure
    style_profesionally(fig)

    # Save figure based on the extension
    file_extension = os.path.splitext(fig_path)[1].lower()
    if file_extension == ".html":
        fig.write_html(fig_path)
    elif file_extension in [".png", ".pdf"]:
        fig.write_image(fig_path, engine="kaleido")
    else:
        print("Unsupported file format. Please provide either a .html or .png extension.")
    print(f"Figure saved at {fig_path}")


def main():
    """
    Main function to load data, process it, and generate the plot
    """
    dfs = []
    labels = []

    # Get model configurations by scanning the directories in ./inputs/eenn/workload/focus/
    model_combinations = []
    for root, dirs, files in os.walk("./stream/inputs/eenn/workload/focus/"):
        for dir in sorted(dirs):
            if dir.startswith("model"):
                i, j, k = [int(i) for i in dir.split("_")[1:]]
                model_combinations.append((i, j, k))

    # Process each model configuration
    labels = []
    exit_ratios = []
    pb, pc = PRECISION
    for combination in model_combinations:
        i, j, k = combination
        label = f"Model {i}, {j}, {k} pb{pb} pc{pc}"
        labels.append(label)
        # Load in the pickled data
        data_path = os.path.join(DATA_PATH, f"model_{i}_{j}_{k}_pb{pb}_pc{pc}.pickle")
        with open(data_path, "rb") as f:
            data = pickle.load(f)
        # Load in stats file to get exit ratios and accuracy
        # Find files of the form 'net_x_stats.stats in "./stream/inputs/eenn/workload/focus/model_{i}_{j}_{k}/"
        files = os.listdir(f"./stream/inputs/eenn/workload/focus/model_{i}_{j}_{k}/")
        stats_file = [f for f in files if "net" in f and "stats" in f][0]
        stats_path = os.path.join(f"./stream/inputs/eenn/workload/focus/model_{i}_{j}_{k}/", stats_file)
        with open(stats_path, "r") as f:
            stats = eval(f.read())
        # Save exit ratios and accuracy
        exit_ratios.append(stats["exits_ratios"])
        # Create a dataframe from the data
        df = pd.DataFrame(data)
        df["model"] = f"{i}_{j}_{k}"
        df["accuracy"] = stats["top1_accuracy"]
        dfs.append(df)

    # Generate the scatter plot
    scatter_plot_edp_accuracy(dfs, labels, exit_ratios, FIG_PATH)


if __name__ == "__main__":
    main()
