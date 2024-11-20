import os
import pickle
import re

import pandas as pd
import plotly.graph_objects as go
import plotly.io as pio
from plotly.subplots import make_subplots

pio.kaleido.scope.mathjax = None

DATA_PATH = "outputs-eenn/precision_data/"
FIG_PATH = "outputs-eenn/precision_data/precision.pdf"
SUBPLOT_TITLES = ("Cumulative ET across stages", "Average ET based on exit ratios")
# Define the EXIT_RATIOS dictionary
EXIT_RATIOS = {
    (3, 5, 8, 32, 32): [0.3421, 0.1422, 0.2822, 0.2335],
    (3, 5, 8, 8, 8): [0.2549, 0.1531, 0.3114, 0.2806],
    (3, 5, 8, 8, 4): [0.2418, 0.1512, 0.2931, 0.3139],
    (3, 5, 8, 4, 8): [0.2271, 0.121, 0.2841, 0.3657],
    (3, 5, 8, 4, 4): [0.3149, 0.1233, 0.2372, 0.3246],
}
# Define the precision combinations
PRECISION_COMBINATIONS = [(32, 32), (8, 8), (8, 4), (4, 8), (4, 4)]
# Define the colors for each precision combination
PRECISION_COLORS = {
    (32, 32): "blue",
    (8, 8): "green",
    (8, 4): "indianred",
    (4, 8): "purple",
    (4, 4): "orange",
}


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


def get_exit_ratio_for_label(label):
    """
    Extract i, j, k from the string and return EXIT_RATIOS[(i, j, k)]
    """
    i, j, k, pb, pc = get_params_from_label(label)
    return EXIT_RATIOS.get((i, j, k, pb, pc), "Not found")


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
        gridcolor="black",
        gridwidth=1,
        mirror=True,
        type="log",
        dtick="D10",  # Major ticks at powers of 10
        minor=dict(
            showgrid=True,
            gridcolor="lightgrey",
            gridwidth=0.5,
            ticklen=1,
            dtick="D1",  # Minor ticks at 2x10, 3x10, etc.
            # tickmode="array",
            # tickvals=[],  # No minor tick values
        ),
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
    )
    # Set axis titles
    fig.update_xaxes(title_text="Exit Stage", row=1, col=1)
    fig.update_xaxes(title_text="(Backbone, Exit) Precision", row=1, col=2)
    fig.update_yaxes(title_text="Energy-Delay Product (ET)<br>(J x cycles)", row=1, col=1)

    # # Set y-axis type to log for both subplots
    # fig.update_yaxes(type="log", row=1, col=1)
    # fig.update_yaxes(type="log", row=1, col=2)

    style_legend(fig)


def scatter_plot_edp(dfs, labels, fig_path):
    """
    Plot a scatter plot of the EDP across stages for different precision combinations
    """
    # Add label column to each df
    for df, label in zip(dfs, labels):
        df["label"] = label

    # Create one big df containing all entries
    df = pd.concat(dfs)

    # Create a subplot with 1 row and 2 columns, sharing the y-axis
    fig = make_subplots(rows=1, cols=2, subplot_titles=SUBPLOT_TITLES, shared_yaxes=True)

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

    # Plot cumulative EDP across stages for each label
    for label in labels:
        df_min_label = df_min[df_min["label"] == label]
        _, _, _, pb, pc = get_params_from_label(label)
        legend_name = get_legend_name_from_pb_pc(pb, pc)
        color = PRECISION_COLORS.get((pb, pc), "black")
        trace = go.Scatter(
            x=df_min_label["stage_id"],
            y=df_min_label["cum_edp"],
            mode="lines+markers",
            name=legend_name,
            line=dict(color=color),
        )
        fig.add_trace(trace, row=1, col=1)

    # Compute average EDP based on exit ratios
    data_avg = []
    for label in labels:
        df_min_label = df_min[df_min["label"] == label]
        exit_ratios = get_exit_ratio_for_label(label)
        avg_edp = sum(df_min_label["cum_edp"] * exit_ratios)
        data_avg.append({"label": label, "avg_edp": avg_edp})

    # Create dataframe from data_avg
    df_avg = pd.DataFrame(data_avg)
    # Update the label column to only include the backbone and exit precision as a tuple
    df_avg["label"] = [f"({pb}, {pc})" for pb, pc in PRECISION_COMBINATIONS]
    print(df_avg)
    # Plot the average EDP for each label
    for pb, pc in PRECISION_COMBINATIONS:
        df_avg_label = df_avg[df_avg["label"] == f"({pb}, {pc})"]
        color = PRECISION_COLORS.get((pb, pc), "black")
        trace = go.Bar(x=df_avg_label["label"], y=df_avg_label["avg_edp"], name=f"INT{pb}+{pc}", marker_color=color)
        trace.showlegend = False
        fig.add_trace(trace, row=1, col=2)

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

    # Process each model configuration
    for combination in EXIT_RATIOS:
        i, j, k, pb, pc = combination
        label = f"Model {i}, {j}, {k} pb{pb} pc{pc}"
        labels.append(label)
        # Load in the pickled data
        data_path = os.path.join(DATA_PATH, f"model_{i}_{j}_{k}_pb{pb}_pc{pc}.pickle")
        with open(data_path, "rb") as f:
            data = pickle.load(f)
        # Create a dataframe from the data
        df = pd.DataFrame(data)
        df["pb"] = pb
        df["pc"] = pc
        df["model"] = f"{i}_{j}_{k}"
        dfs.append(df)

    # Generate the scatter plot
    scatter_plot_edp(dfs, labels, FIG_PATH)


if __name__ == "__main__":
    main()
