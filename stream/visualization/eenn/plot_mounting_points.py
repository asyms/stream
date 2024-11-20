import os
import pickle

import pandas as pd
import plotly.graph_objects as go
import plotly.io as pio
from plotly.subplots import make_subplots

pio.kaleido.scope.mathjax = None

DATA_PATH = "outputs-eenn/mounting_points_data/pareto/"
FIG_PATH = "outputs-eenn/mounting_points_data/pareto/mounting_points_et.pdf"
EXIT_RATIOS = {
    (0, 2, 6): [0.155, 0.2443, 0.2435, 0.3572],
    (0, 4, 6): [0.1394, 0.3734, 0.1682, 0.319],
    (0, 4, 8): [0.0911, 0.2841, 0.3478, 0.277],
    (2, 3, 6): [0.3595, 0.1177, 0.1686, 0.3542],
    (3, 5, 8): [0.2549, 0.1531, 0.3114, 0.2806],
}
MODEL_COLORS = {
    "Model 0, 2, 6": "blue",
    "Model 0, 4, 6": "green",
    "Model 0, 4, 8": "indianred",
    "Model 2, 3, 6": "purple",
    "Model 3, 5, 8": "orange",
}


def get_legend_name_from_label(label):
    ids = [int(i) for i in label.lstrip("Model ").split(", ")]
    # Convert the ids to letters where 0 is A, 1 is B, etc.
    letters = [chr(65 + i) for i in ids]
    letters_str = ", ".join(letters)
    return f"{letters_str}, K"


def add_hline_at_y_0(fig):
    # Add a horizontal line at y=0 on both plots because for some reason it's not there
    fig.add_hline(y=0, line=dict(color="lightgray", width=1))
    # TODO: Figure out how to add it to second subplot correctly
    # Add a horizontal line at y=0 on the second subplot
    # x_range = fig.layout['xaxis2'].range
    # print(x_range)
    # trace_hline = go.Scatter(
    #     x=x_range,
    #     y=[0, 0],
    #     mode="lines",
    #     line=dict(color="lightgray", width=1),
    #     showlegend=False
    # )
    # fig.add_trace(trace_hline, row=1, col=2)


def style_legend(fig):
    # Update layout to place the legend on top and style the box
    fig.update_layout(
        legend=dict(
            title=dict(text="Mounting Points of 4 Exit Stages (See Table I)", side="top center"),
            yanchor="bottom",
            y=1.1,  # Position the legend above the plot
            xanchor="center",
            x=0.5,  # Center the legend horizontally
            bgcolor="aliceblue",  # Light blue background with transparency
            bordercolor="Black",
            borderwidth=2,
            font=dict(family="Arial", size=12, color="black"),
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
    fig.update_xaxes(title_text="Exit Stage", row=1, col=1)
    fig.update_xaxes(title_text="Model", tickvals=[], ticktext=[], row=1, col=2)
    fig.update_yaxes(title_text="Energy-Delay Product (ET)<br>(J x cycles)", row=1, col=1)

    style_legend(fig)
    add_hline_at_y_0(fig)


def scatter_plot_edp(dfs, labels, fig_path):
    """
    Plot a scatter plot of the edp across stages for different mounting points
    """
    # Add label column to each df
    for df, label in zip(dfs, labels):
        df["label"] = label
    # Create one big df containing all entries
    df = pd.concat(dfs)

    # Create a subplot with 1 row and 2 column
    fig = make_subplots(
        rows=1,
        cols=2,
        subplot_titles=("ET at different exit stages", "Average ET based on exit ratios"),
        shared_yaxes=True,
    )

    # Convert energy from pJ to J
    df["energy"] = df["energy"] / 1e12

    # Add column that computes the EDP for each stage (called ET in the labels)
    df["edp"] = df["latency"] * df["energy"]

    # Create new dataframe containing for each label and each stage the minimum EDP, and its associated energy and latency from the right row
    min_edp = df.groupby(["label", "stage_id"]).apply(lambda x: x.loc[x["edp"].idxmin()]).reset_index(drop=True)
    # For each label, compute the cumulative EDP across stages by summing the energies and latencies and then multiplying them
    for label in labels:
        min_edp_label = min_edp[min_edp["label"] == label]
        # Drop all columns except 'stage_id', 'energy', 'latency' and 'edp'
        min_edp_label = min_edp_label[["stage_id", "energy", "latency", "edp"]]
        # Compute the cumulative energy and latency
        min_edp_label["cum_energy"] = min_edp_label["energy"].cumsum()
        min_edp_label["cum_latency"] = min_edp_label["latency"].cumsum()
        # Compute the cumulative EDP by multiplying the cumumative energy and latency
        min_edp_label["cum_edp"] = min_edp_label["cum_energy"] * min_edp_label["cum_latency"]
        # Set the original min_edp df cum_edp column to the computed cum_edp
        min_edp.loc[min_edp["label"] == label, "cum_edp"] = min_edp_label["cum_edp"]
    print(min_edp)
    # Plot cumulative EDP across stages for each label
    for label in labels:
        min_edp_label = min_edp[min_edp["label"] == label]
        color = MODEL_COLORS.get(label, "black")
        name = get_legend_name_from_label(label)
        trace = go.Scatter(
            x=min_edp_label["stage_id"],
            y=min_edp_label["cum_edp"],
            mode="lines+markers",
            name=name,
            line=dict(color=color),
        )
        fig.add_trace(trace, row=1, col=1)

    style_profesionally(fig)

    # Add annotation that shows for the different models and stages the exit ratios (ER) in a table
    # annotations = []
    # for label in labels:
    #     min_edp_label = min_edp[min_edp["label"] == label]
    #     exit_ratios = EXIT_RATIOS[int(label.split(" ")[1])]
    #     # Create a table with the exit ratios
    #     table = go.Table(
    #         header=dict(values=["Stage ID", "Exit Ratio"]),
    #         cells=dict(values=[min_edp_label["stage_id"], exit_ratios]),
    #         domain=dict(x=[0, 0.5], y=[0, 0.5]),
    #     )
    #     fig.add_trace(table, row=1, col=1)

    # TODO: Add second subplot that shows the average ET based on the exit ratios (bar chart with x the different labels)
    data_avg = []
    assert len(labels) == len(EXIT_RATIOS)
    for label, k in zip(labels, EXIT_RATIOS):
        min_edp_label = min_edp[min_edp["label"] == label]
        exit_ratios = EXIT_RATIOS[k]
        avg_et = sum(min_edp_label["cum_edp"] * exit_ratios)
        print(f"Average ET for {label}: {avg_et}")
        # Add entry to df_avg
        data_avg.append({"label": label, "avg_et": avg_et})
    # Create df from data_avg
    df_avg = pd.DataFrame(data_avg)
    # Plot the average ET for each label
    trace = go.Bar(
        x=df_avg["label"],
        y=df_avg["avg_et"],
        name="Average EDP",
        marker_color=[MODEL_COLORS[label] for label in df_avg["label"]],
    )
    trace.showlegend = False
    fig.add_trace(trace, row=1, col=2)

    # Save figure based on the extension
    file_extension = os.path.splitext(fig_path)[1].lower()
    if file_extension == ".html":
        fig.write_html(fig_path)
    elif file_extension in [".png", ".pdf"]:
        fig.write_image(fig_path, engine="kaleido")
    else:
        print("Unsupported file format. Please provide either a .html or .png extension.")
    print(f"Figure saved at {fig_path}")


if __name__ == "__main__":
    dfs = []
    labels = [f"Model {i}, {j}, {k}" for (i, j, k) in EXIT_RATIOS]
    for model_id in EXIT_RATIOS:
        i, j, k = model_id
        # Load in the pickled data
        data_path = os.path.join(DATA_PATH, f"model_{i}_{j}_{k}.pickle")
        with open(data_path, "rb") as f:
            data = pickle.load(f)
        # Create a dataframe from the data
        df = pd.DataFrame(data)
        dfs.append(df)
    scatter_plot_edp(dfs, labels, FIG_PATH)
