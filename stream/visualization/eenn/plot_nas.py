import os
import pickle

import pandas as pd
import plotly.graph_objects as go
import plotly.io as pio
from plotly.subplots import make_subplots
from utils import add_pareto, add_star, save_fig, style_figure, style_legend

pio.kaleido.scope.mathjax = None

MODEL_DIR = "./stream/inputs/eenn/workload/nas/"
DF_CACHE_PATH = "./outputs-eenn/nas/df_cache.pickle"  # To cache the created dataframe for speedup

#################### For the average ET vs. Accuracy plot ####################
# For the average ET vs. Accuracy plot
# FIG_PATH = "outputs-eenn/nas/nas_accuracy_et.pdf"
# X_AXIS_TITLE = "Accuracy (%)"
# Y_AXIS_TITLE = "Average ET (J x cycles)"
# PLOT_X_METRIC = "accuracy"
# PLOT_Y_METRIC = "avg_edp"
# ADD_PARETO = True

#################### For the average ET vs. ET Reduction plot ####################
FIG_PATH = "outputs-eenn/nas/nas_et_average_et_reduction.pdf"
X_AXIS_TITLE = "ET Reduction (Factor)"
Y_AXIS_TITLE = "Average ET (J x cycles)"
PLOT_X_METRIC = "edp_reduction"
PLOT_Y_METRIC = "avg_edp"
ADD_PARETO = False

EDP_UB = 2000  # Upper bound for the EDP scatter plot
# Define the colors for each precision combination
MOUNTING_POINTS_COLORS = {
    2: "blue",
    3: "green",
    4: "indianred",
    5: "purple",
    6: "orange",
    7: "red",
    8: "brown",
    9: "cyan",
    10: "magenta",
}
MARKER_SIZE = 8


def scatter_plot_edp_accuracy(df, labels, exit_ratios, fig_path):
    """
    Plot a scatter plot of the EDP across stages for different precision combinations
    """
    # Create a figure
    fig = make_subplots()

    # Convert energy from pJ to J
    df["energy"] = df["energy"] / 1e12

    # Add column that computes the EDP for each stage
    df["edp"] = df["latency"] * df["energy"]
    # Add column that computes the base EDP for each stage based on only the 'block'
    df["edp_base"] = df["block_latency"] * df["energy"]

    # Create new dataframe containing for each label and each stage the minimum EDP
    min_data = []
    for label in labels:
        df_label = df[df["label"] == label]
        for stage_id in df_label["stage_id"].unique():
            df_label_stage = df_label[df_label["stage_id"] == stage_id]
            min_edp = df_label_stage.loc[df_label_stage["edp"].idxmin()]
            min_data.append(min_edp)
    df_min = pd.DataFrame(min_data)

    # Compute for each label the cumulative EDP across stages by summing the energies and latencies and then multiplying them
    for label in labels:
        df_min_label = df_min[df_min["label"] == label]
        # With early exits
        df_min_label["cum_energy"] = df_min_label["energy"].cumsum()
        df_min_label["cum_latency"] = df_min_label["latency"].cumsum()
        df_min_label["cum_edp"] = df_min_label["cum_energy"] * df_min_label["cum_latency"]
        df_min.loc[df_min["label"] == label, "cum_edp"] = df_min_label["cum_edp"]
        # Without early exits (base)
        df_min_label["cum_energy_base"] = df_min_label["energy"].cumsum()
        df_min_label["cum_latency_base"] = df_min_label["block_latency"].cumsum()
        df_min_label["cum_edp_base"] = df_min_label["cum_energy_base"] * df_min_label["cum_latency_base"]
        df_min.loc[df_min["label"] == label, "cum_edp_base"] = df_min_label["cum_edp_base"]

    # Compute average EDP based on exit ratios
    data_avg = []
    for label, exit_ratio in zip(labels, exit_ratios):
        df_min_label = df_min[df_min["label"] == label]
        avg_edp = sum(df_min_label["cum_edp"] * exit_ratio)
        # Get base EDP from the last stage
        last_stage = df_min_label["stage_id"].max()
        edp_base = df_min_label[df_min_label["stage_id"] == last_stage]["cum_edp_base"].iloc[0]
        edp_reduction_factor = edp_base / avg_edp
        print(label, edp_reduction_factor)
        data_avg.append(
            {
                "label": label,
                "avg_edp": avg_edp,
                "accuracy": 100 * df_min_label["accuracy"].iloc[0],
                "edp_base": edp_base,
                "edp_reduction": edp_reduction_factor,
            }
        )

    # Create dataframe from data_avg
    df_avg = pd.DataFrame(data_avg)

    # Add nb_exits column to df_avg
    df_avg["nb_exits"] = df_avg["label"].apply(lambda x: len(x.split("_")) - 2)

    # Get subset of dataframe with average EDP below the upper bound
    df_avg_bounded = df_avg[df_avg["avg_edp"] < EDP_UB]

    # Plot the average EDP for each label
    for nb_exits in sorted(df_avg_bounded["nb_exits"].unique()):
        nb_exits = int(nb_exits)
        df_avg_bounded_nb_exits = df_avg_bounded[df_avg_bounded["nb_exits"] == nb_exits]
        if df_avg_bounded_nb_exits.empty:
            continue
        color = MOUNTING_POINTS_COLORS[nb_exits]
        trace = go.Scatter(
            x=df_avg_bounded_nb_exits[PLOT_X_METRIC],
            y=df_avg_bounded_nb_exits[PLOT_Y_METRIC],
            mode="markers",
            name=f"{nb_exits}",
            marker=dict(color=color, size=MARKER_SIZE),
        )
        fig.add_trace(trace)

    # Add a pareto curve based on the traces present in the figure
    if ADD_PARETO:
        add_pareto(fig)

    # Add star
    add_star(fig, df_avg_bounded, EDP_UB, PLOT_X_METRIC, PLOT_Y_METRIC, colors=MOUNTING_POINTS_COLORS)

    # Style the figure
    style_figure(fig, X_AXIS_TITLE, Y_AXIS_TITLE)
    style_legend(fig, title="Number of exits")

    # Save the figure
    save_fig(fig, fig_path)


def get_model_id_data_path_and_stats_path_from_dir(model_dir):
    """
    Return the model id, path to the model data and stats file for the given model directory
    """
    files = os.listdir(model_dir)
    model_data_path = [f for f in files if "pickle" in f][0]
    model_id = [int(i) for i in model_data_path.rstrip(".pickle").split("_")[1:]]
    stats_path = [f for f in files if "stats" in f][0]
    return model_id, os.path.join(model_dir, model_data_path), os.path.join(model_dir, stats_path)


def get_model_str_from_id(model_id):
    """
    Return a string representation of the model id
    """
    return "_".join(map(str, model_id))


def main():
    """
    Main function to load data, process it, and generate the plot
    """

    # Get model configurations by scanning the files in MODEL_DIR
    model_dirs = []
    for root, dirs, files in os.walk(MODEL_DIR):
        if any("model.onnx" in f for f in files):
            model_dirs.append(root)

    if not os.path.exists(DF_CACHE_PATH):
        print("Cache does not exist. Generating dataframe...")
        # Iterate through the directories and create a dataframe for each model configuration
        exit_ratios = []
        dfs = []
        labels = []
        for model_dir in model_dirs:
            model_id, model_data_path, stats_path = get_model_id_data_path_and_stats_path_from_dir(model_dir)
            with open(model_data_path, "rb") as f:
                data = pickle.load(f)
            with open(stats_path, "r") as f:
                stats = eval(f.read())
            df = pd.DataFrame(data)
            model_str = get_model_str_from_id(model_id) + "_" + model_dir.split("/")[-2]
            df["label"] = model_str
            df["accuracy"] = stats["top1_accuracy"]
            exit_ratios.append(stats["exits_ratios"])
            dfs.append(df)
            labels.append(model_str)
        df = pd.concat(dfs)
        # Save the dataframes to a cache file for future use
        with open(DF_CACHE_PATH, "wb") as f:
            pickle.dump((df, labels, exit_ratios), f)
    else:
        print("Cache exists. Loading dataframe...")
        with open(DF_CACHE_PATH, "rb") as f:
            df, labels, exit_ratios = pickle.load(f)

    # Generate the scatter plot
    scatter_plot_edp_accuracy(df, labels, exit_ratios, FIG_PATH)


if __name__ == "__main__":
    main()
