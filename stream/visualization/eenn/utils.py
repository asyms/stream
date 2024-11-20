import os

import plotly.graph_objects as go

PARETO_COLOR = "gold"


def style_legend(fig, title=None):
    # Update layout to place the legend on top and style the box
    legend_dict = dict(
        title=dict(text=title, side="top center"),
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
    if not title:
        legend_dict.pop("title")
    fig.update_layout(legend=legend_dict)


def style_figure(fig, X_AXIS_TITLE, Y_AXIS_TITLE):
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
    fig.update_xaxes(title_text=X_AXIS_TITLE)
    fig.update_yaxes(title_text=Y_AXIS_TITLE)


def add_star(fig, df, EDP_UB, PLOT_X_METRIC, PLOT_Y_METRIC, colors):
    # Get the point closest to 88% accuracy that has 5 exits from df
    df_5 = df[df["nb_exits"] == 5]
    closest_point = df_5.iloc[(df_5["accuracy"] - 88).abs().argsort()[:1]]
    # Check that this point is below the EDP upper bound
    if closest_point["avg_edp"].iloc[0] > EDP_UB:
        raise ValueError("The point closest to 88% accuracy with 5 exits is above the EDP upper bound.")
    # Add star symbol to this point
    x = closest_point[PLOT_X_METRIC].iloc[0]
    y = closest_point[PLOT_Y_METRIC].iloc[0]
    color = colors[5]
    star_trace = go.Scatter(
        x=[x],
        y=[y],
        mode="markers",
        marker=dict(color=color, size=14, symbol="star", line=dict(width=0.3, color="black")),
    )
    star_trace.showlegend = False
    star_trace.name = "TODO"
    fig.add_trace(star_trace)


def add_pareto(fig):
    # Extract points from the traces
    points = []
    names = []
    for trace in fig.data:
        for x, y in zip(trace.x, trace.y):
            points.append((x, y))
            names.append(trace.name)

    # Sort points and names together like this
    points, names = zip(*sorted(zip(points, names), key=lambda pair: (-pair[0][0], pair[0][1])))

    # Find the Pareto front
    pareto_points = []
    pareto_names = []
    current_best_y = float("inf")
    for (x, y), name in zip(points, names):
        if y < current_best_y:
            pareto_points.append((x, y))
            current_best_y = y
            pareto_names.append(name)

    # Separate the Pareto points into X and Y lists
    pareto_x, pareto_y = zip(*pareto_points)

    # Add the Pareto line to the figure
    pareto_trace = go.Scatter(
        x=pareto_x, y=pareto_y, mode="lines", name="Pareto Front", line=dict(color=PARETO_COLOR, width=2, dash="dash")
    )
    pareto_trace.showlegend = False
    fig.add_trace(pareto_trace)


def save_fig(fig, fig_path):
    # Save figure based on the extension
    file_extension = os.path.splitext(fig_path)[1].lower()
    if file_extension == ".html":
        fig.write_html(fig_path)
    elif file_extension in [".png", ".pdf"]:
        fig.write_image(fig_path, engine="kaleido")
    else:
        print("Unsupported file format. Please provide either a .html or .png extension.")
    print(f"Figure saved at {fig_path}")
