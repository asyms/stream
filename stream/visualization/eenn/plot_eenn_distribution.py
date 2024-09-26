import os

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import plotly.io as pio
from plotly.subplots import make_subplots
from zigzag.utils import pickle_load

pio.kaleido.scope.mathjax = None

# Global variables
FIG_PATH = "outputs-eenn/distribution_cum_energy.pdf"
DATA_PATH = "outputs-eenn/eenn_data.pickle"


def style_legend(fig):
    # Update layout to place the legend on top and style the box
    fig.update_layout(
        legend=dict(
            yanchor="bottom",
            y=1.05,  # Position the legend above the plot
            xanchor="center",
            x=0.5,  # Center the legend horizontally
            bgcolor="aliceblue",  # Light blue background with transparency
            bordercolor="Black",
            borderwidth=2,
            font=dict(family="Arial", size=16, color="black"),
        )
    )


def style_profesionally(fig, df):
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
        showline=True,  # add line at x=0
        linecolor="black",  # line color
        linewidth=1,  # line size
        ticks="outside",  # ticks outside axis
        tickfont=font_dict,  # tick label font
        tickwidth=1,  # tick width
        tickcolor="black",  # tick color
        showgrid=True,
        gridcolor="lightgrey",
        gridwidth=0.5,
        # tickformat='.1e',
        # showexponent = 'last',
        # exponentformat = 'e',
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
    fig.update_xaxes(title_text="Exit Stages", row=1, col=1)
    fig.update_yaxes(title_text="Cumulative Latency (norm.)", row=1, col=1, gridwidth=1, secondary_y=False)
    fig.update_xaxes(title_text="Exit Stages", row=1, col=2)
    fig.update_yaxes(title_text="Cumulative Energy (norm.)", row=1, col=2, gridwidth=1, secondary_y=False)

    # Align the right y-axis with the left for both subplots
    fig.update_yaxes(matches="y", row=1, col=1, secondary_y=True)  # First subplot
    fig.update_yaxes(matches="y2", row=1, col=2, secondary_y=True)  # Second subplot (doesn't seem to work)

    # Get the minimum and maximum across all relevant y-axes
    min_value_y2 = min(df["cum_energy_fraction"].min(), df["cum_macs_fraction"].min())
    max_value_y2 = max(df["cum_energy_fraction"].max(), df["cum_macs_fraction"].max())

    # Ensure both left and right axes have the same range for subplot 2
    fig.update_yaxes(range=[min_value_y2 * 0.9, max_value_y2 * 1.1], row=1, col=2, secondary_y=False)
    fig.update_yaxes(range=[min_value_y2 * 0.9, max_value_y2 * 1.1], row=1, col=2, secondary_y=True)

    style_legend(fig)


def save_plot():
    global FIG_PATH

    # Convert DATA_LIST to a DataFrame
    data = pickle_load(DATA_PATH)
    df = pd.DataFrame(data)

    # Create subplots: 1 row, 2 columns
    fig = make_subplots(rows=1, cols=2, horizontal_spacing=0.2, specs=[[{"secondary_y": True}, {"secondary_y": True}]])

    # Cumulative latency fraction plot
    latency_fig = px.box(df, x="stage_id", y="cum_latency_fraction", points="all")

    # Energy distribution plot with energy fraction
    energy_fig = px.box(df, x="stage_id", y="cum_energy_fraction", points="all")

    # Scatter line plot with dashes for cum_macs_fraction on secondary y-axis (subplot 1)  .
    df_unique = df.drop_duplicates(subset=["stage_id", "cum_macs_fraction"])
    fig.add_trace(
        go.Scatter(
            x=df_unique["stage_id"],
            y=df_unique["cum_macs_fraction"],
            mode="lines+markers",  # Adds markers to the line plot
            line=dict(dash="dash", color="black"),  # Line color set to black
            marker=dict(size=8, color="black"),  # Marker size and color
            name="Fraction of MACs Completed",
            showlegend=True,  # Show legend for this trace
        ),
        row=1,
        col=1,
        secondary_y=True,
    )

    # For the second subplot, hide the legend entry
    fig.add_trace(
        go.Scatter(
            x=df_unique["stage_id"],
            y=df_unique["cum_macs_fraction"],
            mode="lines+markers",  # Adds markers to the line plot
            line=dict(dash="dash", color="black"),  # Line color set to black
            marker=dict(size=8, color="black"),  # Marker size and color
            name="Cumulative MACs Fraction",
            showlegend=False,  # Hide legend for this trace
        ),
        row=1,
        col=2,
        secondary_y=True,
    )

    # Add latency plot traces to the first subplot
    for trace in latency_fig.data:
        fig.add_trace(trace, row=1, col=1)

    # Add energy plot traces to the second subplot
    for trace in energy_fig.data:
        fig.add_trace(trace, row=1, col=2)

    style_profesionally(fig, df)

    # Get the file extension
    file_extension = os.path.splitext(FIG_PATH)[1].lower()

    # Save figure based on the extension
    if file_extension == ".html":
        fig.write_html(FIG_PATH)
    elif file_extension in [".png", ".pdf"]:
        fig.write_image(FIG_PATH, engine="kaleido")
    else:
        print("Unsupported file format. Please provide either a .html or .png extension.")
    print(f"Figure saved at {FIG_PATH}")


if __name__ == "__main__":
    # Call the function to save the plot
    save_plot()
