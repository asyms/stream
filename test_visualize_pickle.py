from stream.visualization.node_hw_performances import (
    visualize_node_hw_performances_pickle,
)


pickle_filepath = "outputs\saved_cn_hw_cost-heterogeneous_quadcore-resnet18-hintloop_.pickle"
fig_path = "outputs/test.png"

visualize_node_hw_performances_pickle(pickle_filepath, fig_path=fig_path)
