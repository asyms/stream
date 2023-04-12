from stream.visualization.node_hw_performances import (
    visualize_node_hw_performances_pickle,
)


pickle_filepath = "outputs/saved_CN_HW_cost-heterogeneous_quadcore-onnx-CNmode_1-hintloop_[('OY', 'all')].pickle"
fig_path = "outputs/nodes_hw_cost-heterogeneous_quadcore-onnx-oy_all.png"

visualize_node_hw_performances_pickle(pickle_filepath, fig_path=fig_path)
