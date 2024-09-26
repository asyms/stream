import logging as _logging
import re

from stream.api import optimize_allocation_ga
from stream.visualization.memory_usage import plot_memory_usage
from stream.visualization.schedule import (
    visualize_timeline_plotly,
)

_logging_level = _logging.INFO
_logging_format = "%(asctime)s - %(name)s.%(funcName)s +%(lineno)s - %(levelname)s - %(message)s"
_logging.basicConfig(level=_logging_level, format=_logging_format)

############################################INPUTS############################################
accelerator = "stream/inputs/eenn/hardware/edge_tpu_like_quad_core.yaml"
workload_path = "stream/inputs/eenn/workload/conv-only.onnx"
mapping_path = "stream/inputs/eenn/mapping/edge_tpu_like_quad_core.yaml"
mode = "lbl"
nb_ga_generations = 16
nb_ga_individuals = 16
layer_stacks = list((i,) for i in range(120))
##############################################################################################

################################PARSING###############################
hw_name = accelerator.split("/")[-1].split(".")[0]
wl_name = re.split(r"/|\.", workload_path)[-1]
if wl_name == "onnx":
    wl_name = re.split(r"/|\.", workload_path)[-2]
experiment_id = f"eenn-{hw_name}-{wl_name}-{mode}"
######################################################################

##############PLOTTING###############
plot_file_name = f"-{experiment_id}-"
plot_full_schedule = True
draw_dependencies = True
plot_data_transfer = True
section_start_percent = (0,)
percent_shown = (100,)
#####################################


################################PATHS################################
timeline_fig_path_plotly = f"outputs-eenn/{experiment_id}-schedule.html"
memory_fig_path = f"outputs-eenn/{experiment_id}-memory.png"
#####################################################################

scme = optimize_allocation_ga(
    hardware=accelerator,
    workload=workload_path,
    mapping=mapping_path,
    mode=mode,
    layer_stacks=layer_stacks,
    nb_ga_generations=nb_ga_generations,
    nb_ga_individuals=nb_ga_individuals,
    experiment_id=experiment_id,
    output_path="outputs-eenn",
    skip_if_exists=False,
)

# Plotting schedule timeline of best SCME
visualize_timeline_plotly(
    scme,
    draw_dependencies=draw_dependencies,
    draw_communication=plot_data_transfer,
    fig_path=timeline_fig_path_plotly,
)
# Plotting memory usage of best SCME
plot_memory_usage(scme, section_start_percent, percent_shown, fig_path=memory_fig_path)
