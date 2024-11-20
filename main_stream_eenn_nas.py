import argparse
import logging as _logging
import os
import re

from stream.api import optimize_allocation_ga
from stream.visualization.memory_usage import plot_memory_usage
from stream.visualization.schedule import (
    visualize_timeline_plotly,
)


def parse_id(arg):
    return list(map(int, arg.split(",")))


parser = argparse.ArgumentParser(description="Run the EENN experiment.")
parser.add_argument(
    "-relpath",
    "--relative_path",
    dest="relpath",
    required=True,
    type=str,
    help="The relative path to the model directory.",
)
parser.add_argument(
    "-id", "--model_id", dest="id", required=True, type=parse_id, help="The model id as a csv of numbers x,y,.."
)

args = parser.parse_args()

_logging_level = _logging.INFO
_logging_format = "%(asctime)s - %(name)s.%(funcName)s +%(lineno)s - %(levelname)s - %(message)s"
_logging.basicConfig(level=_logging_level, format=_logging_format)


BASE_MODEL_DIR = "./stream/inputs/eenn/workload/nas/"
############################################INPUTS############################################
accelerator = "stream/inputs/eenn/hardware/edge_tpu_like_quad_core.yaml"
model_path = os.path.join(BASE_MODEL_DIR, args.relpath)
workload_path = os.path.join(model_path, "model.onnx")
mapping_path = "stream/inputs/eenn/mapping/edge_tpu_like_quad_core.yaml"
mode = "lbl"
nb_ga_generations = 64
nb_ga_individuals = 64
layer_stacks = list((i,) for i in range(200))
precision_backbone = 8
precision_classifier = 8
model_id_list = args.id
model_id_str = "_".join(map(str, model_id_list))
##############################################################################################

################################PARSING###############################
hw_name = accelerator.split("/")[-1].split(".")[0]
wl_name = re.split(r"/|\.", workload_path)[-1]
if wl_name == "onnx":
    wl_name = re.split(r"/|\.", workload_path)[-2]
experiment_id = os.path.join(args.relpath, f"eenn-{hw_name}-model_{model_id_str}-{mode}")
output_path = os.path.join("outputs-eenn/nas", args.relpath)
######################################################################

##############PLOTTING###############
plot_full_schedule = True
draw_dependencies = True
plot_data_transfer = True
section_start_percent = (0,)
percent_shown = (100,)
#####################################


################################PATHS################################
timeline_fig_path_plotly = os.path.join(output_path, experiment_id, "schedule.html")
memory_fig_path = os.path.join(output_path, experiment_id, "memory.png")
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
    output_path=output_path,
    skip_if_exists=True,
    model_id=model_id_list,
    precision_backbone=precision_backbone,
    precision_classifier=precision_classifier,
    model_path=model_path,
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
