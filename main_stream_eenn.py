import argparse
import logging as _logging
import os

from stream.api import optimize_allocation_ga
from stream.visualization.memory_usage import plot_memory_usage
from stream.visualization.schedule import (
    visualize_timeline_plotly,
)


def parse_tuple(arg):
    return tuple(map(int, arg.split(",")))


parser = argparse.ArgumentParser(description="Run the EENN experiment for one (hardware, model) pair.")
parser.add_argument(
    "-id",
    "--model_id",
    dest="id",
    required=True,
    type=parse_tuple,
    help="Block indices the intermediate exits are mounted after, e.g. 3,5,8.",
)
parser.add_argument(
    "-hw",
    "--hardware",
    dest="hw",
    required=False,
    default="stream/inputs/eenn/hardware/edge_tpu_like_quad_core.yaml",
    help="Path to the accelerator yaml.",
)
parser.add_argument(
    "-map",
    "--mapping",
    dest="mapping",
    required=False,
    default=None,
    help="Path to the mapping yaml. Defaults to stream/inputs/eenn/mapping/<hardware name>.yaml.",
)
parser.add_argument(
    "-wl",
    "--workload_dir",
    dest="workload_dir",
    required=False,
    default="stream/inputs/eenn/workload/focus",
    help="Directory holding the model_<i>_<j>_<k>/model.onnx workloads.",
)
parser.add_argument(
    "-o",
    "--output_root",
    dest="output_root",
    required=False,
    default="outputs-eenn/hw_sweep",
    help="Root directory for all outputs of this sweep.",
)
parser.add_argument(
    "-pb", "--precision_backbone", dest="pb", required=False, type=int, default=8, help="Backbone precision."
)
parser.add_argument(
    "-pc", "--precision_classifier", dest="pc", required=False, type=int, default=8, help="Classifier precision."
)
parser.add_argument(
    "-lb",
    "--last_block",
    dest="last_block",
    required=False,
    type=int,
    default=11,
    help="Index of the last backbone block, i.e. where the final exit sits (12-block MobileNetV2 -> 11).",
)
parser.add_argument(
    "-w",
    "--workload",
    dest="workload",
    required=False,
    default=None,
    help="Direct path to a model.onnx. Overrides the path derived from --workload_dir and -id, "
    "which is what the NAS workloads need since they are laid out as iter_<n>/net_<m>/.",
)
parser.add_argument(
    "--max_layers",
    dest="max_layers",
    required=False,
    type=int,
    default=200,
    help="Upper bound on the number of workload layers, used to build the layer stacks.",
)
parser.add_argument("-g", "--generations", dest="generations", required=False, type=int, default=64)
parser.add_argument("-i", "--individuals", dest="individuals", required=False, type=int, default=64)

args = parser.parse_args()

_logging_level = _logging.INFO
_logging_format = "%(asctime)s - %(name)s.%(funcName)s +%(lineno)s - %(levelname)s - %(message)s"
_logging.basicConfig(level=_logging_level, format=_logging_format)


############################################INPUTS############################################
accelerator = args.hw
hw_name = os.path.splitext(os.path.basename(accelerator))[0]
mapping_path = args.mapping or f"stream/inputs/eenn/mapping/{hw_name}.yaml"
model_id_str = "_".join(map(str, args.id))
workload_path = args.workload or os.path.join(args.workload_dir, f"model_{model_id_str}", "model.onnx")
# Stream needs the *complete* exit list: the fitness evaluator derives one EENN stage per entry
# of model_id, so a missing final exit silently drops the last backbone blocks and the final
# classifier. Two naming conventions are in use and both are accepted here:
#   - the focus/pareto workloads are named after their intermediate mounting points only
#     (model_3_5_8), so the final exit has to be appended;
#   - the NAS workloads already carry it as the last id (2,4,5,8,9,11).
# If -id already ends at the last block it is taken as complete.
if args.id[-1] == args.last_block:
    model_id = tuple(args.id)
elif args.id[-1] < args.last_block:
    model_id = tuple(args.id) + (args.last_block,)
else:
    raise ValueError(f"last mounting point ({args.id[-1]}) is beyond last_block ({args.last_block}).")
mode = "lbl"
nb_ga_generations = args.generations
nb_ga_individuals = args.individuals
# One stack per layer (mode is "lbl"). The bound only has to exceed the number of layers in
# the workload; entries beyond that are never referenced. It was 120, which covers the 4-exit
# focus models but not the larger NAS models: a 6- or 8-exit network has more classifier nodes
# and reaches layer id 120+, which fails with "Layer id 120 not in hint_loops". 200 matches
# what main_stream_eenn_nas.py has always used.
layer_stacks = list((i,) for i in range(args.max_layers))
precision_backbone = args.pb
precision_classifier = args.pc
##############################################################################################

for path in (accelerator, mapping_path, workload_path):
    if not os.path.exists(path):
        raise FileNotFoundError(f"Required input does not exist: {path}")

################################PATHS################################
# Layout (everything for one hardware + precision lives under one directory):
#   <output_root>/<hw_name>/pb<pb>_pc<pc>/
#       stage_data/model_<ids>.pickle   <- per-exit-stage GA trace, consumed by the plot scripts
#       runs/model_<ids>/scme.pickle, saved_cn_hw_cost.pickle, schedule.html, memory.png
# The sweep runner treats stage_data/model_<ids>.pickle as the completion marker, since it is
# written last. Keep skip_if_exists=False: reloading a cached scme.pickle skips the GA and would
# therefore leave the per-stage trace unwritten.
output_path = os.path.join(args.output_root, hw_name, f"pb{precision_backbone}_pc{precision_classifier}")
experiment_id = os.path.join("runs", f"model_{model_id_str}")
stage_data_path = os.path.join(output_path, "stage_data")
os.makedirs(stage_data_path, exist_ok=True)

timeline_fig_path_plotly = os.path.join(output_path, experiment_id, "schedule.html")
memory_fig_path = os.path.join(output_path, experiment_id, "memory.png")
#####################################################################

##############PLOTTING###############
plot_full_schedule = True
draw_dependencies = True
plot_data_transfer = True
section_start_percent = (0,)
percent_shown = (100,)
#####################################

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
    skip_if_exists=False,
    model_id=model_id,
    precision_backbone=precision_backbone,
    precision_classifier=precision_classifier,
    model_path=stage_data_path,
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
