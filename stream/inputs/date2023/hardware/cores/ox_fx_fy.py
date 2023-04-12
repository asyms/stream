from zigzag.classes.hardware.architecture.core import Core

import stream.inputs.date2023.hardware.cores.core_definition as core_def
import stream.inputs.date2023.hardware.cores.core_description as core_desc


def get_dataflows(quad_core=False):
    if quad_core:
        return [
            {
                "D1": ("OX", core_def.quad_core_multiplier_array_size_3D[0]),
                "D2": ("FX", core_def.quad_core_multiplier_array_size_3D[1]),
                "D3": ("FY", core_def.quad_core_multiplier_array_size_3D[2]),
            },
        ]
    else:
        return [
            {
                "D1": ("OX", core_def.single_core_multiplier_array_size_3D[0]),
                "D2": ("FX", core_def.single_core_multiplier_array_size_3D[1]),
                "D3": ("FY", core_def.single_core_multiplier_array_size_3D[2]),
            }
        ]


def get_core(id, quad_core=False):
    operational_array = core_desc.get_multiplier_array_3D(quad_core)
    memory_hierarchy = core_desc.get_memory_hierarchy_OX_FX_FY_dataflow(operational_array)
    dataflows = get_dataflows(quad_core)
    core = Core(id, operational_array, memory_hierarchy, dataflows)
    return core
