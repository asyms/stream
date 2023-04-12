from multi_dnn_raw import workload
file = open("multi_dnn.py", 'w')

def reduce_spatial_dimension(loop_dimension, spatial_size, loop_dim_size):
    max_unrolling = spatial_size

    if max_unrolling > loop_dim_size[loop_dimension]:
        max_unrolling = loop_dim_size[loop_dimension]
        print("too big, reduce loop dimension size to", max_unrolling)

    return max_unrolling

file.write('workload = {')

key_list = ['data_flow_a', 'data_flow_b', 'data_flow_c', 'data_flow_d', 'data_flow_e', 'data_flow_f', 'data_flow_g']

for index, layer in workload.items():
    for key in key_list:

        for spatial_dimension, item in layer[key].items():
            print(layer['loop_dim_size'])
            print(item)
            loop_dimension = item[0]
            spatial_size = item[1]
            new_max_unrolling = reduce_spatial_dimension(loop_dimension, spatial_size, layer['loop_dim_size'])
            
            new_dataflow = (loop_dimension, new_max_unrolling)
            layer[key][spatial_dimension] = new_dataflow
            print(item)

    file.write('\n\t%s: {\n' % index)

    total_number_keys = len(layer)
    current_count = 0

    unused_keys = ["spatial_mapping", "core_allocation", "data_flow_a", "data_flow_b", "data_flow_c", "data_flow_d", "data_flow_e", "data_flow_f", "data_flow_g", "stride"]

   # add operator_type field
    if layer['operand_precision']['W'] == 0:
        operator_type = "MaxPool"
    else:
        operator_type = "Conv"
    
    file.write("\t\t'%s': '%s'" % ("operator_type", operator_type))
    if current_count != total_number_keys-1:
        file.write(",\n")
    current_count += 1

    for key, value in layer.items():
        if key in unused_keys:
            continue

        # rename key
        if key == "equation_relations":
            key = "dimension_relations"

        if type(value) == str:
            file.write("\t\t'%s': '%s'" % (key, value))
        else:
            file.write("\t\t'%s': %s" % (key, value))

        if current_count != total_number_keys-1:
            file.write(",\n")

        current_count += 1

    # add constant_operands field
    file.write("\t\t'%s': %s" % ("constant_operands", "['W']"))
    if current_count != total_number_keys-1:
        file.write(",\n")
    current_count += 1

    # add operand_source_dimension_mapping
    pred = layer['operand_source']['I']
    if pred != []:
        if 'G' in workload[pred[0]]['loop_dim_size']:
            file.write("\t\t'%s': %s" % ("operand_source_dimension_mapping", "{'I': {'IX': 'OX', 'IY': 'OY', 'C': 'G'}}"))
        else:
            file.write("\t\t'%s': %s" % ("operand_source_dimension_mapping", "{'I': {'IX': 'OX', 'IY': 'OY', 'C': 'K'}}"))
        if current_count != total_number_keys-1:
            file.write(",\n")
        current_count += 1

    # add padding
    file.write("\t\t'%s': %s" % ("padding", "{'IY': (0, 0), 'IX': (0, 0)}"))
    if current_count != total_number_keys-1:
        file.write(",\n")
    current_count += 1

    file.write('\n\t}\n\t,')


file.write('\n}')