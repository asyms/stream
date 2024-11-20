import os

import onnx


def find_classifier_subnets(model_path):
    model = onnx.load(model_path)
    graph = model.graph
    model_id = []
    for node in graph.node:
        if "classifier" in node.name and any("block" in input_name for input_name in node.input):
            input_name = next(input_name for input_name in node.input if "block" in input_name)
            block_id = int(input_name.split("/")[1].split(".")[1])
            model_id.append(block_id)

    return sorted(model_id)


def main():
    base_dir = "./stream/inputs/eenn/workload/nas/"
    model_dirs = []

    for root, dirs, files in sorted(os.walk(base_dir)):
        if "model.onnx" in files:
            rel_path = os.path.relpath(root, base_dir)
            model_path = os.path.join(root, "model.onnx")
            classifier_subnets = find_classifier_subnets(model_path)
            if classifier_subnets:
                model_dirs.append((rel_path, classifier_subnets))

    for model_dir in model_dirs:
        print(f"Directory: {model_dir[0]}, Classifier subnets: {model_dir[1]}")


if __name__ == "__main__":
    main()
