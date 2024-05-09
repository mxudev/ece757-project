import onnx
from onnx import helper
from google.protobuf.json_format import MessageToDict
import onnxruntime
import math


def get_model_summary(onnx_model):
    summary = onnx.helper.printable_graph(onnx_model.graph)
    return summary


def get_convolution_and_gemm_nodes(onnx_model):
    convolution_nodes = []
    gemm_nodes = []
    node_outputs = set()

    # Find convolution and Gemm nodes
    for node in onnx_model.graph.node:
        if node.op_type == "Conv":
            convolution_nodes.append(node)
            for output in node.output:
                node_outputs.add(output)
        elif node.op_type == "Gemm":
            gemm_nodes.append(node)
            for input in node.input:
                node_outputs.add(input)

    # Find dependencies
    dependent_nodes = []
    for node in onnx_model.graph.node:
        for input in node.input:
            if input in node_outputs:
                dependent_nodes.append(node)
                for output in node.output:
                    node_outputs.add(output)

    return convolution_nodes, gemm_nodes, dependent_nodes

def get_input_size(model):
    input_shapes = {}
    for input in model.graph.input:
        input_shapes[input.name] = [dim.dim_value for dim in input.type.tensor_type.shape.dim]
    return input_shapes

def process_layers(model):
    # print(get_input_size(model))
    weight_shapes = get_input_size(model)
    outputs = {}
    outputs["data"] = weight_shapes["data"][1:] #C, H, W
    # print(outputs)
    for node in model.graph.node:
        print(node)
        if node.op_type == "BatchNormalization":
            x = node.input[0]
            if x in outputs:
                outputs[node.output[0]] = outputs[x] #Batch norm does't change shape
        elif node.op_type == "Conv":
            x = node.input[0]
            w = node.input[1]
            y = node.output[0]
            if x in outputs:
                input_shape = outputs[x]
                conv_dim = weight_shapes[w]
                #out_shape = [(W−K+2P)/S]+1
                hw_shape = math.floor((input_shape[1] - conv_dim[2] + 2 * node.attribute[3].ints[0]) / node.attribute[4].ints[0]) + 1
                c_shape = conv_dim[0]
                outputs[y] = (c_shape, hw_shape, hw_shape)
                # print(conv_dim)
                # breaks
        elif node.op_type == "Relu":
            x = node.input[0]
            y = node.output[0]
            if x in outputs:
                outputs[y] = outputs[x] #Relu doesn't change shape
        elif node.op_type == "MaxPool":
            x = node.input[0]
            y = node.output[0]
            if x in outputs:
                input_shape = outputs[x]
                hw_shape = math.floor((input_shape[1] - node.attribute[0].ints[0] + 2 * node.attribute[1].ints[0]) / node.attribute[2].ints[0]) + 1
                c_shape = input_shape[0]
                outputs[y] = (c_shape, hw_shape, hw_shape)
        elif node.op_type == "Add":
            a = node.input[0]
            b = node.input[1]
            c = node.output[0]
            if a in outputs:
                outputs[c] = outputs[a]
                continue
            if b in outputs:
                outputs[c] = outputs[b]
                continue
        else:
            break
    print(outputs)





def main():
    model_path = "../models/resnet50-v2-7.onnx"
    model = onnx.load(model_path)
    process_layers(model)


if __name__ == "__main__":
    main()
