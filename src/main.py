import onnx
from onnx import helper
from google.protobuf.json_format import MessageToDict
import math
import sram
from power_info import *
import re
import numpy as np
import csv

# cache_cap = [2048, 1048576, 2097152, 3145728, 4194304]
# sram_read = [0.00757572, 0.114846, 0.179646, 0.218159, 0.253573]
# sram_write = [0.00849597, 0.121253, 0.185333, 0.22447, 0.261366]

def get_model_summary(onnx_model):
    summary = onnx.helper.printable_graph(onnx_model.graph)
    return summary

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
        # print(node)
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
            if b in outputs: #never actually called
                outputs[c] = outputs[b]
                continue
        else:
            break
    return outputs


def run_sim(config, model, outputs, weight_shapes):
    
    state = sram.SramState(sram_size[config], 0.5)
    tot_sram_ld_kernel = 0
    tot_sram_ld_feat = 0
    tot_sram_st_feat = 0
    tot_dram_ld_kernel = 0
    tot_dram_ld_feat = 0
    tot_dram_st_feat = 0
    for node in model.graph.node:
        if node.op_type == "Conv":
            pattern = r"resnetv24_.+_conv3_fwd"
            if (re.match(pattern, node.name)):
                continue
            in_shape = outputs[node.input[0]]
            out_shape = outputs[node.output[0]]
            conv_shape = weight_shapes[node.input[1]]
            conv_stride = node.attribute[4].ints[0]
            sram_ld_kernel, sram_ld_feat, sram_st_feat, dram_ld_kernel, dram_ld_feat, dram_st_feat = state.operate_conv(in_shape, out_shape, conv_shape, conv_stride)
            tot_sram_ld_kernel += sram_ld_kernel
            tot_sram_ld_feat += sram_ld_feat
            tot_sram_st_feat += sram_st_feat
            tot_dram_ld_kernel += dram_ld_kernel
            tot_dram_ld_feat += dram_ld_feat
            tot_dram_st_feat += dram_st_feat
    tot_dram_ld = tot_dram_ld_kernel + tot_dram_ld_feat
    tot_dram_st = tot_sram_st_feat
    tot_sram_ld = tot_sram_ld_kernel + tot_sram_ld_feat + tot_dram_st
    tot_sram_st = tot_sram_st_feat + tot_dram_ld

    tot_energy = tot_sram_ld * sram_read[config] + tot_sram_st * sram_write[config] + tot_dram_ld + tot_dram_st
    tot_dram_acc = tot_dram_ld + tot_dram_st
    return (tot_energy, tot_dram_acc)
def csv_print(file_name, value1, value2, value3):
    # Open the file in append mode
    with open(file_name, mode='a', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(["byte", "nJ", "dram_acc"])
        # Write the two values as one row in the CSV
        writer.writerow([value1, value2, value3])

def main():
    model_path = "../models/resnet50-v2-7.onnx"
    model = onnx.load(model_path)
    outputs = process_layers(model)
    weight_shapes = get_input_size(model)
    energy = []
    # print(len(sram_size))
    with open("out.csv", mode="w", newline='') as file:
        writer = csv.writer(file)
        writer.writerow(["byte", "nJ", "dram_acc"])
        for config in range (0, len(sram_size)):
            en, tot_dram_acc = run_sim(config, model, outputs, weight_shapes)
            print(en)
            writer.writerow([sram_size[config], en, tot_dram_acc])
            energy.append(en)
        print(np.array(energy)/1000)


if __name__ == "__main__":
    main()
