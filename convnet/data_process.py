import sys
import os
import json
import copy

def fancy_print(route):
  for waypoint in route:
    print(waypoint)

def get_kernel_name(full_name):
    full_name = full_name.split('(')[0]
    return full_name.split("::")[-1]

def collect_data(filename):
    total_cycle = 0
    fops = 0
    total_ops = 0
    with open(filename,) as f:
        for num, lines in enumerate(f, 1):
            if lines[:6] == "kernel" or lines[0] in ["4", "5", "6"] and num < 7:
                seg = lines.split()
                total_cycle += int(seg[6])
            elif lines[:7] == "instr_f" and lines[7:9] != 'ct' and lines[7:9] != 'mv' and lines[7:9] != 'sg':
                seg = lines.split()
                count = int(seg[1])
                weight = 2 if seg[0] == "instr_fmadd" or seg[0] == "instr_fmsub" else 1
                total_ops += count * weight
                fops += count * weight
            elif lines[:11] == "instr_total":
                seg = lines.split()
                total_ops = int(seg[1])

    return total_cycle, total_ops, fops

class KernelData:
    def __init__(self, cycle, ops, fops):
        self.cycle = cycle
        self.ops = ops
        self.fops = fops

def add_data(dt, name, cycle, ops, fops):
    if name not in dt:
        dt[name] = KernelData(cycle, ops, fops)
    else:
        dt[name].cycle += cycle
        dt[name].ops += ops
        dt[name].fops += fops

with open("training_kernel.json",) as f:
    route = json.load(f)

print("total number of training jobs: " + str(len(route)))

kernel_info_tr = dict()
convnet_cycle_tr = 0
convnet_ops_tr = 0
convnet_fops_tr = 0
layer_info_tr = dict()

for i in range(len(route)):
    cmd = copy.deepcopy(route)
    name = "test_training_kernel_%03d_hb/stats/manycore_stats.log" % i
    if not os.path.exists(name):
        continue
    cycle, ops, fops = collect_data(name)
    convnet_cycle_tr += cycle
    convnet_ops_tr += ops
    convnet_fops_tr += fops
    kernel_name = "{}-Tr".format(get_kernel_name(cmd[i]["signature"]))
    add_data(kernel_info_tr, kernel_name, cycle, ops, fops)

    if i == 0:
        add_data(layer_info_tr, "Conv-Tr", cycle, ops, fops)
    elif i in [1, 3, 7, 11]:
        add_data(layer_info_tr, "BatchNorm-Tr", cycle, ops, fops)
    elif i in [2, 4, 8 , 12]:
        add_data(layer_info_tr, "ReLU-Tr", cycle, ops, fops)
    elif i in [6, 10, 14]:
        add_data(layer_info_tr, "MaxPooling-Tr", cycle, ops, fops)
    elif i in [5, 9, 13]:
        add_data(layer_info_tr, "Merge_Residual-Tr", cycle, ops, fops)
    elif i in range(15, 18):
        add_data(layer_info_tr, "Linear-Tr", cycle, ops, fops)
    elif i == 18:
        add_data(layer_info_tr, "Scale-Tr", cycle, ops, fops)
    elif i in range(19, 23):
        add_data(layer_info_tr, "Softmax-Tr", cycle, ops, fops)
    elif i in range(23, 26):
        add_data(layer_info_tr, "Softmax_back-Tr", cycle, ops, fops)
    elif i in range(26, 28):
        add_data(layer_info_tr, "Scale_back-Tr", cycle, ops, fops)
    elif i in range(28, 34):
        add_data(layer_info_tr, "Linear_back-Tr", cycle, ops, fops)
    elif i in [34, 38, 42]:
        add_data(layer_info_tr, "MaxPooling_back-Tr", cycle, ops, fops)
    elif i in [35, 39, 43, 46]:
        add_data(layer_info_tr, "ReLU_back-Tr", cycle, ops, fops)
    elif i in [37, 41, 45]:
        add_data(layer_info_tr, "Merge_Residual_back-Tr", cycle, ops, fops)
    elif i in [36, 40, 44, 47]:
        add_data(layer_info_tr, "BatchNorm_back-Tr", cycle, ops, fops)
    else:
        add_data(layer_info_tr, "Conv_back-Tr", cycle, ops, fops)


with open("inference_kernel.json",) as f:
    route2 = json.load(f)

print("total number of inference jobs: " + str(len(route2)))

kernel_info_pr = dict()
convnet_cycle_pr = 0
convnet_ops_pr = 0
convnet_fops_pr = 0
layer_info_pr = dict()

for i in range(len(route2)):
    cmd = copy.deepcopy(route2)
    name = "test_inference_kernel_%03d_hb/stats/manycore_stats.log" % i
    if not os.path.exists(name):
        continue
    cycle, ops, fops = collect_data(name)
    convnet_cycle_pr += cycle
    convnet_ops_pr += ops
    convnet_fops_pr += fops
    kernel_name = "{}-Pr".format(get_kernel_name(cmd[i]["signature"]))
    add_data(kernel_info_pr, kernel_name, cycle, ops, fops)

    if i in [0, 2, 6, 10]:
        add_data(layer_info_pr, "BatchNorm-Pr", cycle, ops, fops)
    elif i in [1, 3, 7, 11]:
        add_data(layer_info_pr, "ReLU-Pr", cycle, ops, fops)
    elif i in [4, 8, 12]:
        add_data(layer_info_pr, "Merge_Residual-Pr", cycle, ops, fops)
    elif i in [5, 9, 13]:
        add_data(layer_info_pr, "MaxPooing-Pr", cycle, ops, fops)
    elif i in range(14, 17):
        add_data(layer_info_pr, "Linear-Pr", cycle, ops, fops)
    else:
        add_data(layer_info_pr, "Scale-Pr", cycle, ops, fops)



conv_layer = ["0", "1", "1s", "2", "2s", "3", "3s"]

for i in conv_layer:
    name = "test_inference_kernel_conv"+i+"_hb/stats/manycore_stats.log"
    cycle, ops, fops = collect_data(name)
    convnet_cycle_pr += cycle
    convnet_ops_pr += ops
    convnet_fops_pr += fops
    add_data(kernel_info_pr, "conv2d", cycle, ops, fops)
    add_data(kernel_info_tr, "conv2d", cycle, ops, fops)
    add_data(layer_info_pr, "Conv-Pr", cycle, ops, fops)
    add_data(layer_info_tr, "Conv-Tr", cycle, ops, fops)


for i in conv_layer:
    name = "test_training_kernel_conv"+i+"_hb/stats/manycore_stats.log"
    kernel_name = "conv2d_backward"
    cycle, ops, fops = collect_data(name)
    convnet_cycle_tr += cycle
    convnet_ops_tr += ops
    convnet_fops_tr += fops

    cycle = cycle if i != "0" else cycle - kernel_info_pr["conv2d"].cycle
    ops = ops if i != "0" else ops - kernel_info_pr["conv2d"].ops
    fops = fops if i != "0" else fops - kernel_info_pr["conv2d"].fops

    add_data(kernel_info_tr, "conv2d_backward", cycle, ops, fops)
    add_data(layer_info_tr, "Conv_back-Tr", cycle, ops, fops)


freq = 1e9 # Hz
pod_num = 1

convnet_cycle = convnet_cycle_tr + convnet_cycle_pr
convnet_ops = convnet_ops_tr + convnet_ops_pr
convnet_fops = convnet_fops_tr + convnet_fops_pr

print("Total Cycle of ConvNet = {}\n".format(convnet_cycle))
print("Training Cycle: {}, Inference Cycle: {}\n".format(convnet_cycle_tr, convnet_cycle_pr))
print("Total time of one Iter (1GHz) = {}s\n".format(float(convnet_cycle)/freq))
print("Total time of one Iter (2GHz) = {}s\n".format(float(convnet_cycle)/freq/2))
print("Total GOPs of ConvNet  = {}\n".format(float(convnet_ops)/1e9))
print("Total GFOP of ConvNet  = {}\n".format(float(convnet_fops)/1e9))
print("Total GOPs of ConvNet (1GHz) = {}\n".format(float(convnet_ops)/(float(convnet_cycle)/freq)*pod_num / 1e9))
print("Total GOPs of ConvNet (2GHz) = {}\n".format(float(convnet_ops)/(float(convnet_cycle)/freq/2)*pod_num / 1e9))
print("Total GFlops of ConvNet (1GHz) = {}\n".format(float(convnet_fops)/(float(convnet_cycle)/freq)*pod_num / 1e9))
print("Total GFlops of ConvNet (2GHz) = {}\n".format(float(convnet_fops)/(float(convnet_cycle)/freq/2)*pod_num / 1e9))

print("===============================Training Kernel Info=====================================\n")
print("Kernel\t\tCycles\t\tFlop Ratio\t\tFOPS\n")
for kernel in kernel_info_tr.keys():
    print("{}\t\t{}\t\t{}\t\t{}\n".format(kernel, kernel_info_tr[kernel].cycle, float(kernel_info_tr[kernel].fops)/kernel_info_tr[kernel].ops, kernel_info_tr[kernel].fops))
print("===========================================================\n")


print("===============================Inference Kernel Info=====================================\n")
print("Kernel\t\tCycles\t\tFlop Ratio\t\tFOPS\n")
for kernel in kernel_info_pr.keys():
    print("{}\t\t{}\t\t{}\t\t{}\n".format(kernel, kernel_info_pr[kernel].cycle, float(kernel_info_pr[kernel].fops)/kernel_info_pr[kernel].ops, kernel_info_pr[kernel].fops))
print("===========================================================\n")

print("===============================Training Layer Info=====================================\n")
print("Layer\t\tCycles\t\tFlop Ratio\t\tFOPS\n")
for layer in layer_info_tr.keys():
    print("{}\t\t{}\t\t{}\t\t{}\n".format(layer, layer_info_tr[layer].cycle, float(layer_info_tr[layer].fops)/layer_info_tr[layer].ops, layer_info_tr[layer].fops))
print("===========================================================\n")

print("===============================Inference Layer Info=====================================\n")
print("Layer\t\tCycles\t\tFlop Ratio\t\tFOPS\n")
for layer in layer_info_pr.keys():
    print("{}\t\t{}\t\t{}\t\t{}\n".format(layer, layer_info_pr[layer].cycle, float(layer_info_pr[layer].fops)/layer_info_pr[layer].ops, layer_info_pr[layer].fops))
print("===========================================================\n")

