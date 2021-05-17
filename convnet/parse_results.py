import sys
import os
import json
import copy
import subprocess
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--bladerunner-dir', type=str)
args = parser.parse_args()
BLADERUNNER_DIR = args.bladerunner_dir

# ======================================================
# Read training kernels
# ======================================================

with open('training_kernel.json',) as f:
  route1 = json.load(f)


print("total number of jobs: " + str(len(route1)))

# ======================================================
# Parse training kernels
# ======================================================

for i in range(len(route1)):
  hb_name = "test_training_kernel_%03d_hb" % i
  # run uw's profiling data tool
  sh_cmd = "(cd " + hb_name + ";python {}/bsg_manycore/software/py/vanilla_parser/stats_parser.py --stats vanilla_stats.csv)".format(args.bladerunner_dir)
  print(sh_cmd)
  os.system(sh_cmd)

# ======================================================
# Read inference kernels
# ======================================================

with open('inference_kernel.json',) as f:
  route2 = json.load(f)

print("total number of jobs: " + str(len(route2)))

# ======================================================
# Parse inference kernels
# ======================================================

for i in range(len(route2)):
  hb_name = "test_inference_kernel_%03d_hb" % i
  # run uw's profiling data tool
  sh_cmd = "(cd " + hb_name + ";python {}/bsg_manycore/software/py/vanilla_parser/stats_parser.py --stats vanilla_stats.csv)".format(args.bladerunner_dir)
  print(sh_cmd)
  os.system(sh_cmd)

# ======================================================
# Parse conv2d training kernels
# ======================================================

conv_layer = ["0", "1", "1s", "2", "2s", "3", "3s"]
for i in conv_layer:
  hb_name = "test_training_kernel_conv{}_hb".format(i)
  # run uw's profiling data tool
  sh_cmd = "(cd " + hb_name + ";python {}/bsg_manycore/software/py/vanilla_parser/stats_parser.py --stats vanilla_stats.csv)".format(args.bladerunner_dir)
  print(sh_cmd)
  os.system(sh_cmd)

# ======================================================
# Parse conv2d inference kernels
# ======================================================

for i in conv_layer:
  hb_name = "test_inference_kernel_conv{}_hb".format(i)
  # run uw's profiling data tool
  sh_cmd = "(cd " + hb_name + ";python {}/bsg_manycore/software/py/vanilla_parser/stats_parser.py --stats vanilla_stats.csv)".format(args.bladerunner_dir)
  print(sh_cmd)
  os.system(sh_cmd)

