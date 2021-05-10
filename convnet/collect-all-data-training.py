import sys
import os
import json
import copy
import subprocess

sys.path.append('/mnt/users/ssd2/homes/xingyaoz/research/hb-pytorch-cosim-v6/hammerblade/scripts')
from compare_aten_op import compare, average_aten_op

# ======================================================
# Helper
# ======================================================

def fancy_print(route):
  for waypoint in route:
    print(waypoint)

def compare_wrapper(full, chunk, stats):
  assert len(full) == len(chunk)
  ops = []
  for i in range(len(full)):
    ops.append(compare(full[i], chunk[i], stats))
  aten_op = average_aten_op(ops)
  return aten_op

# ======================================================
# Read kernels
# ======================================================
# '''
with open('cmd.json',) as f:
  route = json.load(f)

fancy_print(route)
print()

print("total number of jobs: " + str(len(route)))
# '''
# ======================================================
# Main loop
# ======================================================

kernels = []

conv_layer = ["0", "1", "1s", "2", "2s", "3", "3s"]
l = [40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 56, 58, 59, 63, 70, 82, 101, 107, 110, 120, 132, 140, 142, 143, 145, 147, 148, 149, 153, 156, 157, 159, 175, 176, 184, 194]

# for i in range(len(route)):
# for i in conv_layer:
for i in l:
  hb_name = "test_kernel_%03d_hb" % i
  # hb_name = "test_kernel_convF" + i + "_hb"
  # name = "test_kernel_%03d_data_new_format" % i
  # work dir
  # sh_cmd = "mkdir -p " + name
  # print(sh_cmd)
  # os.system(sh_cmd)
  '''
  for j in range(10):
    xeon_name = "test_kernel_%03d_xeon_%02d" % (i, j)
    # collect full stack
    sh_cmd = "cp " + xeon_name + "/out.std " + name + ("/full-%02d.std" % j)
    print(sh_cmd)
    os.system(sh_cmd)
  # collect cosim chunk stack for cross check
  sh_cmd = "cp " + hb_name + "/out.std " + name + "/chunk-cosim.std"
  print(sh_cmd)
  os.system(sh_cmd)
  '''
  # run uw's profiling data tool
  sh_cmd = "(cd " + hb_name + ";python /mnt/users/ssd2/homes/xingyaoz/research/bsg_bladerunner_vcs_v6_new/bsg_manycore/software/py/vanilla_parser/stats_parser.py --stats vanilla_stats.csv)"
  print(sh_cmd)
  os.system(sh_cmd)
  '''
  # collect output
  sh_cmd = "cp " + hb_name + "/stats/manycore_stats.log " + name + "/"
  print(sh_cmd)
  os.system(sh_cmd)

  # ======================================================
  # Actual data processing
  # ======================================================
  full = ["{0}/full-{1:02d}.std".format(name, j) for j in range(10)]
  chunk = ["{0}/full-{1:02d}.std".format(name, j) for j in range(10)]
  stats = "{0}/manycore_stats.log".format(name)
  kernels.append(compare_wrapper(full, chunk, stats))
  print(kernels[-1].draw_cpu_log())
  print()
  print(kernels[-1])
  # ======================================================
  # End of actual data procesing
  # ======================================================

  # run postprocessing script for cross check
  sh_cmd = "python ~/work/sdh/brg-hb-pytorch/hammerblade/scripts/compare_aten_op.py --full {0}/full-00.std --chunk {0}/chunk-cosim.std --manycore-stats {0}/manycore_stats.log > {0}/cross_check.txt 2>&1".format(name)
  print(sh_cmd)
  os.system(sh_cmd)
  '''

'''
# ======================================================
# Print all kernels at the end
# ======================================================
total_xeon = 0
total_host = 0
total_device = 0

buf = """
+-----------------------+---------------+--------------------+--------------------+-----------------+---------------------+-----------------+-------------------+
|        ATen OP        |     Input     |     Full  Size     |     Chunk Size     |    Xeon Time    |    HB Total Time    |    Host Time    |    Device Time    |
+-----------------------+---------------+--------------------+--------------------+-----------------+---------------------+-----------------+-------------------+"""

for k in kernels:
  total_xeon += k.xeon_time
  total_host += k.hb_host_time
  total_device += k.hb_device_time
  buf += k.fancy_print()

template = "\n| {func:<22}| {tensor:<14}| {full:<19}| {chunk:<19}|{xeon:>16} |{hb:>20} |{host:>16} |{device:>18} |"

buf += template.format(
                func = "Total",
                tensor = "",
                full = "",
                chunk = "",
                xeon = "{:.2f}".format(total_xeon / 1000.0),
                hb = "{:.2f}".format((total_host + total_device) / 1000.0),
                host = "{:.2f}".format(total_host / 1000.0),
                device = "{:.2f}".format(total_device / 1000.0))
buf += """
+-----------------------+---------------+--------------------+--------------------+-----------------+---------------------+-----------------+-------------------+"""
print(buf)
'''