import argparse
import sys
import os
import json
import subprocess
import pprint
import copy

# config
saifgen_path = "/work/global/zz546/profile_result/saif-lgc-ista-hbspmvxcel-04-01/"
pt_path      = "/work/global/lc873/work/sdh/handoff/lgc-ista-xcel/"
script       = "/work/global/zz546/profile_result/sdh-phase2-eval-power-scripts/parse-all-pt.py"

# default
default = { 'FP ALU': 0,
  'HBM': 0.0,
  'L2 cache': 0,
  'est clk': 0,
  'icache': 0,
  'integer ALU': 0,
  'network': 0,
  'other': 0,
  'register file': 0,
  'scratchpad': 0}
default = json.loads(str(default).replace("'","\""))

with open('lgc_ista.json',) as f:
  route = json.load(f)

parser = argparse.ArgumentParser()
parser.add_argument('--id', required=True, nargs='+', type=int,
                    help="training kernel instance id")
args = parser.parse_args()

def collect_power(i):
  aten = route[i]['signature']
  print(aten)
  name = "lgc_ista_hb_saif_%d" % i
  func = aten
  func = func.split("(")[0]
  func = func.split("::")[-1]
  func = "aten::" + func
  print(func)

  try:
    subprocess.check_output("ls " + name + "/run.saif.gz", shell=True)
    data = subprocess.check_output("gzip -cd " + name + "/run.saif.gz | head -n 20", shell=True)
    data = data.decode("utf-8").splitlines()
    print(data)
    device_time = 0
    for d in data:
      if d.startswith("(DURATION"):
        device_time = float((d.split()[-1])[:-2])
        break
  except:
    print("this kernel does not have device power")
    return func, copy.deepcopy(default)

  print(device_time)
  cmd = "(cd {0}; python {1} -d {2};)".format(pt_path+name, script, int(device_time))
  print(cmd)
  data = subprocess.check_output(cmd, shell=True).decode('utf-8')
  print(data)
  print(func)

  data = json.loads(data.replace("'","\""))
  print(data)

  # add HBM controller 1pj/bit energy
  cmd = "(cd {0}; python /work/global/zz546/bsg_replicant/examples/cuda/test_profiler/dramsim3.py;)".format(saifgen_path + name)
  print(cmd)
  hbm = subprocess.check_output(cmd, shell=True).decode('utf-8')
  print(hbm)
  bits = hbm.split()[-1]
  print(bits)

  data["HBM"] = float(bits)

  return func,data

total_energy = {}

for i in args.id:
  func, data = collect_power(i)

  if func not in total_energy:
    total_energy[func] = copy.deepcopy(default)

  old = total_energy[func]
  for key in old:
    old[key] += data[key]
    total_energy[func] = old

pp = pprint.PrettyPrinter(indent=2)
pp.pprint(total_energy)
