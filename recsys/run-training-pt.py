import argparse
import sys
import os
import json
import subprocess

# config
saifgen_path = "/work/global/lc873/work/sdh/phase2-eval/recsys-03-31-saif/"
pt_path      = "/work/global/secure/en-ec-brg-vip-gf-14nm-14lppxl-nda/hb-phase2-eval/power/recsys-03-31/"
script       = "/work/global/secure/en-ec-brg-vip-gf-14nm-14lppxl-nda/hb-phase2-eval/power/scripts/run-all-pt.py"

with open('training.json',) as f:
  route = json.load(f)

parser = argparse.ArgumentParser()
parser.add_argument('--id', required=True, nargs='+', type=int,
                    help="training kernel instance id")
args = parser.parse_args()

def launch_pt(i):
  aten = route[i]['signature']
  print(aten)
  name = "Recsys_training_kernel_%03d_hb-saifgen" % i
  try:
    ret = subprocess.check_output("gzip -d " + saifgen_path + name + "/run.saif.gz", shell=True)
  except:
    print("No saif file ... exit ...")
    exit()
  # create a folder at pt path
  try:
    ret = subprocess.check_output("mkdir " + pt_path + name, shell=True)
  except:
    print("Build exists ... manully remove it if you want to rerun ...")
    exit()
  # start pt
  saif = saifgen_path + name + "/run.saif"
  cmd = "(cd {0}; python {1} --saif {2};)".format(pt_path+name, script, saif)
  print(cmd)
  ret = subprocess.check_output(cmd, shell=True)

for i in args.id:
  launch_pt(i)
