import argparse
import sys
import os
import json
import subprocess

# config
saifgen_path = "/work/global/lc873/work/sdh/phase2-eval/recsys-03-31-saif/"
pt_path      = "/work/global/secure/en-ec-brg-vip-gf-14nm-14lppxl-nda/hb-phase2-eval/power/recsys-03-31/"
script       = "/work/global/secure/en-ec-brg-vip-gf-14nm-14lppxl-nda/hb-phase2-eval/power/scripts/parse-all-pt.py"

with open('training.json',) as f:
  route = json.load(f)

parser = argparse.ArgumentParser()
parser.add_argument('--id', required=True, nargs='+', type=int,
                    help="training kernel instance id")
args = parser.parse_args()

def check_pt(i):
  aten = route[i]['signature']
  print(aten)
  name = "Recsys_training_kernel_%03d_hb-saifgen" % i
  try:
    cmd = "ls {0}".format(pt_path+name)
    ret = subprocess.check_output(cmd, shell=True)
  except:
    print("not an active saif task ... nothing to do")
    exit()
  try:
    cmd = "(cd {0}; python {1} -d 1;)".format(pt_path+name, script)
    print(cmd)
    ret = subprocess.check_output(cmd, shell=True)
  except:
    print("crash happended ... please fix manually")
    exit()
  try:
    cmd = "(cd {0}; head -n 20 ./vcore-0-0/inputs/run.saif > duration.txt;)".format(pt_path+name)
    print(cmd)
    ret = subprocess.check_output(cmd, shell=True)
    ret = subprocess.check_output("gzip " + saifgen_path + name + "/run.saif &", shell=True)
  except:
    print("No saif file ... should not happen")
    exit()

for i in args.id:
  check_pt(i)
