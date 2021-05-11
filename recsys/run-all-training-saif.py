import sys
import os
import json
import copy
import subprocess

qsub_template = '(cd {path}; qsub -N {job_name} -l walltime=72:00:00 -l nodes=1:ppn=6 -l mem="30gb" -d {path} -o {path}/qsub.out -e {path}/qsub.err -V run.sh)\n'

def fancy_print(route):
  for waypoint in route:
    print(waypoint)

with open('training.json',) as f:
  route = json.load(f)

fancy_print(route)
print()

print("total number of jobs: " + str(len(route)))

for i in range(46,len(route)):
  # generate one-hot json
  cmd = copy.deepcopy(route)
  cmd[i]['offload'] = True
  fancy_print(cmd)
  print()

  # create kernel folder
  name = "Recsys_training_kernel_%03d_hb-saifgen" % i
  sh_cmd = "mkdir -p " + name
  print(sh_cmd)
  os.system(sh_cmd)

  # write json file
  with open(name + "/training.json", 'w') as outfile:
    json.dump(cmd, outfile, indent=4, sort_keys=True)

  # link recsys input dataset
  sh_cmd = "ln -s /work/global/lc873/work/sdh/phase2-eval/recsys/hb-pytorch-apps/recsys/data " + name + "/data"
  print(sh_cmd)
  os.system(sh_cmd)

  # create run script
  script = "(source /work/global/lc873/work/sdh/venv_cosim/bin/activate; source /work/global/lc873/work/sdh/playground/bigblade-03-31/bsg_bladerunner/setup.sh; pycosim.saif /work/global/lc873/work/sdh/phase2-eval/recsys/hb-pytorch-apps/recsys/recsys.py --nbatch 1 --training > out.std 2>&1; gzip run.saif;)\n"
  with open(name + "/run.sh", 'w') as outfile:
    outfile.write(script)

  # get current path
  path = str(os.path.abspath(os.getcwd())) + "/" + name
  print(path)

  # generate qsub script
  qsub_starter = qsub_template.format(job_name=name, path=path)
  print(qsub_starter)
  with open(name + "/qsub.sh", 'w') as outfile:
    outfile.write(qsub_starter)

  print("starting cosim job ...")
  cosim_run = subprocess.Popen(["sh", name + "/qsub.sh"], env=os.environ)
