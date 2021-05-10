import sys
import os
import json
import copy
import subprocess

# batch_size, channels, filters, image_dim, kernel_size
conv2d_config = []
conv2d_config.append([4, 3, 16, 32, 3])
conv2d_config.append([4, 16, 16, 32, 3])
conv2d_config.append([4, 32, 64, 16, 3])
conv2d_config.append([4, 64, 128, 8, 3])


conv2d_residual_config = []
conv2d_residual_config.append([4, 16, 16, 32, 1])
conv2d_residual_config.append([4, 32, 64, 16, 1])
conv2d_residual_config.append([4, 64, 128, 8, 1])

# conv2d training
for i in range(len(conv2d_config)):
  name = "test_training_kernel_conv%d_hb" % i
  sh_cmd = "mkdir -p " + name
  print(sh_cmd)
  os.system(sh_cmd)
  sh_cmd = "ln -s ../conv2d.py " + name + "/conv2d.py"
  print(sh_cmd)
  os.system(sh_cmd)
  script = "(cd " + name + "; pycosim.profile conv2d.py --batch-size {} --channels {} --filters {} --image-dim {} --kernel-size {} --padding > out.std 2>&1)".format(x for x in conv2d_config[i])
  with open(name + "/run.sh", 'w') as outfile:
    outfile.write(script)
  print("starting cosim job ...")
  cosim_run = subprocess.Popen(["sh", name + "/run.sh"], env=os.environ)

for i in range(len(conv2d_residual_config)):
  name = "test_training_kernel_conv%ds_hb" % i+1
  sh_cmd = "mkdir -p " + name
  print(sh_cmd)
  os.system(sh_cmd)
  sh_cmd = "ln -s ../conv2d.py " + name + "/conv2d.py"
  print(sh_cmd)
  os.system(sh_cmd)
  script = "(cd " + name + "; pycosim.profile conv2d.py --batch-size {} --channels {} --filters {} --image-dim {} --kernel-size {} > out.std 2>&1)".format(x for x in conv2d_residual_config[i])
  with open(name + "/run.sh", 'w') as outfile:
    outfile.write(script)
  print("starting cosim job ...")
  cosim_run = subprocess.Popen(["sh", name + "/run.sh"], env=os.environ)

# conv2d forward
for i in range(len(conv2d_config)):
  name = "test_inference_kernel_conv%d_hb" % i
  sh_cmd = "mkdir -p " + name
  print(sh_cmd)
  os.system(sh_cmd)
  sh_cmd = "ln -s ../conv2d_inference.py " + name + "/conv2d_inference.py"
  print(sh_cmd)
  os.system(sh_cmd)
  script = "(cd " + name + "; pycosim.profile conv2d_inference.py --batch-size {} --channels {} --filters {} --image-dim {} --kernel-size {} --padding > out.std 2>&1)".format(x for x in conv2d_config[i])
  with open(name + "/run.sh", 'w') as outfile:
    outfile.write(script)
  print("starting cosim job ...")
  cosim_run = subprocess.Popen(["sh", name + "/run.sh"], env=os.environ)

for i in range(len(conv2d_residual_config)):
  name = "test_inference_kernel_conv%ds_hb" % i+1
  sh_cmd = "mkdir -p " + name
  print(sh_cmd)
  os.system(sh_cmd)
  sh_cmd = "ln -s ../conv2d_inference.py " + name + "/conv2d_inference.py"
  print(sh_cmd)
  os.system(sh_cmd)
  script = "(cd " + name + "; pycosim.profile conv2d_inference.py --batch-size {} --channels {} --filters {} --image-dim {} --kernel-size {} > out.std 2>&1)".format(x for x in conv2d_residual_config[i])
  with open(name + "/run.sh", 'w') as outfile:
    outfile.write(script)
  print("starting cosim job ...")
  cosim_run = subprocess.Popen(["sh", name + "/run.sh"], env=os.environ)

