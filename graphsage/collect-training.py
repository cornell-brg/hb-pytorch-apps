import sys
import os
import json
import subprocess

with open('training.json',) as f:
  route = json.load(f)

device_dict = {}

#for i in [13, 37, 40, 42, 43, 45, 46, 48, 49, 51, 52, 54, 55, 57, 58]:
#for i in [13]:
#for i in range(0, 19):
for i in range(len(route)):
  aten = route[i]['signature']
  print(aten)
  name = "graphsage_training_%d" % i
  data = subprocess.check_output(['grep', 'finished --', name+'/out.std'])
  data = data.decode("utf-8").splitlines()
  print(data)
  device_time = 0
  for d in data[1:]:
    device_time += int(d.split("=")[1])
  print(device_time)
  func = aten
  func = func.split("(")[0]
  func = func.split("::")[-1]
  func = "aten::" + func
  print(func)
  if func in device_dict:
    device_dict[func] += device_time
  else:
    device_dict[func] = device_time

# data dump
total = 0
print()
print("----------------------------------------------------------------------- (ns)")
for func in device_dict:
  # print("{func:50}     {time:.2f}".format(func=func, time=(device_dict[func] / 1000000.0))) # ms
  # total += (device_dict[func] / 1000000.0) # ms
  print("{func:50}     {time:.2f}".format(func=func, time=(device_dict[func] / 1))) # ns
  total += (device_dict[func] / 1) # ns
print()
print("{func:50}     {time:.2f}".format(func="total", time=total))


