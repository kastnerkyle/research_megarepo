#!/usr/bin/env python                                                                                                                                                                     
# gpu_stat.py [DELAY [COUNT]]                                                                                                                                                             
# dump gpu stats as a line of json                                                                                                                                                        
# {"time": 1474168378.146957, "pci_tx": 146000, "pci_rx": 1508000,                                                                                                                        
#     "gpu_util": 42, "mem_util": 24, "mem_used": 11710,                                                                                                                                  
#     "temp": 76, "fan_speed": 44 }                                                                                                                                                       

from collections import OrderedDict
import json
import subprocess
import sys
import time
import xml.etree.ElementTree

try:
  delay = float(sys.argv[1])
except:
  delay = 1

try:
  count = float(sys.argv[2])
except:
  count = None

def extract(elem, tag, drop_s):
  text = elem.find(tag).text
  if drop_s not in text: raise Exception(text)
  return int(text.replace(drop_s, ""))

i = 0
while True:
  d = OrderedDict()
  d["time"] = time.time()

  cmd = ['nvidia-smi', '-q', '-x']
  cmd_out = subprocess.check_output(cmd)
  gpu = xml.etree.ElementTree.fromstring(cmd_out).find("gpu")

  pci = gpu.find("pci")
  d["pci_tx"] = extract(pci, "tx_util", "KB/s")
  d["pci_rx"] = extract(pci, "rx_util", "KB/s")

  util = gpu.find("utilization")
  d["gpu_util"] = extract(util, "gpu_util", "%")
  d["mem_util"] = extract(util, "memory_util", "%")

  d["mem_used"] = extract(gpu.find("fb_memory_usage"), "used", "MiB")

  d["temp"] = extract(gpu.find("temperature"), "gpu_temp", "C")
  d["fan_speed"] = extract(gpu, "fan_speed", "%")

  print json.dumps(d)
  sys.stdout.flush()

  if count != None:
    i += 1
    if i == count:
      exit(0)
  time.sleep(delay)
