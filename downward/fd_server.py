# downward/fd_server.py
import os
import json
import time
import shutil

# 1) Compute project‐root (one level above this file)
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))

# 2) Seed the directory with the two files if they aren't already here
for fname in ("merged_transition_systems.json", "causal_graph.json"):
    src = os.path.join(ROOT, fname)
    dst = os.path.join(os.getcwd(), fname)
    if os.path.exists(src) and not os.path.exists(dst):
        shutil.copy(src, dst)

i = 0
while True:
    in_path  = os.path.join("gnn_output", f"merge_{i}.json")
    out_path = os.path.join("fd_output", f"ts_{i}.json")
    # wait for the GNN’s file
    while not os.path.exists(in_path):
        time.sleep(0.1)
    # echo it
    with open(in_path) as f:
        data = json.load(f)
    with open(out_path, "w") as g:
        json.dump(data, g)
    i += 1
