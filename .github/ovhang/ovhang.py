"""Minimal pure-OpenVINO reproducer for the AsyncInferQueue hang (ultralytics#25923)."""

import os
import sys

import numpy as np
import openvino as ov

xml, hint, iters, batch = sys.argv[1], sys.argv[2], int(sys.argv[3]), int(sys.argv[4])
queue_per_call = (sys.argv[5] if len(sys.argv) > 5 else "percall") == "percall"
core = ov.Core()
cfg = {"PERFORMANCE_HINT": hint}
if os.environ.get("NUM_STREAMS"):
    cfg["NUM_STREAMS"] = os.environ["NUM_STREAMS"]
compiled = core.compile_model(core.read_model(xml), "CPU", cfg)
name = compiled.inputs[0].get_any_name()
im = np.random.rand(batch, 3, 640, 640).astype(np.float32)
print(f"streams={compiled.get_property('NUM_STREAMS')} threads={compiled.get_property('INFERENCE_NUM_THREADS')} percall={queue_per_call}", flush=True)

queue = None
for it in range(iters):
    if queue_per_call or queue is None:  # ultralytics builds a new queue on every forward()
        queue = ov.AsyncInferQueue(compiled)
        queue.set_callback(lambda request, userdata: None)
    for i in range(batch):
        queue.start_async(inputs={name: im[i : i + 1]}, userdata=i)
    queue.wait_all()
    print(f"iter {it}", flush=True)
print("DONE", flush=True)
