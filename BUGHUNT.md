# Bug hunt

Building a YOLO model from its YAML config crashes.

## Reproduce

```bash
pip install -e .
python bughunt_repro.py
```

Expected: the script prints prediction results for one image.
Actual: it raises an `AssertionError` while the model is being built.

## Task

Find the root cause and fix it. Talk through what you check and why.

You can edit anything, add prints or breakpoints, and run any command.
