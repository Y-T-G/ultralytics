# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

"""Minimal reproduction for the model build failure.

Run: python bughunt_repro.py
"""

from ultralytics import YOLO

if __name__ == "__main__":
    model = YOLO("bughunt_model.yaml")
    print(model.predict("https://ultralytics.com/images/bus.jpg"))
