import os
import sys

project = lambda: os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if sys.platform == "win32":
    storage = lambda: "S:/work/arneir/Master-thesis-storage/"
else:
    storage = lambda: "/work/arneir/Master-thesis-storage/"

data = lambda: os.path.join(project(), "data")
images = lambda: os.path.join(storage(), "images")

data_gen = lambda: os.path.join(storage(), "generated_datasets")
hyperparams_tuning = lambda: os.path.join(storage(), "hyperparams_tuning")
timers = lambda: os.path.join(storage(), "timers")
models = lambda: os.path.join(storage(), "models")
py_objects = lambda: os.path.join(storage(), "python_objects")
cf = lambda: os.path.join(storage(), "counterfactuals")
#hyperparams_tuning = lambda: "S:\\arneir\\Hyperparams_tuning"