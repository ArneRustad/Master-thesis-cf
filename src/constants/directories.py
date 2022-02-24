import os
import sys
from .set_global_dirs import project_dir as project
from .set_global_dirs import storage_dir as storage

if project is None:
    project = lambda: os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

if storage is None:
    if sys.platform == "win32":
        storage = lambda: "S:/arneir/Master-thesis-storage/"
    elif sys.platform == "linux":
        storage = lambda: "/work/arneir/Master-thesis-storage/"
    else:
        raise RuntimeError("Did not recognize the system platform and therefore does not know how to initialize the storage directory path.")

data = lambda: os.path.join(project(), "data")
images = lambda: os.path.join(project(), "images")

data_gen = lambda: os.path.join(storage(), "generated_datasets")
hyperparams_tuning = lambda: os.path.join(storage(), "hyperparams_tuning")
timers = lambda: os.path.join(storage(), "timers")
models = lambda: os.path.join(storage(), "models")
py_objects = lambda: os.path.join(storage(), "python_objects")
cf = lambda: os.path.join(storage(), "counterfactuals")