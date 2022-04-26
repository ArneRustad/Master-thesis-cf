import os
import sys
import socket
from .set_global_dirs import project_dir as project
from .set_global_dirs import storage_dir as storage

if project is None:
    project = lambda: os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

hostname = socket.gethostname()
if storage is None:
    if hostname == "DESKTOP-IEDF8NC":
        storage = lambda: os.path.join("S:", "arneir", "Master-thesis-storage")
    elif hostname == "markov.math.ntnu.no":
        storage = lambda: os.path.join("/work", "arneir", "Master-thesis-storage/")
    elif hostname[:4] == "idun":
        storage = lambda: os.path.join("/cluster", "work", "arneir")
    else:
        raise RuntimeError(f"Did not recognize the system hostname {hostname} and therefore does not know how to initialize the storage directory path.")

data = lambda: os.path.join(project(), "data")
images = lambda: os.path.join(project(), "images")

data_gen = lambda: os.path.join(storage(), "generated_datasets")
hyperparams_tuning = lambda: os.path.join(storage(), "hyperparams_tuning")
timers = lambda: os.path.join(storage(), "timers")
models = lambda: os.path.join(storage(), "models")
py_objects = lambda: os.path.join(storage(), "python_objects")
cf = lambda: os.path.join(storage(), "counterfactuals")
data_temp = lambda: os.path.join(storage(), "data_temp")
n_epochs_comparison = lambda: os.path.join(storage(), "n_epochs_comparison")
