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
images_hp = lambda: os.path.join(images(), "Hyperparameter tuning")
images_hp_v2 = lambda: os.path.join(images(), "hyperparameter_tuning_v2")
images_hp_v3 = lambda: os.path.join(images(), "hyperparameter_tuning_v3")
images_hp_v4 = lambda: os.path.join(images(), "hyperparameter_tuning_v4")
images_hp_v5 = lambda: os.path.join(images(), "hyperparameter_tuning_v5")

data_gen = lambda: os.path.join(storage(), "generated_datasets")
data_comparison = lambda: os.path.join(storage(), "datasets_comparison")
data_comparison_gen = lambda: os.path.join(storage(), "comparison_generated_datasets")
hyperparams_tuning = lambda: os.path.join(storage(), "hyperparams_tuning")
hp_tuning_v2 = lambda: os.path.join(storage(), "hyperparams_tuning_v2")
hp_tuning_v3 = lambda: os.path.join(storage(), "hyperparams_tuning_v3")
hp_tuning_v4 = lambda: os.path.join(storage(), "hyperparams_tuning_v4")
hp_tuning_v5 = lambda: os.path.join(storage(), "hyperparams_tuning_v5")
timers = lambda: os.path.join(storage(), "timers")
models = lambda: os.path.join(storage(), "models")
py_objects = lambda: os.path.join(storage(), "python_objects")
cf = lambda: os.path.join(storage(), "counterfactuals")
data_temp = lambda: os.path.join(storage(), "data_temp")
n_epochs_comparison = lambda: os.path.join(storage(), "n_epochs_comparison")

for dir in [data, images, images_hp, images_hp_v2, data_gen, hyperparams_tuning, hp_tuning_v2, timers, models,
            py_objects, cf, data_temp, n_epochs_comparison, hp_tuning_v3, images_hp_v3,
            hp_tuning_v4, images_hp_v4, data_comparison, data_comparison_gen, hp_tuning_v5, images_hp_v5]:
    os.makedirs(dir(), exist_ok=True)
