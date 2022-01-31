import os
dir_project = lambda: os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
dir_datasets = lambda: os.path.join(dir_project(), "Datasets")
dir_timers = lambda: os.path.join(dir_project(), "Timers")
dir_models = lambda: os.path.join(dir_project(), "Models")
dir_images = lambda: os.path.join(dir_project(), "Images")
dir_py_objects = lambda: os.path.join(dir_project(), "Python_objects")
dir_cf = lambda: os.path.join(dir_project(), "Counterfactuals")
dir_hyperparams_tuning = lambda: "S:\\arneir\\Hyperparams_tuning"