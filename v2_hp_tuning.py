print("Starting hyperparameter tuning on Idun")
import os
import helpers.hp_tuning.hp_gen
from tabGAN import TabGAN
from src import constants as const
import numpy as np
import pandas as pd

PROGRESS_BAR_SUBSUBPROCESS = False
JIT_COMPILE_TRAIN_STEP = False
N_EPOCHS = 100

tabgan_args_dict = {
    "batch_size": 500,
    "jit_compile_train_step": JIT_COMPILE_TRAIN_STEP,
    # WGAN parameters
    "n_critic": 10,
    "wgan_lambda": 10,
    # Optimizer parameters
    "optimizer": "adam",
    "opt_lr": 0.0002,
    "adam_beta1": 0.5,
    "adam_beta2": 0.999,
    # Transformation parameters
    "quantile_rand_transformation": True,
    "quantile_transformation_int": True,
    "qtr_spread": 0.8,
    "qtr_lbound_apply": 0.05,
    "max_quantile_share": 1,
    "n_quantiles_int": 1000,
    "qt_n_subsample": 1e5,
    "noise_discrete_unif_max": 0.01,
    # Neural network parameters
    "gumbel_temperature": 0.1,
    "activation_function": "GELU",
    "gelu_approximate": True,
    "dim_hidden": 256,
    "dim_latent": 128,
    # Conditional sampling parameters
    "ctgan": True,
    "ctgan_binomial_loss": True,
    "ctgan_log_frequency": True,
    # Packing parameters
    "pac": 1
}

dataset_train_path = os.path.join(const.dir.data(), "df_adult_edited_train.csv")
data_train = pd.read_csv(dataset_train_path)

hp_info = {}

def create_tabGAN_for_qtr_spread(qtr_spread):
    temp_args_dict = dict(tabgan_args_dict)
    temp_args_dict["qtr_spread"] = qtr_spread
    tg_qtr = TabGAN(data_train, **temp_args_dict)
    return tg_qtr

hp_info["qtr_spread"] = {
    "vec": np.round(np.linspace(0, 1, 21), 2),
    "n_synthetic_datasets": 10,
    "n_epochs": 300,
    "tabGAN_func": create_tabGAN_for_qtr_spread,
    "batch_size": 500
}



