print("Starting hyperparameter tuning on Idun")
import os
import helpers.hp_tuning.hp_gen
from tabGAN import TabGAN
from src import constants as const
import numpy as np
import pandas as pd
import copy

JIT_COMPILE_TRAIN_STEP = False
N_EPOCHS = 100
BATCH_SIZE = 500

tabgan_args_dict = {
    "batch_size": BATCH_SIZE,
    "jit_compile": JIT_COMPILE_TRAIN_STEP,
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
    "activation_function": "LeakyReLU",
    #"gelu_approximate": True,
    "dim_hidden": 256,
    "dim_latent": 128,
    # Conditional sampling parameters
    "ctgan": False,
    # "ctgan_binomial_loss": True,
    # "ctgan_log_frequency": True,
    # Packing parameters
    "pac": 1
}

ctabgan_args_dict = {
    "batch_size": BATCH_SIZE,
    "jit_compile": JIT_COMPILE_TRAIN_STEP,
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
    "noise_discrete_unif_max": 0,
    # Neural network parameters
    "gumbel_temperature": 0.5,
    "activation_function": "LeakyReLU",
    #"gelu_approximate": True,
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

def fetch_hp_info(method="ctabGAN-qtr"):
    if method == "tabGAN-qtr":
        method_args_dict = tabgan_args_dict
    elif method == "ctabGAN-qtr":
        method_args_dict = ctabgan_args_dict
    else:
        raise ValueError(f"Entered method name {method} has not yet any hyperparameter tuning info")

    hp_info = {}

    def create_tabGAN_for_qtr_spread(qtr_spread):
        temp_args_dict = copy.deepcopy(method_args_dict)
        temp_args_dict["qtr_spread"] = qtr_spread
        tg_qtr = TabGAN(data_train, **temp_args_dict)
        return tg_qtr

    hp_info["qtr_spread"] = {
        "vec": np.round(np.linspace(0, 1, 21), 2),
        "n_synthetic_datasets": 10,
        "n_epochs": N_EPOCHS,
        "tabGAN_func": create_tabGAN_for_qtr_spread,
        "batch_size": BATCH_SIZE,
        "hyperparams_subname": None
    }

    noise_discrete_unif_max_vec_partial = np.arange(0, 0.21, 0.01).tolist() + [0.001, 0.003, 0.005, 0.007, 0.015, 0.025]
    gumbel_temp_vec_partial = [0.1, 0.2, 0.3, 0.4, 0.5, 0.7, 1]
    noise_and_gumbel_temp_vec = [(noise_discrete_unif_max, gumbel_temp)
                                 for noise_discrete_unif_max in noise_discrete_unif_max_vec_partial
                                 for gumbel_temp in gumbel_temp_vec_partial]

    def create_tabGAN_for_noise_and_gumbel_temperature(noise_discrete_unif_max, gumbel_temperature):
        temp_args_dict = copy.deepcopy(method_args_dict)
        temp_args_dict["noise_discrete_unif_max"] = noise_discrete_unif_max
        temp_args_dict["gumbel_temperature"] = gumbel_temperature
        tg_qtr = TabGAN(data_train, **temp_args_dict)
        return tg_qtr

    hp_info["oh_encoding_choices"] = {
        "vec": noise_and_gumbel_temp_vec,
        "n_synthetic_datasets": 10,
        "n_epochs": N_EPOCHS,
        "tabGAN_func": create_tabGAN_for_noise_and_gumbel_temperature,
        "batch_size": BATCH_SIZE,
        "hyperparams_subname": ["noise_discrete_unif_max", "gumbel_temp"]
    }

