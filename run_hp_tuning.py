import pathlib
import os
print("Starting hyperparameter tuning")

import utils
import importlib
tabGAN = importlib.import_module("tabGAN")
from tabGAN import TableGAN
from src import constants as const
import numpy as np
import pandas as pd

n_critic_vec = np.arange(1,26)
n_synthetic_datasets_n_critic_comparison = 10
n_epochs_n_critic = 100

n_epochs = 100
n_critic = 10
adam_lr = 0.0002
adam_beta1 = 0.5
noise_discrete_unif_max = 0

batch_size = 500

adam_beta1_vec_partial = [0, 0.5, 0.9]
adam_beta2_vec_partial = [0.01, 0.5, 0.999]
adam_betas_vec = [(beta1, beta2) for beta1 in adam_beta1_vec_partial for beta2 in adam_beta2_vec_partial]
n_synthetic_datasets_adam_betas_comparison = 10
n_epochs_adam_betas = 100
print("adam_betas:", adam_betas_vec)

dataset_train_path = os.path.join(const.dir.data(), "df_adult_edited_train.csv")
dataset_test_path = os.path.join(const.dir.data(), "df_adult_edited_test.csv")

data_train = pd.read_csv(dataset_train_path)
data_test = pd.read_csv(dataset_test_path)
discrete_columns = data_train.columns[data_train.dtypes == "object"]

def create_tabGAN_for_adam_betas(adam_beta1, adam_beta2):
    tg_qtr = TableGAN(data_train, n_critic = n_critic, adam_lr = adam_lr, adam_beta1 = adam_beta1,
                      adam_beta2=adam_beta2, quantile_transformation_int = True, quantile_rand_transformation = True,
                      noise_discrete_unif_max = noise_discrete_unif_max)
    return tg_qtr

utils.hp_tuning.generate_multiple_datasets_for_multiple_hyperparameters(
    create_tabGAN_func=create_tabGAN_for_adam_betas,
    hyperparams_vec=adam_betas_vec,
    n_epochs=n_epochs_adam_betas,
    dataset_dir=const.dir.hyperparams_tuning(),
    batch_size=batch_size,
    subfolder="tabGAN-qtr",
    n_synthetic_datasets=n_synthetic_datasets_adam_betas_comparison,
    restart = True,
    redo_hyperparams_vec = [],
    plot_only_new_progress = True,
    hyperparams_name = "adam_betas",
    add_comparison_folder=True,
    overwrite_dataset=False,
    progress_bar_subprocess=True
)
