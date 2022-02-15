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

dataset_train_path = os.path.join(const.dir.data(), "df_adult_edited_train.csv")
dataset_test_path = os.path.join(const.dir.data(), "df_adult_edited_test.csv")

data_train = pd.read_csv(dataset_train_path)
data_test = pd.read_csv(dataset_test_path)
discrete_columns = data_train.columns[data_train.dtypes == "object"]


gumbel_temp_vec = np.round(np.linspace(0.001, 0.009, 9), 4).tolist()
gumbel_temp_vec += np.round(np.linspace(0.01, 0.19, 19), 3).tolist()
gumbel_temp_vec += np.round(np.linspace(0.2, 2, 19),2).tolist()
print("Creating Gumbel hp:", gumbel_temp_vec)
n_synthetic_datasets_gumbel_temp_comparison = 10
n_epochs_gumbel_temp = 100

def create_tabGAN_for_gumbel_temp(gumbel_temp):
    tg_qtr = TableGAN(data_train, n_critic = n_critic, adam_lr = adam_lr, adam_beta1 = adam_beta1,
                      quantile_transformation_int = True, quantile_rand_transformation = True,
                      noise_discrete_unif_max = noise_discrete_unif_max,
                      gumbel_temperature=gumbel_temp)
    return tg_qtr

utils.hp_tuning.generate_multiple_datasets_for_multiple_hyperparameters(
    create_tabGAN_func=create_tabGAN_for_gumbel_temp,
    hyperparams_vec=gumbel_temp_vec,
    n_epochs=n_epochs_gumbel_temp,    dataset_dir=const.dir.hyperparams_tuning(),
    batch_size=batch_size,
    subfolder="tabGAN-qtr",
    n_synthetic_datasets=n_synthetic_datasets_gumbel_temp_comparison,
    restart = True,
    redo_hyperparams_vec = [],
    plot_only_new_progress = True,
    hyperparams_name = "gumbel_temp",
    add_comparison_folder=True,
    overwrite_dataset=False,
    progress_bar_subprocess=True,
    force_tqdm_cmd=False
)

dim_hidden_vec = [16, 32, 64, 96, 128, 192, 256, 384, 512, 786, 1024]
n_synthetic_datasets_dim_hidden_comparison = 10
n_epochs_dim_hidden = 100

def create_tabGAN_for_dim_hidden(dim_hidden):
    tg_qtr = TableGAN(data_train, n_critic = n_critic, adam_lr = adam_lr, adam_beta1 = adam_beta1,
                      quantile_transformation_int = True, quantile_rand_transformation = True,
                      noise_discrete_unif_max = noise_discrete_unif_max,
                      dim_hidden=dim_hidden)
    return tg_qtr

utils.hp_tuning.generate_multiple_datasets_for_multiple_hyperparameters(
    create_tabGAN_func=create_tabGAN_for_dim_hidden,
    hyperparams_vec=dim_hidden_vec,
    n_epochs=n_epochs_dim_hidden,
    dataset_dir=const.dir.hyperparams_tuning(),
    batch_size=batch_size,
    subfolder="tabGAN-qtr",
    n_synthetic_datasets=n_synthetic_datasets_dim_hidden_comparison,
    restart = True,
    redo_hyperparams_vec = [],
    plot_only_new_progress = True,
    hyperparams_name = "dim_hidden",
    add_comparison_folder=True,
    overwrite_dataset=False,
    progress_bar_subprocess=True,
    force_tqdm_cmd=False
)