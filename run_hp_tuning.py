import pathlib
import os
print("Starting hyperparameter tuning")

import utils
import importlib
tabGAN = importlib.import_module("tabGAN")
from tabGAN import TabGAN
from src import constants as const
import numpy as np
import pandas as pd

n_epochs = 100
n_critic = 10
opt_lr = 0.0002
adam_beta1 = 0.5
noise_discrete_unif_max = 0

batch_size = 500

dataset_train_path = os.path.join(const.dir.data(), "df_adult_edited_train.csv")
dataset_test_path = os.path.join(const.dir.data(), "df_adult_edited_test.csv")

data_train = pd.read_csv(dataset_train_path)
data_test = pd.read_csv(dataset_test_path)
discrete_columns = data_train.columns[data_train.dtypes == "object"]

noise_discrete_unif_max_vec = np.round(np.arange(0, 0.31, 0.01), 3).tolist() + [0.005]
n_synthetic_datasets_noise_discrete_unif_max_comparison = 10
n_epochs_noise_discrete_unif_max = 100

def create_tabGAN_for_noise_discrete_unif_max(noise_discrete_unif_max):
    tg_qtr = TabGAN(data_train, n_critic = n_critic, opt_lr = opt_lr, adam_beta1 = adam_beta1,
                    quantile_transformation_int = True, quantile_rand_transformation = True,
                    noise_discrete_unif_max = noise_discrete_unif_max)
    return tg_qtr

utils.hp_tuning.generate_multiple_datasets_for_multiple_hyperparameters(
    create_tabGAN_func=create_tabGAN_for_noise_discrete_unif_max,
    hyperparams_vec=noise_discrete_unif_max_vec,
    n_epochs=n_epochs_noise_discrete_unif_max,
    dataset_dir=const.dir.hyperparams_tuning(),
    batch_size=batch_size,
    subfolder="tabGAN-qtr",
    n_synthetic_datasets=n_synthetic_datasets_noise_discrete_unif_max_comparison,
    restart = True,
    redo_hyperparams_vec = [],
    plot_only_new_progress = True,
    hyperparams_name = "noise_discrete_unif_max",
    add_comparison_folder=True,
    overwrite_dataset=False,
    progress_bar_subprocess=True
)

batch_size_vec = [50, 100, 250, 500, 750, 1000, 2500, 5000]
n_synthetic_datasets_batch_size_comparison = 10
n_epochs_batch_size = 100

def create_tabGAN_for_batch_size(batch_size):
    tg_qtr = TabGAN(data_train, n_critic = n_critic, opt_lr = opt_lr, adam_beta1 = adam_beta1,
                    quantile_transformation_int = True, quantile_rand_transformation = True,
                    noise_discrete_unif_max = noise_discrete_unif_max,
                    batch_size=batch_size)
    return tg_qtr

utils.hp_tuning.generate_multiple_datasets_for_multiple_hyperparameters(
    create_tabGAN_func=create_tabGAN_for_batch_size,
    hyperparams_vec=batch_size_vec,
    n_epochs=n_epochs_batch_size,
    dataset_dir=const.dir.hyperparams_tuning(),
    batch_size=None,
    subfolder="tabGAN-qtr",
    n_synthetic_datasets=n_synthetic_datasets_batch_size_comparison,
    restart = True,
    redo_hyperparams_vec = [],
    plot_only_new_progress = True,
    hyperparams_name = "batch_size",
    add_comparison_folder=True,
    overwrite_dataset=False,
    progress_bar_subprocess=True
)

batch_size_300_epochs_vec = [50, 100, 250, 500, 750, 1000, 2500, 5000]
n_synthetic_datasets_batch_size_300_epochs_comparison = 10
n_epochs_batch_size_300_epochs = 300

def create_tabGAN_for_batch_size_300_epochs(batch_size_300_epochs):
    tg_qtr = TabGAN(data_train, n_critic = n_critic, opt_lr = opt_lr, adam_beta1 = adam_beta1,
                    quantile_transformation_int = True, quantile_rand_transformation = True,
                    noise_discrete_unif_max = noise_discrete_unif_max,
                    batch_size=batch_size_300_epochs)
    return tg_qtr

utils.hp_tuning.generate_multiple_datasets_for_multiple_hyperparameters(
    create_tabGAN_func=create_tabGAN_for_batch_size_300_epochs,
    hyperparams_vec=batch_size_300_epochs_vec,
    n_epochs=n_epochs_batch_size_300_epochs,
    dataset_dir=const.dir.hyperparams_tuning(),
    batch_size=None,
    subfolder="tabGAN-qtr",
    n_synthetic_datasets=n_synthetic_datasets_batch_size_300_epochs_comparison,
    restart = True,
    redo_hyperparams_vec = [],
    plot_only_new_progress = True,
    hyperparams_name = "batch_size_300_epochs",
    add_comparison_folder=True,
    overwrite_dataset=False,
    progress_bar_subprocess=True
)