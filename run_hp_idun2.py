print("Starting hyperparameter tuning on Idun")
import os
import helpers.hp_tuning.hp_gen
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
progress_bar_subsubprocess = False
jit_compile_train_step = False

const.dir.storage = lambda: "/cluster/work/arneir"
print("Storage dir:", const.dir.storage())

dataset_train_path = os.path.join(const.dir.data(), "df_adult_edited_train.csv")
dataset_test_path = os.path.join(const.dir.data(), "df_adult_edited_test.csv")

data_train = pd.read_csv(dataset_train_path)
data_test = pd.read_csv(dataset_test_path)
discrete_columns = data_train.columns[data_train.dtypes == "object"]

# tg_qtr = TabGAN(data_train, n_critic = n_critic, opt_lr = opt_lr, adam_beta1 = adam_beta1,
#                 quantile_transformation_int = True, quantile_rand_transformation = False,
#                 noise_discrete_unif_max = noise_discrete_unif_max,
#                 gumbel_temperature=0.5, jit_compile=False)
# n_epochs_vec = np.arange(1,5).tolist() + np.arange(5, 1001, 5).tolist()
# n_synthetic_datasets_epochs_comparison = 25
#
# helpers.hp_tuning.generate_multiple_datasets_for_multiple_epochs_fast(
#     tg_qtr,
#     dataset_dir = const.dir.hyperparams_tuning(),
#     subfolder = "tabGAN-qtr",
#     batch_size=batch_size,
#     n_synthetic_datasets = n_synthetic_datasets_epochs_comparison,
#     n_epochs_vec = n_epochs_vec,
#     redo_n_epochs_vec=[],
#     overwrite_dataset=False,
#     restart=False)

gumbel_temp_vec = np.round(np.linspace(0.001, 0.009, 9), 4).tolist()
gumbel_temp_vec += np.round(np.linspace(0.01, 0.19, 19), 3).tolist()
gumbel_temp_vec += np.round(np.linspace(0.2, 2, 19),2).tolist()
n_synthetic_datasets_gumbel_temp_comparison = 10
n_epochs_gumbel_temp = 100

def create_tabGAN_for_gumbel_temp(gumbel_temp):
    tg_qtr = TabGAN(data_train, n_critic = n_critic, opt_lr = opt_lr, adam_beta1 = adam_beta1,
                    quantile_transformation_int = True, quantile_rand_transformation = True,
                    noise_discrete_unif_max = noise_discrete_unif_max,
                    gumbel_temperature=gumbel_temp)
    return tg_qtr

helpers.hp_tuning.generate_multiple_datasets_for_multiple_hyperparameters(
    create_tabGAN_func=create_tabGAN_for_gumbel_temp,
    hyperparams_vec=gumbel_temp_vec,
    n_epochs=n_epochs_gumbel_temp,
    dataset_dir=const.dir.hyperparams_tuning(),
    batch_size=batch_size,
    subfolder="tabGAN-qtr",
    n_synthetic_datasets=n_synthetic_datasets_gumbel_temp_comparison,
    restart = False,
    redo_hyperparams_vec = [],
    plot_only_new_progress = True,
    hyperparams_name = "gumbel_temp",
    add_comparison_folder=True,
    overwrite_dataset=False,
    progress_bar_subprocess=True,
)

qtr_lbound_apply_vec = np.round(np.linspace(0.02, 0.2, 10),2).tolist() + np.round(np.linspace(0.002, 0.01, 5),3).tolist()
n_synthetic_datasets_qtr_lbound_apply_comparison = 10
n_epochs_qtr_lbound_apply = 100

def create_tabGAN_for_qtr_lbound_apply(qtr_lbound_apply):
    tg_qtr = TabGAN(data_train, n_critic = n_critic, opt_lr = opt_lr, adam_beta1 = adam_beta1,
                    quantile_transformation_int = True, quantile_rand_transformation = True,
                    noise_discrete_unif_max = noise_discrete_unif_max,
                    qtr_lbound_apply=qtr_lbound_apply)
    return tg_qtr

helpers.hp_tuning.generate_multiple_datasets_for_multiple_hyperparameters(
    create_tabGAN_func=create_tabGAN_for_qtr_lbound_apply,
    hyperparams_vec=qtr_lbound_apply_vec,
    n_epochs=n_epochs_qtr_lbound_apply,
    dataset_dir=const.dir.hyperparams_tuning(),
    batch_size=batch_size,
    subfolder="tabGAN-qtr",
    n_synthetic_datasets=n_synthetic_datasets_qtr_lbound_apply_comparison,
    restart=False,
    redo_hyperparams_vec = [],
    plot_only_new_progress = True,
    hyperparams_name = "qtr_lbound_apply",
    add_comparison_folder=True,
    overwrite_dataset=False,
    progress_bar_subprocess=True,
)


rmsprop_rho_vec_partial = [0, 0.1, 0.3, 0.5, 0.7, 0.9, 0.99]
rmsprop_momentum_vec_partial = [0.01, 0.1, 0.3, 0.5, 0.7, 0.9, 0.999]
rmsprop_centered_vec_partial = [False, True]
rmsprop_vec = [(rho, momentum, centered) for rho in rmsprop_rho_vec_partial for momentum in rmsprop_momentum_vec_partial
               for centered in rmsprop_centered_vec_partial]
n_synthetic_datasets_rmsprop_comparison = 10
n_epochs_rmsprop = 100

def create_tabGAN_for_rmsprop(rmsprop_rho, rmsprop_momentum, rmsprop_centered):
    tg_qtr = TabGAN(data_train, n_critic = n_critic, opt_lr = opt_lr, rmsprop_rho = rmsprop_rho, rmsprop_centered=rmsprop_centered,
                    rmsprop_momentum=rmsprop_momentum, quantile_transformation_int = True, quantile_rand_transformation = True,
                    noise_discrete_unif_max = noise_discrete_unif_max)
    return tg_qtr

helpers.hp_tuning.generate_multiple_datasets_for_multiple_hyperparameters(
    create_tabGAN_func=create_tabGAN_for_rmsprop,
    hyperparams_vec=rmsprop_vec,
    n_epochs=n_epochs_rmsprop,
    dataset_dir=const.dir.hyperparams_tuning(),
    batch_size=batch_size,
    subfolder="tabGAN-qtr",
    n_synthetic_datasets=n_synthetic_datasets_rmsprop_comparison,
    restart = False,
    redo_hyperparams_vec = [],
    plot_only_new_progress = True,
    hyperparams_name = "rmsprop",
    hyperparams_subname = ["rho", "momentum", "centered"],
    add_comparison_folder=True,
    overwrite_dataset=False,
    progress_bar_subprocess=True
)

noise_discrete_unif_max_vec_partial = np.arange(0, 0.21, 0.01).tolist() + [0.001, 0.003, 0.005, 0.007, 0.015, 0.025]
gumbel_temp_vec_partial = [0.1, 0.2, 0.3, 0.4, 0.5, 0.7, 1]
noise_and_gumbel_temp_vec = [(noise_discrete_unif_max, gumbel_temp)
                             for noise_discrete_unif_max in noise_discrete_unif_max_vec_partial
                             for gumbel_temp in gumbel_temp_vec_partial]
n_synthetic_datasets_noise_and_gumbel_temp_comparison = 10
n_epochs_noise_and_gumbel_temp = 100

def create_tabGAN_for_noise_and_gumbel_temp(noise_discrete_unif_max, gumbel_temp):
    tg_qtr = TabGAN(data_train, n_critic = n_critic, optimizer="adam", opt_lr = opt_lr,
                    quantile_transformation_int = True, quantile_rand_transformation = True,
                    noise_discrete_unif_max = noise_discrete_unif_max,
                    gumbel_temperature=gumbel_temp)
    return tg_qtr

helpers.hp_tuning.generate_multiple_datasets_for_multiple_hyperparameters(
    create_tabGAN_func=create_tabGAN_for_noise_and_gumbel_temp,
    hyperparams_vec=noise_and_gumbel_temp_vec,
    n_epochs=n_epochs_noise_and_gumbel_temp,
    dataset_dir=const.dir.hyperparams_tuning(),
    batch_size=batch_size,
    subfolder="tabGAN-qtr",
    n_synthetic_datasets=n_synthetic_datasets_noise_and_gumbel_temp_comparison,
    restart = False,
    redo_hyperparams_vec = [],
    plot_only_new_progress = True,
    hyperparams_name = "oh_encoding_choices",
    hyperparams_subname = ["noise_discrete_unif_max", "gumbel_temp"],
    add_comparison_folder=True,
    overwrite_dataset=False,
    progress_bar_subprocess=True,
    progress_bar_subsubprocess=progress_bar_subsubprocess
)