print("Starting hyperparameter tuning on Idun")
import os
import utils.hp_tuning.hp_gen
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
progress_bar_subsubprocess=False
jit_compile_train_step=False

const.dir.storage = lambda: "/cluster/work/arneir"
print("Storage dir:", const.dir.storage())

dataset_train_path = os.path.join(const.dir.data(), "df_adult_edited_train.csv")
dataset_test_path = os.path.join(const.dir.data(), "df_adult_edited_test.csv")

data_train = pd.read_csv(dataset_train_path)
data_test = pd.read_csv(dataset_test_path)
discrete_columns = data_train.columns[data_train.dtypes == "object"]

noise_discrete_unif_max_vec_partial = np.arange(0, 0.21, 0.01) + [0.005, 0.015, 0.025]
gumbel_temp_vec_partial = [0.1, 0.3, 0.5, 0.7, 1]
noise_and_gumbel_temp_vec = [(noise_discrete_unif_max, gumbel_temp)
                             for noise_discrete_unif_max in noise_discrete_unif_max_vec_partial
                             for gumbel_temp in gumbel_temp_vec_partial]
n_synthetic_datasets_noise_and_gumbel_temp_comparison = 10
n_epochs_noise_and_gumbel_temp = 100

def create_tabGAN_for_noise_and_gumbel_temp(noise_discrete_unif_max, gumbel_temp):
    tg_qtr = TabGAN(data_train, n_critic = n_critic, optimizer="adam", opt_lr = opt_lr,
                    quantile_transformation_int = True, quantile_rand_transformation = True,
                    noise_discrete_unif_max = noise_discrete_unif_max,
                    gumbel_temperature=gumbel_temp, jit_compile_train_step=jit_compile_train_step)
    return tg_qtr

utils.hp_tuning.generate_multiple_datasets_for_multiple_hyperparameters(
    create_tabGAN_func=create_tabGAN_for_noise_and_gumbel_temp,
    hyperparams_vec=noise_and_gumbel_temp_vec,
    n_epochs=n_epochs_noise_and_gumbel_temp,
    dataset_dir=const.dir.hyperparams_tuning(),
    batch_size=batch_size,
    subfolder="tabGAN-qtr",
    n_synthetic_datasets=n_synthetic_datasets_noise_and_gumbel_temp_comparison,
    restart = True,
    redo_hyperparams_vec = [],
    plot_only_new_progress = True,
    hyperparams_name = "oh_encoding_choices",
    hyperparams_subname = ["noise_discrete_unif_max", "gumbel_temp"],
    add_comparison_folder=True,
    overwrite_dataset=False,
    progress_bar_subprocess=True,
    progress_bar_subsubprocess=progress_bar_subsubprocess
)