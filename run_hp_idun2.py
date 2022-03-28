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

gan_algorithms_partial_vec = ["WGAN-GP", "WGAN-SGP"]
wgan_lambda_partial_vec = [4, 6, 8, 10, 12, 14]
gan_method_vec = [(wgan_lambda, algorithm)
                  for wgan_lambda in wgan_lambda_partial_vec
                  for algorithm in gan_algorithms_partial_vec
                  ]
n_synthetic_datasets_gan_method_comparison = 25
n_epochs_gan_method = 100

def create_tabGAN_for_gan_method(wgan_lambda, gan_method):
    tg_qtr = TabGAN(data_train, n_critic = n_critic, opt_lr = opt_lr, adam_beta1 = adam_beta1,
                    quantile_transformation_int = True, quantile_rand_transformation = True,
                    noise_discrete_unif_max = noise_discrete_unif_max, tf_data_use=True,
                    gan_method=gan_method, wgan_lambda=wgan_lambda)
    return tg_qtr

helpers.hp_tuning.generate_multiple_datasets_for_multiple_hyperparameters(
    create_tabGAN_func=create_tabGAN_for_gan_method,
    hyperparams_vec=gan_method_vec,
    hyperparams_name = "gan_method",
    hyperparams_subname=["lambda", "algorithm"],
    n_epochs=n_epochs_gan_method,
    dataset_dir=const.dir.hyperparams_tuning(),
    batch_size=batch_size,
    subfolder="tabGAN-qtr",
    n_synthetic_datasets=n_synthetic_datasets_gan_method_comparison,
    restart = True,
    redo_hyperparams_vec = [],
    plot_only_new_progress = True,
    add_comparison_folder=True,
    overwrite_dataset=False,
    progress_bar_subprocess=True,
    progress_bar_subsubprocess=progress_bar_subsubprocess
)
