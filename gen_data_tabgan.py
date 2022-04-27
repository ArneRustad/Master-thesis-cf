from tabGAN import TabGAN
from src import constants as const
from utils import print_header_method
import helpers
import os
import pandas as pd

import numpy as np

dataset_train_path = os.path.join(const.dir.data(), "df_adult_edited_train.csv")
data_train = pd.read_csv(dataset_train_path)

n_epochs = 300
batch_size = 500

n_critic = 10
opt_lr = 0.0002
adam_beta1 = 0.5
noise_discrete_unif_max = 0

n_synthetic_datasets = 25
jit_compile = False

def gen_datasets_tabgan(data_train, quantile_transformation=False, quantile_transformation_randomized=False,
                        ctgan=False, ctgan_log_freq=True):
    method_name = ""
    if ctgan:
        method_name += "c"
    method_name += "tabGAN"
    if quantile_transformation:
        method_name += "-qt"
        if quantile_transformation_randomized:
            method_name += "r"
    if ctgan and not ctgan_log_freq:
        method_name += "-log_freq=False"
    print_header_method(method_name)

    tg = TabGAN(data_train, n_critic=n_critic, opt_lr=opt_lr, adam_beta1=adam_beta1,
                quantile_transformation_int=quantile_transformation,
                quantile_rand_transformation=quantile_transformation_randomized,
                noise_discrete_unif_max=noise_discrete_unif_max, jit_compile=jit_compile,
                ctgan=True, ctgan_log_frequency=ctgan_log_freq, tf_data_use=not ctgan)

    helpers.generate_multiple_datasets(tg, const.dir.data_gen(), n_synthetic_datasets, n_epochs, subfolder=method_name,
                                       batch_size=batch_size, overwrite_dataset=False, progress_bar_dataset=False)

gen_datasets_tabgan(data_train, quantile_transformation=False, quantile_transformation_randomized=False)
gen_datasets_tabgan(data_train, quantile_transformation=True, quantile_transformation_randomized=False)
gen_datasets_tabgan(data_train, quantile_transformation=True, quantile_transformation_randomized=True)
gen_datasets_tabgan(data_train, quantile_transformation=True, quantile_transformation_randomized=True,
                    ctgan=True, ctgan_log_freq=True)
# gen_datasets_tabgan(data_train, quantile_transformation=True, quantile_transformation_randomized=True,
#                     ctgan=True, ctgan_log_freq=False)

# tg_qtr = TabGAN(data_train, n_critic = n_critic, opt_lr = opt_lr, adam_beta1 = adam_beta1,
#                 quantile_transformation_int = True, quantile_rand_transformation = True,
#                 noise_discrete_unif_max = noise_discrete_unif_max,
#                 gumbel_temperature = 0.5, jit_compile=False)
# n_epochs_vec = np.arange(1, 5).tolist() + np.arange(5, 10001, 5).tolist()
# n_synthetic_datasets_epochs_comparison = 5
#
# helpers.hp_tuning.generate_multiple_datasets_for_multiple_epochs_fast(
#     tg_qtr,
#     dataset_dir = const.dir.hyperparams_tuning(),
#     subfolder = "tabGAN-qtr",
#     batch_size=batch_size,
#     n_synthetic_datasets = n_synthetic_datasets_epochs_comparison,
#     n_epochs_vec = n_epochs_vec,
#     overwrite_dataset=False)

