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

n_synthetic_datasets = 25
jit_compile = False

def gen_datasets_tabgan(data_train, quantile_transformation=False, quantile_transformation_randomized=False,
                        ctgan=False, ctgan_log_freq=True, pac=1, qtr_spread=0.4, hp=False,
                        noise_discrete_unif_max=0, hp2=False):
    if sum([hp, hp2, qtr_spread != 0.4]) > 1:
        raise ValueError("qtr_spread is changed when using hyperparameters found after tuning."
                         " Thus qtr_spread can't be changed when entering hp=True or hp2=True."
                         " Also hp=True and hp2=True can't be entered simultaneously.")
    method_name = ""
    if ctgan:
        method_name += "c"
    method_name += "tabGAN"
    if quantile_transformation:
        method_name += "-qt"
        if quantile_transformation_randomized:
            method_name += "r"
    if pac > 1:
        method_name += f"-pac{pac}"
    if ctgan and not ctgan_log_freq:
        method_name += "-log_freq=False"
    if qtr_spread != 0.4:
        method_name += f"-qtr_spread={qtr_spread}"

    extra_tabGAN_params = {}
    if hp or hp2:
        method_name += "-hp"
        qtr_spread = 0.8
        if ctgan:
            pass
        else:
            noise_discrete_unif_max = 0.01
            extra_tabGAN_params["gumbel_temperature"] = 0.1
        if hp2:
            method_name += "2"
            extra_tabGAN_params["activation_function"] = "GELU"
            extra_tabGAN_params["gelu_approximate"] = False
            extra_tabGAN_params["gumbel_temperature"] = 0.5

    print_header_method(method_name)

    tg = TabGAN(data_train, n_critic=n_critic, opt_lr=opt_lr, adam_beta1=adam_beta1,
                quantile_transformation_int=quantile_transformation,
                quantile_rand_transformation=quantile_transformation_randomized,
                noise_discrete_unif_max=noise_discrete_unif_max, jit_compile=jit_compile,
                ctgan=ctgan, ctgan_log_frequency=ctgan_log_freq, tf_data_use=(not ctgan),
                pac=pac, qtr_spread=qtr_spread, **extra_tabGAN_params)
    # if hp2:
        #tg.train(300, progress_bar=True, restart_training=False)
        # helpers.generate_multiple_datasets(tg, const.dir.data_gen(), n_synthetic_datasets, n_epochs=5, subfolder=method_name,
        #                                    batch_size=batch_size, overwrite_dataset=False, progress_bar_dataset=False)
        # print(tg.sample_scaled())
        # print(tg.sample())
    helpers.generate_multiple_datasets(tg, const.dir.data_gen(), n_synthetic_datasets, n_epochs, subfolder=method_name,
                                           batch_size=batch_size, overwrite_dataset=False, progress_bar_dataset=False)


# hp-tuned
gen_datasets_tabgan(data_train, quantile_transformation=True, quantile_transformation_randomized=True,
                    ctgan=True, ctgan_log_freq=True, hp=True)
gen_datasets_tabgan(data_train, quantile_transformation=True, quantile_transformation_randomized=True,
                    ctgan=False, hp=True)
gen_datasets_tabgan(data_train, quantile_transformation=True, quantile_transformation_randomized=True,
                    ctgan=False, hp2=True)
gen_datasets_tabgan(data_train, quantile_transformation=True, quantile_transformation_randomized=True,
                    ctgan=True, ctgan_log_freq=True, hp2=True)

# tabGAN types
gen_datasets_tabgan(data_train, quantile_transformation=False, quantile_transformation_randomized=False)
gen_datasets_tabgan(data_train, quantile_transformation=True, quantile_transformation_randomized=False)
gen_datasets_tabgan(data_train, quantile_transformation=True, quantile_transformation_randomized=True)
gen_datasets_tabgan(data_train, quantile_transformation=True, quantile_transformation_randomized=True,
                    ctgan=False, pac=1, qtr_spread=0.8)
gen_datasets_tabgan(data_train, quantile_transformation=True, quantile_transformation_randomized=True,
                    ctgan=False, pac=2)
# ctabGAN types with pac=1
gen_datasets_tabgan(data_train, quantile_transformation=True, quantile_transformation_randomized=True,
                    ctgan=True, ctgan_log_freq=True)
gen_datasets_tabgan(data_train, quantile_transformation=True, quantile_transformation_randomized=True,
                    ctgan=True, ctgan_log_freq=False)
gen_datasets_tabgan(data_train, quantile_transformation=True, quantile_transformation_randomized=True,
                    ctgan=True, ctgan_log_freq=True, pac=1, qtr_spread=0.8)
gen_datasets_tabgan(data_train, quantile_transformation=True, quantile_transformation_randomized=True,
                    ctgan=False, pac=10)

# ctabGAN types with pac>1
gen_datasets_tabgan(data_train, quantile_transformation=True, quantile_transformation_randomized=True,
                    ctgan=True, ctgan_log_freq=False, pac=2)
gen_datasets_tabgan(data_train, quantile_transformation=True, quantile_transformation_randomized=True,
                    ctgan=True, ctgan_log_freq=True, pac=2)
gen_datasets_tabgan(data_train, quantile_transformation=True, quantile_transformation_randomized=True,
                    ctgan=True, ctgan_log_freq=False, pac=2, qtr_spread=0.8)
gen_datasets_tabgan(data_train, quantile_transformation=True, quantile_transformation_randomized=True,
                    ctgan=True, ctgan_log_freq=True, pac=2, qtr_spread=0.8)
gen_datasets_tabgan(data_train, quantile_transformation=True, quantile_transformation_randomized=True,
                    ctgan=True, ctgan_log_freq=False, pac=10, qtr_spread=0.8)
gen_datasets_tabgan(data_train, quantile_transformation=True, quantile_transformation_randomized=True,
                    ctgan=True, ctgan_log_freq=True, pac=10, qtr_spread=0.8)


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

