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

activation_function_vec = [("LeakyReLU", False), ("GELU", False), ("GELU", True)]
n_synthetic_datasets_activation_function_comparison = 10
n_epochs_activation_function = 100

def create_tabGAN_for_activation_function(activation_function, approximate):
    tg_qtr = TabGAN(data_train, n_critic = n_critic, opt_lr = opt_lr, adam_beta1 = adam_beta1,
                    quantile_transformation_int = True, quantile_rand_transformation = True,
                    noise_discrete_unif_max = noise_discrete_unif_max, tf_data_use=True,
                    activation_function=activation_function, gelu_approximate=approximate)
    return tg_qtr

helpers.hp_tuning.generate_multiple_datasets_for_multiple_hyperparameters(
    create_tabGAN_func=create_tabGAN_for_activation_function,
    hyperparams_vec=activation_function_vec,
    n_epochs=n_epochs_activation_function,
    dataset_dir=const.dir.hyperparams_tuning(),
    batch_size=batch_size,
    subfolder="tabGAN-qtr",
    n_synthetic_datasets=n_synthetic_datasets_activation_function_comparison,
    restart=False,
    redo_hyperparams_vec = [],
    hyperparams_name = "activation",
    hyperparams_subname=["function", "approximate"],
    add_comparison_folder=True,
    overwrite_dataset=False,
    progress_bar_subprocess=True,
    progress_bar_subsubprocess=progress_bar_subsubprocess
)



ctgan_vec = [(False, False, False, False)]
ctgan_vec += [(bin_loss, True, log_freq, add_connection)
              for bin_loss in [False, True]
              for log_freq in [False, True]
              for add_connection in [False, True]]
n_synthetic_datasets_ctgan_comparison = 25
n_epochs_ctgan = 100

def create_tabGAN_for_ctgan(ctgan, ctgan_log_frequency, ctgan_binomial_loss, add_connection_query_to_discrete):
    if ctgan:
        tf_data_use=False
    else:
        tf_data_use=True
    tg_qtr = TabGAN(data_train, n_critic = n_critic, opt_lr = opt_lr, adam_beta1 = adam_beta1,
                    quantile_transformation_int = True, quantile_rand_transformation = True,
                    noise_discrete_unif_max = noise_discrete_unif_max, tf_data_use=tf_data_use,
                    ctgan=ctgan, ctgan_log_frequency=ctgan_log_frequency,
                    ctgan_binomial_loss=ctgan_binomial_loss,
                    add_connection_query_to_discrete=add_connection_query_to_discrete)
    return tg_qtr

helpers.hp_tuning.generate_multiple_datasets_for_multiple_hyperparameters(
    create_tabGAN_func=create_tabGAN_for_ctgan,
    hyperparams_vec=ctgan_vec,
    n_epochs=n_epochs_ctgan,
    dataset_dir=const.dir.hyperparams_tuning(),
    batch_size=batch_size,
    subfolder="tabGAN-qtr",
    n_synthetic_datasets=n_synthetic_datasets_ctgan_comparison,
    restart = True,
    redo_hyperparams_vec = [],
    hyperparams_name = "categorical_query",
    hyperparams_subname=["ctgan_binomial_loss", "ctgan", "log_frequency", "add_connection_query_to_discrete"],
    add_comparison_folder=True,
    overwrite_dataset=False,
    progress_bar_subprocess=True,
    progress_bar_subsubprocess=progress_bar_subsubprocess
)