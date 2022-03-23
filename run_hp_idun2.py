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

max_quantile_share_partial = np.round(np.arange(0.3, 1.01, 0.1), 2).tolist()
qtr_spread_vec_partial = np.round(np.arange(0, 0.81, 0.2), 2).tolist()
qt_params_vec = [(max_share, qtr_spread)
                 for max_share in max_quantile_share_partial
                 for qtr_spread in qtr_spread_vec_partial]
n_synthetic_datasets_qt_params_comparison = 25
n_epochs_qt_params = 100

def create_tabGAN_for_qt_params(max_quantile_share, qtr_spread):
    tg_qtr = TabGAN(data_train, n_critic = n_critic, optimizer="adam", opt_lr = opt_lr,
                    quantile_transformation_int = True, quantile_rand_transformation = True,
                    qtr_spread=qtr_spread, max_quantile_share=max_quantile_share)
    return tg_qtr

helpers.hp_tuning.generate_multiple_datasets_for_multiple_hyperparameters(
    create_tabGAN_func=create_tabGAN_for_qt_params,
    hyperparams_vec=qt_params_vec,
    n_epochs=n_epochs_qt_params,
    dataset_dir=const.dir.hyperparams_tuning(),
    batch_size=batch_size,
    subfolder="tabGAN-qtr",
    n_synthetic_datasets=n_synthetic_datasets_qt_params_comparison,
    restart = True,
    redo_hyperparams_vec = [],
    plot_only_new_progress = True,
    hyperparams_name = "qt_params",
    hyperparams_subname = ["max_quantile_share", "qtr_spread"],
    add_comparison_folder=True,
    overwrite_dataset=False,
    progress_bar_subprocess=True,
    progress_bar_subsubprocess=progress_bar_subsubprocess
)