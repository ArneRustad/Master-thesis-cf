from src import constants as const

import pandas as pd
import numpy as np
import os
import subprocess
from tqdm.auto import tqdm
import torch

print(f"GPU available: {torch.cuda.is_available()}")

restart_all = True
restart_specific = []

n_epochs = 300
batch_size = 500
n_critics = 10
n_synthetic_datasets = 10

os.makedirs(const.dir.data_gen(), exist_ok=True)
os.makedirs(const.dir.data_temp(), exist_ok=True)

data_train_tabfairgan = pd.read_csv(os.path.join(const.dir.data(), "df_adult_edited_train.csv"))

dataset_gen_dir_tabfairgan = os.path.join(const.dir.data_gen(), "tabFairGAN")
dataset_gen_dir_tabfairgan_mod = os.path.join(const.dir.data_gen(), "tabFairGAN-mod")
os.makedirs(dataset_gen_dir_tabfairgan, exist_ok=True)
os.makedirs(dataset_gen_dir_tabfairgan_mod, exist_ok=True)

size_of_fake_data_train = data_train_tabfairgan.shape[0]

data_train_tabfairgan.columns = data_train_tabfairgan.columns.str.replace(".", "_", regex = False)
path_data_train_tabfairgan = os.path.join(const.dir.data_temp(), "df_adult_edited_train_tabFairGAN.csv")
data_train_tabfairgan.to_csv(path_data_train_tabfairgan)

dim_hidden_layer = -1 # same as setting equal to input dim
dim_latent_layer = -1 # same as setting equal to input dim
dim_hidden_layer_mod = 256
dim_latent_layer_mod = 128

dataset_train_gen_path_tabfairgan_mod = os.path.join(dataset_gen_dir_tabfairgan, f"gen{0}.csv")

print("-"*10 + "Generating tabFairGAN-mod datasets" + "-"*10)
with tqdm(range(n_synthetic_datasets), desc = "Generated datasets", disable = False) as pbar:
    for i in range(n_synthetic_datasets):
        print(pbar)
        dataset_train_gen_path_tabfairgan_mod = os.path.join(dataset_gen_dir_tabfairgan_mod, f"gen{i}.csv")
        subprocess.call(
            f"python {os.path.join('src', 'TabFairGAN_nofair.py')} {path_data_train_tabfairgan} {n_epochs}"
            f" {batch_size} {dataset_train_gen_path_tabfairgan_mod} --critic_repeat={n_critics}"
            f" --dim_latent_layer={dim_latent_layer_mod} --dim_hidden_layer={dim_hidden_layer_mod}".split()
        )
        data_train_gen_mod = pd.read_csv(dataset_train_gen_path_tabfairgan_mod)
        data_train_gen_mod.columns = data_train_gen_mod.columns.str.replace("_", ".")
        data_train_gen_mod.to_csv(dataset_train_gen_path_tabfairgan_mod, index=True)