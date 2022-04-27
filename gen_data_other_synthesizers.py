from src import constants as const
from utils import print_header_method

import pandas as pd
import numpy as np
import os
import subprocess
from tqdm.auto import tqdm
import torch
from sdv.tabular import CTGAN, TVAE
import ctgan

print(f"GPU available: {torch.cuda.is_available()}")

restart_all = False
restart_specific = []

progress_bar = False

n_epochs = 300
batch_size = 500
n_critics = 10
n_synthetic_datasets = 25

os.makedirs(const.dir.data_gen(), exist_ok=True)
os.makedirs(const.dir.data_temp(), exist_ok=True)

data_train = pd.read_csv(os.path.join(const.dir.data(), "df_adult_edited_train.csv"))

def gen_datasets_tabfairgan(data_train, tabFairGAN_mod=False, progress_bar=True):
    if tabFairGAN_mod:
        name = "tabFairGAN-mod"
        dim_hidden_layer = 256
        dim_latent_layer = 128
    else:
        name = "tabFairGAN"
        dim_hidden_layer = -1 # same as setting equal to input dim
        dim_latent_layer = -1 # same as setting equal to input dim

    print_header_method(name)

    dataset_gen_dir_tabfairgan = os.path.join(const.dir.data_gen(), name)
    os.makedirs(dataset_gen_dir_tabfairgan, exist_ok=True)

    data_train_tabfairgan = data_train.copy(deep=True)
    data_train_tabfairgan.columns = data_train_tabfairgan.columns.str.replace(".", "_", regex = False)
    path_data_train_tabfairgan = os.path.join(const.dir.data_temp(), "df_adult_edited_train_tabFairGAN.csv")
    data_train_tabfairgan.to_csv(path_data_train_tabfairgan)

    with tqdm(range(n_synthetic_datasets), desc = "Generated datasets", disable = False) as pbar:
        for i in range(n_synthetic_datasets):
            print(pbar)
            dataset_train_gen_path = os.path.join(dataset_gen_dir_tabfairgan, f"gen{i}.csv")
            if restart_all or name in restart_specific or not os.path.exists(dataset_train_gen_path):
                subprocess.call(
                    f"python {os.path.join('src', 'TabFairGAN_nofair.py')} {path_data_train_tabfairgan} {n_epochs}"
                    f" {batch_size} {dataset_train_gen_path} --critic_repeat={n_critics}"
                    f" --dim_latent_layer={dim_latent_layer} --dim_hidden_layer={dim_hidden_layer}"
                    f" --size_of_fake_data={data_train.shape[0]} --progress_bar={progress_bar}".split()
                )
                data_train_gen = pd.read_csv(dataset_train_gen_path, index_col=0)
                data_train_gen.columns = data_train_gen.columns.str.replace("_", ".")
                data_train_gen.to_csv(dataset_train_gen_path, index=False)

def gen_datasets_ctgan(data_train, pac=10, log_frequency=True, orig_implementation=False):
    if orig_implementation:
        ctgan_method = ctgan.CTGANSynthesizer
        method_name = f"CTGAN-orig-pac{pac}"
    else:
        ctgan_method = CTGAN
        method_name = f"CTGAN-pac{pac}"
    if not log_frequency:
        method_name += "-log_freq=False"
    dir_gen_data_ctgan = os.path.join(const.dir.data_gen(), method_name)
    os.makedirs(dir_gen_data_ctgan, exist_ok=True)
    print_header_method(method_name)

    with tqdm(total=n_synthetic_datasets) as pbar:
        for i in range(n_synthetic_datasets):
            curr_path = os.path.join(dir_gen_data_ctgan, f"gen{i}.csv")
            if restart_all or method_name in restart_specific or not os.path.exists(curr_path):
                model = ctgan_method(epochs=n_epochs, batch_size=batch_size, discriminator_steps=n_critics, verbose=0,
                                     cuda=True, embedding_dim=128, generator_dim=(256, 256),
                                     discriminator_dim=(256, 256), pac=pac, log_frequency=log_frequency
                                     )
                if orig_implementation:
                    model.fit(data_train, discrete_columns=data_train.select_dtypes(exclude=[np.number]).columns.values)
                    data_gen = model.sample(data_train.shape[0])
                else:
                    model.fit(data_train)
                    data_gen = model.sample(num_rows=data_train.shape[0])
                data_gen.to_csv(curr_path, index=False)
            pbar.update(1)
            print(pbar)

def gen_datasets_tvae(data_train, modified=False):
    method_name = "TVAE"
    if modified:
        method_name += "-mod"
        hidden_dim = 256
    else:
        hidden_dim = 128

    dir_gen_data_tvae = os.path.join(const.dir.data_gen(), method_name)
    os.makedirs(dir_gen_data_tvae, exist_ok=True)
    print_header_method(method_name)

    with tqdm(total=n_synthetic_datasets) as pbar:
        for i in range(n_synthetic_datasets):
            curr_path = os.path.join(dir_gen_data_tvae, f"gen{i}.csv")
            if restart_all or method_name in restart_specific or not os.path.exists(curr_path):
                model = TVAE(epochs=n_epochs, batch_size=batch_size, cuda=True,
                             embedding_dim=128, compress_dims=(hidden_dim, hidden_dim),
                             decompress_dims=(hidden_dim, hidden_dim),
                             loss_factor=2
                             )
                model.fit(data_train)
                data_gen = model.sample(num_rows=data_train.shape[0])
                data_gen.to_csv(curr_path, index=False)
            pbar.update(1)
            print(pbar)


gen_datasets_tabfairgan(data_train, tabFairGAN_mod=False)
gen_datasets_tabfairgan(data_train, tabFairGAN_mod=True)
gen_datasets_ctgan(data_train, pac=10, log_frequency=True)
gen_datasets_ctgan(data_train, pac=1, log_frequency=True)
gen_datasets_tvae(data_train, modified=False)
gen_datasets_tvae(data_train, modified=True)
gen_datasets_ctgan(data_train, pac=2, log_frequency=True)
gen_datasets_ctgan(data_train, pac=10, log_frequency=False)
gen_datasets_ctgan(data_train, pac=1, log_frequency=False)
gen_datasets_ctgan(data_train, pac=10, log_frequency=True, orig_implementation=True)
