from src import constants as const
from utils import print_header_method
import helpers.comparison

import pandas as pd
import numpy as np
import os
import re
from tqdm.auto import tqdm
import torch
from sdv.tabular import CTGAN, TVAE, GaussianCopula, CopulaGAN
import ctgan
from datetime import datetime
import subprocess

N_EPOCHS = 300
BATCH_SIZE = 500
N_CRITICS = 10
N_SYNTHETIC_DATASETS = 15 #10 for adult_edited

RESTART_ALL = False
RESTART_SPECIFIC = []
PROGRESS_BAR_TASK = True
PROGRESS_BAR_MODEL_FIT = False
DATASET_TASKS = ["covtype_edited", "creditcard_edited", "news_edited", "adult_edited"]

MODELS = ["TVAE", "TVAE-mod", "TVAESynthesizer", "TVAESynthesizer-mod", #count 4
          "CTGAN-pac10", "CTGAN-pac1", "CTGANSynthesizer-pac1", "CTGANSynthesizer-pac10", #count 8
          "tabFairGAN", "tabFairGAN-mod", "GaussianCopula", "CopulaGAN"]

slurm_array_task_id = os.getenv('SLURM_ARRAY_TASK_ID')
if slurm_array_task_id is not None:
    slurm_array_task_id = int(slurm_array_task_id)
    task_id = (slurm_array_task_id % 1000) // 100
    model_id = slurm_array_task_id % 100
    dataset_id = slurm_array_task_id // 1000
    if task_id > 0:
        if task_id <= len(DATASET_TASKS):
            DATASET_TASKS = [DATASET_TASKS[task_id - 1]]
            if DATASET_TASKS[0] == "adult_edited":
                N_SYNTHETIC_DATASETS = 25
            if DATASET_TASKS[0] == "news_edited":
                N_SYNTHETIC_DATASETS = 25
        else:
            raise ValueError(f"task_id can't be larger than length of DATASET_TASKS. You entered {dataset_id}.")
    if dataset_id > 0:
        if dataset_id <= N_SYNTHETIC_DATASETS:
            SPECIFIC_DATASET_NUMBER = dataset_id - 1
        else:
            raise ValueError(f"dataset_id can't be larger than N_SYNTHETIC_DATASETS. You entered {dataset_id}.")
    else:
        SPECIFIC_DATASET_NUMBER = None
    if model_id > 0:
        if model_id <= len(MODELS):
            MODELS = [MODELS[model_id - 1]]
        else:
            raise ValueError(f"model_id can't be larger than length of MODELS. You entered {model_id}.")

    print(f"Starting comparison array task with dataset_id {dataset_id} and model_id {model_id}")

dict_default_arguments = {
    "datasets": DATASET_TASKS,
    "n_synthetic_datasets": N_SYNTHETIC_DATASETS,
    "progress_bar_task": PROGRESS_BAR_TASK,
    "_specific_dataset_number": SPECIFIC_DATASET_NUMBER
}

def tvae_synthesizer(data_train, modified=False, orig=False):
    if modified:
        hidden_dim = 256
    else:
        hidden_dim = 128

    if orig:
        tvae_class = ctgan.TVAESynthesizer
    else:
        tvae_class = TVAE

    model = tvae_class(epochs=N_EPOCHS, batch_size=BATCH_SIZE, cuda=True,
                             embedding_dim=128, compress_dims=(hidden_dim, hidden_dim),
                             decompress_dims=(hidden_dim, hidden_dim),
                             loss_factor=2
                             )
    if orig:
        model.fit(data_train, discrete_columns=data_train.select_dtypes(exclude=[np.number]).columns.values)
        data_gen = model.sample(samples=data_train.shape[0])
    else:
        model.fit(data_train)
        data_gen = model.sample(num_rows=data_train.shape[0])
    return data_gen

def ctgan_synthesizer(data_train, pac=10, log_frequency=True, orig_implementation=False):
    if orig_implementation:
        ctgan_method = ctgan.CTGANSynthesizer
    else:
        ctgan_method = CTGAN

    model = ctgan_method(epochs=N_EPOCHS, batch_size=BATCH_SIZE, discriminator_steps=N_CRITICS, verbose=0,
                         cuda=True, embedding_dim=128, generator_dim=(256, 256),
                         discriminator_dim=(256, 256), pac=pac, log_frequency=log_frequency
                         )
    if orig_implementation:
        model.fit(data_train, discrete_columns=data_train.select_dtypes(exclude=[np.number]).columns.values)
        data_gen = model.sample(data_train.shape[0])
    else:
        model.fit(data_train)
        data_gen = model.sample(num_rows=data_train.shape[0])
    return data_gen

def CopulaGAN_synthesizer(data_train, log_frequency=True):
    model = CopulaGAN(epochs=N_EPOCHS, batch_size=BATCH_SIZE, discriminator_steps=N_CRITICS, verbose=0,
                         cuda=True, embedding_dim=128, generator_dim=(256, 256),
                         discriminator_dim=(256, 256), log_frequency=log_frequency
                      )
    model.fit(data_train)
    return model.sample(num_rows=data_train.shape[0])

def tabfairgan_synthesizer(data_train, modified=False):
    if modified:
        dim_hidden_layer = 256
        dim_latent_layer = 128
    else:
        dim_hidden_layer = -1 # same as setting equal to input dim
        dim_latent_layer = -1 # same as setting equal to input dim

    path_data_train_tabfairgan = os.path.join(const.dir.data_temp(),
                                              f"df_tabFairGAN_temp_{datetime.now().strftime('%H_%M_%S')}.csv")
    data_train.to_csv(path_data_train_tabfairgan)

    path_data_gen_tabfairgan = os.path.join(const.dir.data_temp(),
                                            f"df_gen_tabFairGAN_temp_{datetime.now().strftime('%H_%M_%S')}.csv")

    subprocess.call(
        f"python {os.path.join('src', 'TabFairGAN_nofair.py')} {path_data_train_tabfairgan} {N_EPOCHS}"
        f" {BATCH_SIZE} {path_data_gen_tabfairgan} --critic_repeat={N_CRITICS}"
        f" --dim_latent_layer={dim_latent_layer} --dim_hidden_layer={dim_hidden_layer}"
        f" --size_of_fake_data={data_train.shape[0]} --progress_bar={PROGRESS_BAR_MODEL_FIT}".split()
    )
    data_train_gen = pd.read_csv(path_data_gen_tabfairgan, index_col=0)

    if os.path.exists(path_data_train_tabfairgan):
        os.remove(path_data_train_tabfairgan)
    if os.path.exists(path_data_gen_tabfairgan):
        os.remove(path_data_gen_tabfairgan)

    return data_train_gen

def GaussianCopula_synthesizer(data_train):
    model = GaussianCopula()
    model.fit(data_train)
    return model.sample(num_rows=data_train.shape[0])

if "TVAE" in MODELS:
    synthesizer_name = "TVAE"
    helpers.comparison.synthesize_multiple_datasets(synthesizer=lambda data: tvae_synthesizer(data, modified=False, orig=False),
                                                    synthesizer_name=synthesizer_name,
                                                    overwrite_dataset=RESTART_ALL or synthesizer_name in RESTART_SPECIFIC,
                                                    **dict_default_arguments
                                                    )

if "TVAE-mod" in MODELS:
    synthesizer_name = "TVAE-mod"
    helpers.comparison.synthesize_multiple_datasets(synthesizer=lambda data: tvae_synthesizer(data, modified=True, orig=False),
                                                    synthesizer_name=synthesizer_name,
                                                    overwrite_dataset=RESTART_ALL or synthesizer_name in RESTART_SPECIFIC,
                                                    **dict_default_arguments
                                                    )

if "TVAESynthesizer" in MODELS:
    synthesizer_name = "TVAESynthesizer"
    helpers.comparison.synthesize_multiple_datasets(synthesizer=lambda data: tvae_synthesizer(data, modified=False, orig=True),
                                              synthesizer_name=synthesizer_name,
                                              overwrite_dataset=RESTART_ALL or synthesizer_name in RESTART_SPECIFIC,
                                              **dict_default_arguments
                                              )

if "TVAESynthesizer-mod" in MODELS:
    synthesizer_name = "TVAESynthesizer-mod"
    helpers.comparison.synthesize_multiple_datasets(synthesizer=lambda data: tvae_synthesizer(data, modified=True, orig=True),
                                              synthesizer_name=synthesizer_name,
                                              overwrite_dataset=RESTART_ALL or synthesizer_name in RESTART_SPECIFIC,
                                              **dict_default_arguments
                                              )

if "CTGAN-pac10" in MODELS:
    synthesizer_name = "CTGAN-pac10"
    pac = int(re.search( "pac(\d+)", synthesizer_name).group(0).replace("pac", ""))
    helpers.comparison.synthesize_multiple_datasets(
        synthesizer=lambda data: ctgan_synthesizer(data, pac=pac, log_frequency=True,
                                                   orig_implementation=False
                                                   ),
                                                    synthesizer_name=synthesizer_name,
                                                    overwrite_dataset=RESTART_ALL or synthesizer_name in RESTART_SPECIFIC,
                                                    **dict_default_arguments
                                                )

if "CTGAN-pac1" in MODELS:
    synthesizer_name = "CTGAN-pac1"
    pac = int(re.search( "pac(\d+)", synthesizer_name).group(0).replace("pac", ""))
    helpers.comparison.synthesize_multiple_datasets(
        synthesizer=lambda data: ctgan_synthesizer(data, pac=pac, log_frequency=True,
                                                   orig_implementation=False
                                                   ),
        synthesizer_name=synthesizer_name,
        overwrite_dataset=RESTART_ALL or synthesizer_name in RESTART_SPECIFIC,
        **dict_default_arguments
    )

if "CTGANSynthesizer-pac10" in MODELS:
    synthesizer_name = "CTGANSynthesizer-pac10"
    pac = int(re.search( "pac(\d+)", synthesizer_name).group(0).replace("pac", ""))
    helpers.comparison.synthesize_multiple_datasets(
        synthesizer=lambda data: ctgan_synthesizer(data, pac=pac, log_frequency=True,
                                                   orig_implementation=True
                                                   ),
        synthesizer_name=synthesizer_name,
        overwrite_dataset=RESTART_ALL or synthesizer_name in RESTART_SPECIFIC,
        **dict_default_arguments
    )

if "CTGANSynthesizer-pac1" in MODELS:
    synthesizer_name = "CTGANSynthesizer-pac1"
    pac = int(re.search( "pac(\d+)", synthesizer_name).group(0).replace("pac", ""))
    helpers.comparison.synthesize_multiple_datasets(
        synthesizer=lambda data: ctgan_synthesizer(data, pac=pac, log_frequency=True,
                                                   orig_implementation=True
                                                   ),
        synthesizer_name=synthesizer_name,
        overwrite_dataset=RESTART_ALL or synthesizer_name in RESTART_SPECIFIC,
        **dict_default_arguments
    )

if "tabFairGAN" in MODELS:
    synthesizer_name = "tabFairGAN"
    helpers.comparison.synthesize_multiple_datasets(
        synthesizer=lambda data: tabfairgan_synthesizer(data, modified=False),
        synthesizer_name=synthesizer_name,
        overwrite_dataset=RESTART_ALL or synthesizer_name in RESTART_SPECIFIC,
        **dict_default_arguments
    )

if "tabFairGAN-mod" in MODELS:
    synthesizer_name = "tabFairGAN-mod"
    helpers.comparison.synthesize_multiple_datasets(
        synthesizer=lambda data: tabfairgan_synthesizer(data, modified=True),
        synthesizer_name=synthesizer_name,
        overwrite_dataset=RESTART_ALL or synthesizer_name in RESTART_SPECIFIC,
        **dict_default_arguments
    )

if "GaussianCopula" in MODELS:
    synthesizer_name = "GaussianCopula"
    helpers.comparison.synthesize_multiple_datasets(synthesizer=GaussianCopula_synthesizer,
                                              synthesizer_name=synthesizer_name,
                                              overwrite_dataset=RESTART_ALL or synthesizer_name in RESTART_SPECIFIC,
                                              **dict_default_arguments
                                              )

if "CopulaGAN" in MODELS:
    synthesizer_name = "CopulaGAN"
    helpers.comparison.synthesize_multiple_datasets(synthesizer=CopulaGAN_synthesizer,
                                              synthesizer_name=synthesizer_name,
                                              overwrite_dataset=RESTART_ALL or synthesizer_name in RESTART_SPECIFIC,
                                              **dict_default_arguments
                                              )