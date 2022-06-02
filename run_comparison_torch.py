from src import constants as const
from utils import print_header_method
import helpers

import pandas as pd
import numpy as np
import os
import subprocess
from tqdm.auto import tqdm
import torch
from sdv.tabular import CTGAN, TVAE, GaussianCopula
import ctgan

N_EPOCHS = 300
BATCH_SIZE = 500
N_CRITICS = 10
N_SYNTHETIC_DATASETS = 1

RESTART_ALL = False
RESTART_SPECIFIC = []
PROGRESS_BAR_TASK = True
DATASET_TASKS = ["covtype_edited", "creditcard_edited", "news_edited"]

MODELS = ["TVAE", "TVAE-mod", "CTGAN-pac10", "CTGAN-pac1"]
dict_default_arguments = {
    "datasets": DATASET_TASKS,
    "n_synthetic_datasets":N_SYNTHETIC_DATASETS,
    "progress_bar_task":PROGRESS_BAR_TASK}

slurm_array_task_id = os.getenv('SLURM_ARRAY_TASK_ID')
if slurm_array_task_id is not None:
    slurm_array_task_id = int(slurm_array_task_id)
    hp_name_vec = hp_name_vec[(slurm_array_task_id-1):slurm_array_task_id]
    dataset_id = slurm_array_task_id // 100
    model_id = slurm_array_task_id % 100
    if dataset_id > 0:
        if dataset_id <= len(DATASET_TASKS):
            DATASET_TASKS = DATASET_TASKS[dataset_id]
        else:
            raise ValueError(f"dataset_id can't be larger than length of DATASET_TASKS. You entered {dataset_id}.")
    if model_id > 0:
        if model_id <= len(MODELS):
            MODELS = MODELS[model_id]
        else:
            raise ValueError(f"model_id can't be larger than length of MODELS. You entered {model_id}.")

    print(f"Starting comparison array task with dataset_id {dataset_id} and model_id {model_id}")

def tvae_synthesizer(data_train, modified=False):
    if modified:
        hidden_dim = 256
    else:
        hidden_dim = 128

    model = TVAE(epochs=N_EPOCHS, batch_size=BATCH_SIZE, cuda=True,
                             embedding_dim=128, compress_dims=(hidden_dim, hidden_dim),
                             decompress_dims=(hidden_dim, hidden_dim),
                             loss_factor=2
                             )
    model.fit(data_train)
    return model.sample(num_rows=data_train.shape[0])

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

if "TVAE" in models:
    synthesizer_name = "TVAE"
    helpers.comparison.synthesize_multiple_datasets(synthesizer=lambda data: tvae_synthesizer(data, modified=False),
                                                    synthesizer_name=synthesizer_name,
                                                    overwrite_dataset=RESTART_ALL or synthesizer_name in RESTART_SPECIFIC,
                                                    **dict_default_arguments
                                                    )

if "TVAE-mod" in models:
    synthesizer_name = "TVAE-mod"
    helpers.comparison.synthesize_multiple_datasets(synthesizer=lambda data: tvae_synthesizer(data, modified=True),
                                                    synthesizer_name=synthesizer_name,
                                                    overwrite_dataset=RESTART_ALL or synthesizer_name in RESTART_SPECIFIC,
                                                    **dict_default_arguments
                                                    )

if "CTGAN-pac10" in models:
    synthesizer_name = "CTGAN-pac10"
    pac = int(re.search( "pac(\d+)", synthesizer_name).group(0).replace("pac", ""))
    helpers.comparison.synthesize_multiple_datasets(
        synthesizer=lambda data: ctgan_synthesizer(data, pac=pac, log_frequency=True,
                                                   orig_implementation="-orig" in synthesizer_name
                                                   ),
                                                    synthesizer_name=synthesizer_name,
                                                    overwrite_dataset=RESTART_ALL or synthesizer_name in RESTART_SPECIFIC,
                                                    **dict_default_arguments
                                                )

if "CTGAN-pac1" in models:
    synthesizer_name = "CTGAN-pac1"
    pac = int(re.search( "pac(\d+)", synthesizer_name).group(0).replace("pac", ""))
    helpers.comparison.synthesize_multiple_datasets(
        synthesizer=lambda data: ctgan_synthesizer(data, pac=pac, log_frequency=True,
                                                   orig_implementation="-orig" in synthesizer_name
                                                   ),
        synthesizer_name=synthesizer_name,
        overwrite_dataset=RESTART_ALL or synthesizer_name in RESTART_SPECIFIC,
        **dict_default_arguments
    )