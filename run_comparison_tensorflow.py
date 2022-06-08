from src import constants as const
from utils import print_header_method
import helpers.comparison

import pandas as pd
import numpy as np
import os
import re
from tqdm.auto import tqdm
from tabGAN import TabGAN
from datetime import datetime

N_EPOCHS = 300
BATCH_SIZE = 500
N_CRITICS = 10
N_SYNTHETIC_DATASETS = 1

RESTART_ALL = False
RESTART_SPECIFIC = []
PROGRESS_BAR_TASK = True
PROGRESS_BAR_MODEL_FIT = False
DATASET_TASKS = ["covtype_edited", "creditcard_edited", "news_edited"]
MODELS = ["tabGAN", "tabGAN-qt", "tabGAN-qtr", "ctabGAN", "ctabGAN-qt", "ctabGAN-qtr"]

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
                N_SYNTHETIC_DATASETS = 10
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
    "progress_bar_task": PROGRESS_BAR_TASK}


def tabgan_synthesizer(data_train, qt=False, qtr=False, ctgan=False, pac=1):
    if qtr and not qt:
        raise ValueError("qtr parameter can not be set to True when qt parameter is set to False")
    tg = TabGAN(data_train,
                n_critic=N_CRITIC,
                ctgan=ctgan,
                pac=pac,
                optimizer="adam",
                opt_lr=0.0002,
                qtr_spread=1,
                adam_beta1=0.7,
                adam_beta2=0.999,
                qtr_lbound_apply=0.05,
                jit_compile=True,
                batch_size=BATCH_SIZE,
                wgan_lambda=10,
                quantile_transformation_int=qt,
                quantile_rand_transformation=qtr,
                noise_discrete_unif_max=(0 if ctgan else 0.01),
                gumbel_temperature=0.5 if ctgan else 0.1,
                activation_function="Mish",
                max_quantile_share=1,
                n_quantiles_int=1000,
                qt_n_subsample=1e5,
                dim_hidden=256,
                dim_latent=128,
                tf_data_use=(not ctgan),
                ctgan_binomial_loss=True,
                ctgan_log_frequency=True,
                train_step_critic_same_queries_for_critic_and_gen=False,
                train_step_critic_wgan_penalty_query_diversity=False,
                train_step_critic_query_wgan_penalty=True,
                critic_use_query_input=True
                )
    tg.train(n_epochs=N_EPOCHS, batch_size=BATCH_SIZE, progress_bar=PROGRESS_BAR_MODEL_FIT)


if "tabGAN" in MODELS:
    synthesizer_name = "tabGAN"
    pac = int(re.search( "pac(\d+)", synthesizer_name).group(0).replace("pac", ""))
    helpers.comparison.synthesize_multiple_datasets(
        synthesizer=lambda data: tabGAN_synthesizer(data,
                                                    qt="-qt" in synthesizer_name,
                                                    qtr="-qtr" in synthesizer_name,
                                                    ctgan="ctabGAN" in synthesizer_name,
                                                    pac=pac),
        synthesizer_name=synthesizer_name,
        overwrite_dataset=RESTART_ALL or synthesizer_name in RESTART_SPECIFIC,
        **dict_default_arguments
    )

if "tabGAN-qt" in MODELS:
    synthesizer_name = "tabGAN-qt"
    pac = int(re.search( "pac(\d+)", synthesizer_name).group(0).replace("pac", ""))
    helpers.comparison.synthesize_multiple_datasets(
        synthesizer=lambda data: tabGAN_synthesizer(data,
                                                    qt="-qt" in synthesizer_name,
                                                    qtr="-qtr" in synthesizer_name,
                                                    ctgan="ctabGAN" in synthesizer_name,
                                                    pac=pac),
        synthesizer_name=synthesizer_name,
        overwrite_dataset=RESTART_ALL or synthesizer_name in RESTART_SPECIFIC,
        **dict_default_arguments
    )

if "tabGAN-qtr" in MODELS:
    synthesizer_name = "tabGAN-qtr"
    pac = int(re.search( "pac(\d+)", synthesizer_name).group(0).replace("pac", ""))
    helpers.comparison.synthesize_multiple_datasets(
        synthesizer=lambda data: tabGAN_synthesizer(data,
                                                    qt="-qt" in synthesizer_name,
                                                    qtr="-qtr" in synthesizer_name,
                                                    ctgan="ctabGAN" in synthesizer_name,
                                                    pac=pac),
        synthesizer_name=synthesizer_name,
        overwrite_dataset=RESTART_ALL or synthesizer_name in RESTART_SPECIFIC,
        **dict_default_arguments
    )

if "ctabGAN" in MODELS:
    synthesizer_name = "ctabGAN"
    pac = int(re.search( "pac(\d+)", synthesizer_name).group(0).replace("pac", ""))
    helpers.comparison.synthesize_multiple_datasets(
        synthesizer=lambda data: tabGAN_synthesizer(data,
                                                    qt="-qt" in synthesizer_name,
                                                    qtr="-qtr" in synthesizer_name,
                                                    ctgan="ctabGAN" in synthesizer_name,
                                                    pac=pac),
        synthesizer_name=synthesizer_name,
        overwrite_dataset=RESTART_ALL or synthesizer_name in RESTART_SPECIFIC,
        **dict_default_arguments
    )

if "ctabGAN-qt" in MODELS:
    synthesizer_name = "ctabGAN-qt"
    pac = int(re.search( "pac(\d+)", synthesizer_name).group(0).replace("pac", ""))
    helpers.comparison.synthesize_multiple_datasets(
        synthesizer=lambda data: tabGAN_synthesizer(data,
                                                    qt="-qt" in synthesizer_name,
                                                    qtr="-qtr" in synthesizer_name,
                                                    ctgan="ctabGAN" in synthesizer_name,
                                                    pac=pac),
        synthesizer_name=synthesizer_name,
        overwrite_dataset=RESTART_ALL or synthesizer_name in RESTART_SPECIFIC,
        **dict_default_arguments
    )

if "ctabGAN-qtr" in MODELS:
    synthesizer_name = "ctabGAN-qtr"
    pac = int(re.search( "pac(\d+)", synthesizer_name).group(0).replace("pac", ""))
    helpers.comparison.synthesize_multiple_datasets(
        synthesizer=lambda data: tabGAN_synthesizer(data,
                                                    qt="-qt" in synthesizer_name,
                                                    qtr="-qtr" in synthesizer_name,
                                                    ctgan="ctabGAN" in synthesizer_name,
                                                    pac=pac),
        synthesizer_name=synthesizer_name,
        overwrite_dataset=RESTART_ALL or synthesizer_name in RESTART_SPECIFIC,
        **dict_default_arguments
    )