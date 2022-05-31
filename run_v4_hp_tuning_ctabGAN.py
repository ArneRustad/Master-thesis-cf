from v2_hp_tuning import fetch_hp_info
from src import constants as const
import helpers
import os

PROGRESS_BAR_SUBSUBPROCESS = False
METHOD_NAME = "ctabGAN-qtr"
VERSION = 4


hp_info = fetch_hp_info(method=METHOD_NAME, version=VERSION)
hp_name_vec = ["qtr_spread",
               #"noise_ctgan",
               "gumbel_temperature", #"add_connection", "oh_encoding_choices",
               "activation_function", "wgan_penalty_query", "add_connection_advanced", #Count 5
               "qt_transformation", "oh_encoding_activation_function",
               #"reapply_qtr_continuously",
               "spread_and_activation", "spread_and_activations", "adam_beta1" #Count 10
               ]
hp_name_restart_vec = []

slurm_array_task_id = os.getenv('SLURM_ARRAY_TASK_ID')
if slurm_array_task_id is not None:
    slurm_array_task_id = int(slurm_array_task_id)
    hp_name_vec = hp_name_vec[(slurm_array_task_id-1):slurm_array_task_id]
    print(f"Starting slurm array task {slurm_array_task_id}: hyperparameter tuning for {hp_name_vec[0]}")

for hp_name in hp_name_vec:
    print("-"*10 + f"Hyperparameter tuning: {hp_name}" + "-"*10)
    curr_hp_info = hp_info[hp_name]

    helpers.hp_tuning.generate_multiple_datasets_for_multiple_hyperparameters(
        create_tabGAN_func=curr_hp_info["tabGAN_func"],
        hyperparams_vec=curr_hp_info["vec"],
        n_epochs=curr_hp_info["n_epochs"],
        dataset_dir=const.dir.hp_tuning_v4(),
        batch_size=curr_hp_info["batch_size"],
        subfolder=METHOD_NAME,
        n_synthetic_datasets=curr_hp_info["n_synthetic_datasets"],
        hyperparams_name=hp_name,
        hyperparams_subname=curr_hp_info["hyperparams_subname"],
        add_comparison_folder=True,
        overwrite_dataset=hp_name in hp_name_restart_vec,
        progress_bar_subprocess=True,
        progress_bar_subsubprocess=PROGRESS_BAR_SUBSUBPROCESS,
    )
