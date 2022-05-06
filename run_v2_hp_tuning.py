from v2_hp_tuning import hp_info
from src import constants as const
import helpers

PROGRESS_BAR_SUBSUBPROCESS = False

hp_name_vec = ["qtr_spread", "oh_encoding_choices"]
hp_name_restart_vec = []

for hp_name in hp_name_vec:
    print(hp_name)
    curr_hp_info = hp_info[hp_name]

    helpers.hp_tuning.generate_multiple_datasets_for_multiple_hyperparameters(
        create_tabGAN_func=curr_hp_info["tabGAN_func"],
        hyperparams_vec=curr_hp_info["vec"],
        n_epochs=curr_hp_info["n_epochs"],
        dataset_dir=const.dir.hp_tuning_v2(),
        batch_size=curr_hp_info["batch_size"],
        subfolder="tabGAN-qtr",
        n_synthetic_datasets=curr_hp_info["n_synthetic_datasets"],
        hyperparams_name=hp_name,
        hyperparams_subname=curr_hp_info["hyperparams_subname"],
        add_comparison_folder=True,
        overwrite_dataset=hp_name in hp_name_restart_vec,
        progress_bar_subprocess=PROGRESS_BAR_SUBSUBPROCESS,
    )
