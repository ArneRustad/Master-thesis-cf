import os
from tqdm.auto import tqdm
from helpers.tabgan_gen_multiple_datasets import generate_multiple_datasets
import pickle

def generate_multiple_datasets_for_multiple_hyperparameters(create_tabGAN_func, hyperparams_vec, n_epochs, dataset_dir,
                                                            batch_size,
                                                            n_synthetic_datasets, restart=False,
                                                            redo_hyperparams_vec=[],
                                                            hyperparams_name="hyperparam",
                                                            hyperparams_subname=None,
                                                            subfolder=None,
                                                            add_comparison_folder=True,
                                                            overwrite_dataset=True,
                                                            progress_bar=True,
                                                            progress_bar_subprocess=True,
                                                            progress_bar_subsubprocess=None):
    if progress_bar_subsubprocess is None:
        progress_bar_subsubprocess=progress_bar_subprocess
    if subfolder is not None:
        dataset_dir = os.path.join(dataset_dir, subfolder)
    if add_comparison_folder:
        dataset_dir = os.path.join(dataset_dir, f"{hyperparams_name}_comparison")

    with tqdm(total=len(hyperparams_vec),
              desc="Hyperparameters subfolder creation", disable=not progress_bar) as pbar:
        for i, hyperparams in enumerate(hyperparams_vec):
            if isinstance(hyperparams, (list, tuple)):
                if not hyperparams_subname is None:
                    if len(hyperparams_subname) != len(hyperparams):
                        raise ValueError("Length of hyperparams_subname vector must either be of equal length to hyperparams vector or hyperparams_subname must be equal to None")
                    hyperparams_abbreviation = "".join("_" + str(n) + "_" + str(s) for n, s in zip(hyperparams_subname, hyperparams))
                else:
                    hyperparams_abbreviation = "".join("_" + str(s) for s in hyperparams)
                tabGAN = create_tabGAN_func(*hyperparams)
            else:
                hyperparams_abbreviation = "_" + str(hyperparams)
                tabGAN = create_tabGAN_func(hyperparams)

            generate_multiple_datasets(tabGAN, dataset_dir, n_synthetic_datasets, n_epochs=n_epochs,
                                       batch_size=batch_size,
                                       subfolder="{}{}".format(hyperparams_name, hyperparams_abbreviation),
                                       progress_bar_leave=False,
                                       overwrite_dataset=overwrite_dataset,
                                       progress_bar=progress_bar_subprocess,
                                       progress_bar_dataset=progress_bar_subsubprocess)
            pbar.update(1)
