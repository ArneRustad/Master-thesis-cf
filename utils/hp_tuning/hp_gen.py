import os
from tqdm.auto import tqdm
from ..tabgan_gen_multiple_datasets import generate_multiple_datasets
import pickle

def generate_multiple_datasets_for_multiple_hyperparameters(create_tabGAN_func, hyperparams_vec, n_epochs, dataset_dir,
                                                            batch_size,
                                                            n_synthetic_datasets, restart=False,
                                                            redo_hyperparams_vec=[], plot_only_new_progress=True,
                                                            hyperparams_name="hyperparam",
                                                            hyperparams_subname=None,
                                                            subfolder=None,
                                                            tracker_name=None,
                                                            tracker_dir=None,
                                                            add_comparison_folder=True,
                                                            overwrite_dataset=True,
                                                            progress_bar=True,
                                                            progress_bar_subprocess=True,
                                                            progress_bar_subsubprocess=None):
    if progress_bar_subsubprocess is None:
        progress_bar_subsubprocess=progress_bar_subprocess
    if subfolder is not None:
        dataset_dir = os.path.join(dataset_dir, subfolder)
    if tracker_dir is None:
        tracker_dir = os.path.join(dataset_dir, "_tracker_objects")
    if tracker_name is None:
        tracker_name = f"existing_{hyperparams_name}_tracker.pkl"
    if add_comparison_folder:
        dataset_dir = os.path.join(dataset_dir, f"{hyperparams_name}_comparison")
    hyperparams_vec = set(hyperparams_vec)
    redo_hyperparams_vec = set(redo_hyperparams_vec)
    path_finished_hyperparams = os.path.join(tracker_dir, tracker_name)
    if restart or (path_finished_hyperparams is None) or (not os.path.exists(path_finished_hyperparams)):
        existing_hyperparams = set()
        if not path_finished_hyperparams is None:
            os.makedirs(os.path.dirname(path_finished_hyperparams), exist_ok=True)
    else:
        with open(path_finished_hyperparams, 'rb') as handle:
            existing_hyperparams = pickle.load(handle)
    hyperparams_new_vec = hyperparams_vec.difference(existing_hyperparams.difference(redo_hyperparams_vec))

    with tqdm(total=len(hyperparams_new_vec) if plot_only_new_progress else len(hyperparams_vec),
              desc="Hyperparameters subfolder creation", disable=not progress_bar) as pbar:
        if not plot_only_new_progress:
            pbar.update(len(hyperparams_vec) - len(hyperparams_new_vec))
        for i, hyperparams in enumerate(hyperparams_new_vec):
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
            if not path_finished_hyperparams is None:
                existing_hyperparams.add(hyperparams)
                with open(path_finished_hyperparams, 'wb') as handle:
                    pickle.dump(existing_hyperparams, handle)
