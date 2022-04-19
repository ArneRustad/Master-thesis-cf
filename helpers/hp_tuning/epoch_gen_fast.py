import pickle
from tqdm.auto import tqdm
import os
import numpy as np

def generate_multiple_datasets_for_multiple_epochs_fast(tabGAN, dataset_dir, n_epochs_vec,
                                                        batch_size, n_synthetic_datasets,
                                                        redo_n_epochs_vec = [],
                                                        subfolder=None,
                                                        overwrite_dataset=False,
                                                        **kwargs):
    if subfolder is not None:
        dataset_dir = os.path.join(dataset_dir, subfolder)
    dataset_dir = os.path.join(dataset_dir, "n_epochs_comparison")
    n_epochs_vec = set(n_epochs_vec)
    n_epochs_vec.update(redo_n_epochs_vec)
    n_epochs_vec = np.sort(list(n_epochs_vec)).astype(np.int)
    print(n_epochs_vec)

    for j in tqdm(range(n_synthetic_datasets), desc = "Generated datasets", leave = True):
        last_n_epochs = 0
        with tqdm(total=len(n_epochs_vec), desc = "Epoch subfolder creation", leave=False) as pbar:
            for i, n_epochs in enumerate(n_epochs_vec):
                epoch_dataset_dir = os.path.join(dataset_dir, f"Epochs{n_epochs}")
                epoch_dataset_path = os.path.join(epoch_dataset_dir, f"gen{j}.csv")

                if overwrite_dataset or not os.path.exists(epoch_dataset_path):
                    if j == 0:
                        os.makedirs(epoch_dataset_dir, exist_ok = True)
                    if i == 0:
                        restart_training = True
                        n_epochs_diff = n_epochs
                    else:
                        restart_training = False
                        n_epochs_diff = n_epochs - last_n_epochs
                    tabGAN.train(n_epochs_diff, batch_size = batch_size, restart_training = restart_training, plot_loss = False,
                                 progress_bar = True, progress_bar_desc = f"Progress training from epoch {last_n_epochs} to {n_epochs}")
                    fake_train = tabGAN.sample()
                    fake_train.to_csv(epoch_dataset_path)
                    last_n_epochs = n_epochs
                pbar.update(1)