import pickle
from tqdm.auto import tqdm
import os

def generate_multiple_datasets_for_multiple_epochs_fast(tabGAN, dataset_dir, n_epochs_vec,
                                                        batch_size, n_synthetic_datasets,
                                                        restart = False, path_finished_epochs_counter = None,
                                                        redo_n_epochs_vec = [], plot_only_new_progress = True,
                                                        n_synthetic_datasets_existing = 0,
                                                        subfolder=None, tracker_name=None,
                                                        **kwargs):
    if not subfolder is None:
        dataset_dir = os.path.join(dataset_dir, subfolder)
    if path_finished_epochs_counter is None:
        path_finished_epochs_counter = os.path.join(dataset_dir, "_tracker_objects/")
        if tracker_name is None:
            path_finished_epochs_counter = os.path.join(path_finished_epochs_counter,
                                                        "existing_n_epochs_tracker.pkl")
        else:
            path_finished_epochs_counter = os.path.join(path_finished_epochs_counter,
                                                        tracker_name)
    n_epochs_vec = set(n_epochs_vec)
    redo_n_epochs_vec = set(redo_n_epochs_vec)
    if restart or (not os.path.exists(path_finished_epochs_counter)):
        existing_n_epochs = set()
        if not path_finished_epochs_counter is None:
            os.makedirs(os.path.dirname(path_finished_epochs_counter), exist_ok = True)
    else:
        with open(path_finished_epochs_counter, 'rb') as handle:
            existing_n_epochs = pickle.load(handle)
    n_epochs_new_vec = n_epochs_vec.difference(existing_n_epochs.difference(redo_n_epochs_vec))
    if len(n_epochs_new_vec) == 0:
        print("All datasets for all epochs are already generated.")
        return

    for j in tqdm(range(n_synthetic_datasets_existing, n_synthetic_datasets), desc = "Generated datasets",
                  leave = True):
        with tqdm(total = len(n_epochs_new_vec) if plot_only_new_progress else len(n_epochs_vec),
                  desc = "Epoch subfolder creation", leave=False) as pbar:
            if not plot_only_new_progress:
                pbar.update(len(n_epochs_vec) - len(n_epochs_new_vec))
            for i, n_epochs in enumerate(n_epochs_new_vec):
                epoch_dataset_dir = os.path.join(dataset_dir, f"Epochs{n_epochs}")
                if j == 0:
                    os.makedirs(epoch_dataset_dir, exist_ok = True)
                if i == 0:
                    restart_training = True
                    n_epochs_diff = n_epochs
                    last_n_epochs = 0
                else:
                    restart_training = False
                    n_epochs_diff = n_epochs - last_n_epochs
                tabGAN.train(n_epochs_diff, batch_size = batch_size, restart_training = restart_training, plot_loss = False,
                             progress_bar = True, progress_bar_desc = f"Progress training from epoch {last_n_epochs} to {n_epochs}")
                fake_train = tabGAN.sample()
                fake_train.to_csv(os.path.join(epoch_dataset_dir, f"gen{j}.csv"))
                pbar.update(1)
                if j == n_synthetic_datasets:
                    if not path_finished_epochs_counter is None:
                        existing_n_epochs.add(n_epochs)
                        with open(path_finished_epochs_counter, 'wb') as handle:
                            pickle.dump(existing_n_epochs, handle)
                last_n_epochs = n_epochs