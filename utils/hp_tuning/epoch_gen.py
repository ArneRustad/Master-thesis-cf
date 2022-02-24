def generate_multiple_datasets_for_multiple_epochs(tabGAN, dataset_dir, n_synthetic_datasets, n_epochs_vec,
                                                   restart = False, path_finished_epochs_counter = None,
                                                   redo_n_epochs_vec = [], plot_only_new_progress = True,
                                                   progress_bar=True, progress_bar_subprocess=True,
                                                   progress_bar_subsubprocess=None,
                                                   **kwargs):
    if progress_bar_subsubprocess is None:
        progress_bar_subsubprocess=progress_bar_subprocess
    n_epochs_vec = set(n_epochs_vec)
    redo_n_epochs_vec = set(redo_n_epochs_vec)
    if restart or (path_finished_epochs_counter is None) or (not os.path.exists(path_finished_epochs_counter)):
        existing_n_epochs = set()
        if not path_finished_epochs_counter is None:
            os.makedirs(os.path.dirname(path_finished_epochs_counter), exist_ok = True)
    else:
        with open(path_finished_epochs_counter, 'rb') as handle:
            existing_n_epochs = pickle.load(handle)
    n_epochs_new_vec = n_epochs_vec.difference(existing_n_epochs.difference(redo_n_epochs_vec))

    with tqdm(total = len(n_epochs_new_vec) if plot_only_new_progress else len(n_epochs_vec),
              desc = "Epoch subfolder creation", disable=not progress_bar) as pbar:
        if not plot_only_new_progress:
            pbar.update(len(n_epochs_vec) - len(n_epochs_new_vec))
        for i, n_epochs in enumerate(n_epochs_vec):
            if n_epochs in existing_n_epochs and not n_epochs in redo_n_epochs_vec:
                continue
            generate_multiple_datasets(tabGAN, dataset_dir, n_synthetic_datasets, n_epochs = n_epochs,
                                       subfolder = f"Epochs{n_epochs}", progress_bar_leave = False,
                                       progress_bar=progress_bar_subprocess,
                                       progress_bar_dataset=progress_bar_subsubprocess)
            pbar.update(1)
            if not path_finished_epochs_counter is None:
                existing_n_epochs.add(n_epochs)
                with open(path_finished_epochs_counter, 'wb') as handle:
                    pickle.dump(existing_n_epochs, handle)