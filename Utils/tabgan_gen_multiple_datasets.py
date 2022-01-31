def generate_multiple_datasets(tabgan, dataset_dir, n_synthetic_datasets, n_epochs, subfolder = None,
                               n_synthetic_datasets_existing = 0, progress_bar_leave = True):
    if not subfolder is None:
        dataset_dir = os.path.join(dataset_dir, subfolder)
    os.makedirs(dataset_dir, exist_ok = True)
    for i in tqdm(range(n_synthetic_datasets_existing, n_synthetic_datasets), desc = "Generated datasets",
                  leave = progress_bar_leave):

        tabgan.train(n_epochs, batch_size = batch_size, restart_training = True, plot_loss = False,
                     progress_bar = True, progress_bar_desc = f"Progress generating dataset {i+1}")
        fake_train = tabgan.generate_data()
        fake_train.to_csv(os.path.join(dataset_dir, f"gen{i}.csv"))