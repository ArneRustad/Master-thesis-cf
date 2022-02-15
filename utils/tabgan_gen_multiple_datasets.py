import os
from tqdm.auto import tqdm

def generate_multiple_datasets(tabgan, dataset_dir, n_synthetic_datasets, n_epochs, batch_size,
                               subfolder = None, n_synthetic_datasets_existing = 0,
                               progress_bar_leave = True, overwrite_dataset=True,
                               progress_bar=True, progress_bar_dataset=True):
    if not subfolder is None:
        dataset_dir = os.path.join(dataset_dir, subfolder)
    os.makedirs(dataset_dir, exist_ok = True)
    with tqdm(total=n_synthetic_datasets - n_synthetic_datasets_existing,
              desc="Generated datasets", leave = progress_bar_leave,
              disable=not progress_bar) as pbar:
        for i in range(n_synthetic_datasets_existing, n_synthetic_datasets):
            current_path = os.path.join(dataset_dir, f"gen{i}.csv")
            if overwrite_dataset or not os.path.exists(current_path):
                tabgan.train(n_epochs, batch_size=batch_size, restart_training=True,
                             plot_loss=False, progress_bar=progress_bar_dataset,
                             progress_bar_desc=f"Progress generating dataset {i+1}")
                fake_train = tabgan.generate_data()
                fake_train.to_csv(current_path)
            else:
                pass
            pbar.update(1)
            pbar.refresh()