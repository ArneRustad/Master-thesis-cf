import os
from tqdm.auto import tqdm
from src import constants as const
import numpy as np

def generate_bool_indices_for_dataset_tasks(datasets=["covtype_edited", "creditcard_edited", "news_edited"],
                                            n_synthetic_datasets=10,
                                            dataset_dir=const.dir.data_comparison(),
                                            overwrite_indices=False,
                                            progress_bar_task=True,
                                            progress_bar_each_task=True, progress_bar_leave=True,
                                            train_size=0.7):
    with tqdm(total=len(datasets), desc=f"Dataset tasks", leave=progress_bar_leave, disable=not progress_bar_task) \
            as pbar_tasks:
        for dataset_task in datasets:
            curr_dataset = pd.read_csv(os.path.join(dataset_dir, dataset_task + ".csv"))
            curr_dir_dataset_train_indices = os.path.join(dataset_dir, "indices", dataset_task)
            os.makedirs(curr_dir_dataset_train_indices)

            n_train_samples = curr_dataset.shape[0] * train_size
            indices_bool_template = np.repeat([True, False], [n_train_samples, curr_dataset.shape[0] - n_train_samples])

            with tqdm(total=n_synthetic_datasets, desc=f"Train indices generated({synthesizer_name})",
                      leave=progress_bar_leave,
                      disable=not progress_bar_each_task) as pbar_each_task:
                for i in range(n_synthetic_datasets):
                    curr_train_bool_indices = np.load()
                    if overwrite_indices or not os.path.exists(curr_train_bool_indices):
                        curr_train_bool_indices = np.random.shuffle(indices_bool_template)
                        np.save(os.path.join(curr_dir_dataset_train_indices, f"bool_indices_{i}.npy"),
                                curr_train_bool_indices)
                        pbar_each_task.update(1)
                    else:
                        pbar.update(1)
                        pbar_each_task.refresh()
            pbar_tasks.update(1)
            pbar_tasks.refresh()
