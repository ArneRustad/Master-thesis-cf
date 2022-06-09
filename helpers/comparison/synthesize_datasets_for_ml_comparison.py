import os
from tqdm.auto import tqdm
from src import constants as const
import numpy as np
import pandas as pd


def generate_multiple_datasets_for_comparison(synthesizer, synthesizer_name,
                                              datasets=["covtype_edited", "creditcard_edited", "news_edited"],
                                              n_synthetic_datasets=5,
                                              dataset_dir=const.dir.data_comparison(),
                                              gen_dataset_dir=const.dir.data_comparison_gen(),
                                              overwrite_dataset=False,
                                              progress_bar_task=True,
                                              progress_bar_each_task=True, progress_bar_leave=True,
                                              _specific_dataset_number=None,
                                              ):
    dir_dataset_gen_model = os.path.join(gen_dataset_dir, synthesizer_name)
    os.makedirs(dir_dataset_gen_model, exist_ok=True)
    with tqdm(total=len(datasets), desc=f"Dataset tasks", leave=progress_bar_leave, disable=not progress_bar_task) \
            as pbar_tasks:
        for dataset_task in datasets:
            curr_dataset = pd.read_csv(os.path.join(dataset_dir, dataset_task + ".csv"))
            curr_dir_dataset_train_indices = os.path.join(dataset_dir, "indices", dataset_task)
            if not os.path.exists(curr_dir_dataset_train_indices):
                raise ValueError(f"The train indices directory {curr_dir_dataset_train_indices} does not exist")
            current_gen_dir = os.path.join(dir_dataset_gen_model, dataset_task)
            os.makedirs(current_gen_dir, exist_ok=True)

            with tqdm(total=n_synthetic_datasets, desc=f"Synthesized datasets ({synthesizer_name})",
                      leave=False,
                      disable=not progress_bar_each_task) as pbar_each_task:
                if _specific_dataset_number is None:
                    range_synthetic_datasets = range(n_synthetic_datasets)
                else:
                    range_synthetic_datasets = [_specific_dataset_number]
                    print(f"Generating only synthetic dataset {_specific_dataset_number} for model {synthesizer_name} "
                          f"and dataset task {dataset_task}")
                for i in range_synthetic_datasets:
                    current_gen_path = os.path.join(current_gen_dir, f"gen{i}.csv")
                    if overwrite_dataset or not os.path.exists(current_gen_path):
                        curr_train_bool_indices = np.load(os.path.join(curr_dir_dataset_train_indices,
                                                                       f"bool_indices_{i}.npy"))
                        curr_train_dataset = curr_dataset.loc[curr_train_bool_indices, :]
                        fake_train = synthesizer(curr_train_dataset)
                        fake_train.to_csv(current_gen_path, index=False)
                        pbar_each_task.update(1)
                    else:
                        pbar_each_task.update(1)
                        pbar_each_task.refresh()
            pbar_tasks.update(1)
            pbar_tasks.refresh()
