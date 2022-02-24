from tqdm.auto import tqdm
import matplotlib.pyplot as plt
import os
from utils.eval.eval_xgboost_model import fit_and_evaluate_xgboost
import pandas as pd
import numpy as np

def evaluate_n_epochs_through_prediction(data_train, data_test, dataset_dir, n_epochs_vec, n_synthetic_datasets,
                                         save_dir = None, save_path = None, figsize = [14,8], legend_pos="best",
                                         subfolder=None, incl_comparison_folder=True):
    if not subfolder is None:
        dataset_dir = os.path.join(dataset_dir, subfolder)
    if incl_comparison_folder:
        dataset_dir = os.path.join(dataset_dir, "n_epochs_comparison/")
    subfolders = [f"Epochs{n_epochs}" for n_epochs in n_epochs_vec]
    with tqdm(total=len(subfolders)*n_synthetic_datasets, leave=False) as pbar:
        models = subfolders
        result = pd.DataFrame({"n_epochs" : models, "Accuracy" : 0, "AUC" : 0,
                               "SD Accuracy" : 0, "SD AUC" : 0})
        accuracy, auc, categories = fit_and_evaluate_xgboost(data_train, data_test, retcats = True)

        for i, subfolder in enumerate(subfolders):
            accuracy_vec = np.zeros(n_synthetic_datasets)
            auc_vec = np.zeros(n_synthetic_datasets)
            curr_dataset_dir = os.path.join(dataset_dir, subfolder)
            for j in range(n_synthetic_datasets):
                path = os.path.join(curr_dataset_dir, f"gen{j}.csv")
                fake_train = pd.read_csv(path, index_col = 0)
                eval_result = fit_and_evaluate_xgboost(fake_train, data_test, categories = categories)
                accuracy_vec[j] = eval_result[0]
                auc_vec[j] = eval_result[1]
                pbar.update(1)
            accuracy = np.mean(accuracy_vec)
            auc = np.mean(auc_vec)
            accuracy_std = np.std(accuracy_vec)
            auc_std = np.std(auc_vec)
            result.iloc[i,1:] = [accuracy, auc, accuracy_std, auc_std]
    fig, ax = plt.subplots(1, figsize = figsize)
    color_accuracy = next(ax._get_lines.prop_cycler)['color']
    color_auc = next(ax._get_lines.prop_cycler)['color']
    plt.plot(n_epochs_vec, result["Accuracy"], label = "Accuracy", color = color_accuracy)
    plt.scatter(n_epochs_vec, result["Accuracy"], color = color_accuracy)
    plt.fill_between(n_epochs_vec, result["Accuracy"] - result["SD Accuracy"],
                     result["Accuracy"] + result["SD Accuracy"],
                     label = r"Accuracy $\pm$ SD Accuracy", alpha = 0.5, color=color_accuracy)
    plt.plot(n_epochs_vec, result["AUC"], label = "AUC", color=color_auc)
    plt.scatter(n_epochs_vec, result["AUC"], color=color_auc)
    plt.fill_between(n_epochs_vec, result["AUC"] - result["SD AUC"], result["AUC"] + result["SD AUC"],
                     label = r"AUC $\pm$ SD AUC", alpha = 0.5, color=color_auc)
    plt.legend(loc=legend_pos)
    if not save_path is None:
        if not save_dir is None:
            save_path = os.path.join(save_dir, save_path)
            os.makedirs(save_dir, exist_ok=True)
        fig.savefig(save_path)
    return result