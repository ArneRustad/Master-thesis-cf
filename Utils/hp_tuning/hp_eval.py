import os
import matplotlib.pyplot as plt
from tqdm.auto import tqdm
from utils.eval.eval_xgboost_model import fit_and_evaluate_xgboost
import pandas as pd
import numpy as np

def evaluate_hyperparams_through_prediction(data_train, data_test, dataset_dir, hyperparams_vec, n_synthetic_datasets,
                                            save_dir=None, save_path=None, figsize=[14, 8], legend_pos="best",
                                            plot_sd=True, plot_separate=False, subfolder=None,
                                            hyperparams_name="hyperparam", x_scale="linear",
                                            incl_comparison_folder=True,
                                            allow_not_complete_hp_vec=False):
    if not subfolder is None:
        dataset_dir = os.path.join(dataset_dir, subfolder)
    if incl_comparison_folder:
        dataset_dir = os.path.join(dataset_dir, f"{hyperparams_name}_comparison")
    hyperparams_abbreviation_vec = []
    for hyperparams in hyperparams_vec:
        if isinstance(hyperparams, tuple):
            hyperparams_abbreviation_vec.append("".join("_" + str(s) for s in hyperparams))
        else:
            hyperparams_abbreviation_vec.append("_" + str(hyperparams))

    subfolders = [f"{hyperparams_name}{hyperparams}" for hyperparams in hyperparams_abbreviation_vec]

    if allow_not_complete_hp_vec:
        new_subfolders = []
        new_hyperparams_vec = []
        for i, subfolder in enumerate(subfolders):
            curr_dataset_dir = os.path.join(dataset_dir, subfolder)
            path = os.path.join(curr_dataset_dir, f"gen{n_synthetic_datasets-1}.csv")
            if os.path.exists(path):
                new_subfolders.append(subfolder)
                new_hyperparams_vec.append(hyperparams_vec[i])
        subfolders = new_subfolders
        hyperparams_vec = new_hyperparams_vec

    with tqdm(total=len(subfolders) * n_synthetic_datasets) as pbar:
        models = subfolders
        result = pd.DataFrame({"n_epochs": models, "Accuracy": 0, "AUC": 0, "SD Accuracy": 0, "SD AUC": 0})
        accuracy, auc, categories = fit_and_evaluate_xgboost(data_train, data_test, retcats=True)

        for i, subfolder in enumerate(subfolders):
            accuracy_vec = np.zeros(n_synthetic_datasets)
            auc_vec = np.zeros(n_synthetic_datasets)
            curr_dataset_dir = os.path.join(dataset_dir, subfolder)
            for j in range(n_synthetic_datasets):
                path = os.path.join(curr_dataset_dir, f"gen{j}.csv")
                fake_train = pd.read_csv(path, index_col=0)
                eval_result = fit_and_evaluate_xgboost(fake_train, data_test, categories=categories)
                accuracy_vec[j] = eval_result[0]
                auc_vec[j] = eval_result[1]
                pbar.update(1)
            accuracy = np.mean(accuracy_vec)
            auc = np.mean(auc_vec)
            accuracy_std = np.std(accuracy_vec)
            auc_std = np.std(auc_vec)
            result.iloc[i, 1:] = [accuracy, auc, accuracy_std, auc_std]
    fig, ax = plt.subplots(1, figsize=figsize)
    ax.set_xscale(x_scale)
    color_accuracy = next(ax._get_lines.prop_cycler)['color']
    color_auc = next(ax._get_lines.prop_cycler)['color']
    plt.plot(hyperparams_vec, result["Accuracy"], label="Accuracy", color=color_accuracy)
    plt.scatter(hyperparams_vec, result["Accuracy"], color=color_accuracy)
    if plot_sd:
        plt.fill_between(hyperparams_vec, result["Accuracy"] - result["SD Accuracy"],
                         result["Accuracy"] + result["SD Accuracy"], label=r"Accuracy $\pm$ SD Accuracy", alpha=0.5,
                         color=color_accuracy)
    if plot_separate:
        fig, ax = plt.subplots(1, figsize=figsize)
        ax.set_xscale(x_scale)
    plt.plot(hyperparams_vec, result["AUC"], label="AUC", color=color_auc)
    plt.scatter(hyperparams_vec, result["AUC"], color=color_auc)
    if plot_sd:
        plt.fill_between(hyperparams_vec, result["AUC"] - result["SD AUC"], result["AUC"] + result["SD AUC"],
                         label=r"AUC $\pm$ SD AUC", alpha=0.5, color=color_auc)
    plt.legend(loc=legend_pos)
    if not save_path is None:
        if not save_dir is None:
            save_path = os.path.join(save_dir, save_path)
        fig.savefig(save_path)
    return result
