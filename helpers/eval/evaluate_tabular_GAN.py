from tqdm.auto import tqdm
import pandas as pd
from helpers.eval.eval_xgboost_model import fit_and_evaluate_xgboost
import numpy as np
import os

def eval_tabular_GAN_ml_efficacy(data_train, data_test, dataset_dir, subfolders, n_synthetic_datasets,
                                       name_true_train_dataset = "Train dataset",
                                       eval_sd_true_dataset = False, print_all_accuracy = False,
                                 remove_unnamed_cols=True,
                                 metrics=["Accuracy", "AUC", "F1", "F1_0", "F1_1"]):
    progress_bar_total = len(subfolders)*n_synthetic_datasets + (n_synthetic_datasets if eval_sd_true_dataset else 1)
    with tqdm(total=progress_bar_total) as pbar:
        models = [name_true_train_dataset] + subfolders
        result = pd.DataFrame({"Dataset" : models,
                               "Test Accuracy": None, "Test AUC" : None,
                               "Test F1": None,"Test F1_0": None, "Test F1_1": None,
                               "SD Accuracy": None, "SD AUC": None, "SD F1": None, "SD F1_0": None, "SD F1_1": None})
        if eval_sd_true_dataset:
            accuracy_vec = np.zeros(n_synthetic_datasets)
            auc_vec = np.zeros(n_synthetic_datasets)
            f1_vec = np.zeros(n_synthetic_datasets)
            f1_0_vec = np.zeros(n_synthetic_datasets)
            f1_1_vec = np.zeros(n_synthetic_datasets)
            for j in range(n_synthetic_datasets):
                metrics_curr, categories = fit_and_evaluate_xgboost(data_train, data_test, retcats=True)
                accuracy_vec[j] = metrics_curr["accuracy"]
                auc_vec[j] = metrics_curr["auc"]
                f1_vec[j] = metrics_curr["f1"]
                f1_0_vec[j] = metrics_curr["f1_0"]
                f1_1_vec[j] = metrics_curr["f1_1"]
                pbar.update(1)
            accuracy = np.mean(accuracy_vec)
            auc = np.mean(auc_vec)
            f1 = np.mean(f1)
            f1_0 = np.mean(f1_0_vec)
            f1_1 = np.mean(f1_1_vec)
            accuracy_std = np.std(accuracy_vec)
            auc_std = np.std(auc_vec)
            f1_std = np.std(f1_vec)
            f1_0_std = np.std(f1_0_vec)
            f1_1_std = np.std(f1_1_vec)
        else:
            metrics, categories = fit_and_evaluate_xgboost(data_train, data_test, retcats = True)
            accuracy_std, auc_std, f1_std, f1_0_std, f1_1_std = 0, 0, 0, 0, 0
            accuracy, auc = metrics["accuracy"], metrics["auc"]
            f1, f1_0, f1_1 = metrics["f1"], metrics["f1_0"], metrics["f1_1"]
            pbar.update(1)
        result.iloc[0, 1:] = [accuracy, auc, f1, f1_0, f1_1, accuracy_std, auc_std, f1_std, f1_0_std, f1_1_std]


        for i, subfolder in enumerate(subfolders, start=1):
            accuracy_vec = np.zeros(n_synthetic_datasets)
            auc_vec = np.zeros(n_synthetic_datasets)
            f1_vec = np.zeros(n_synthetic_datasets)
            f1_0_vec = np.zeros(n_synthetic_datasets)
            f1_1_vec = np.zeros(n_synthetic_datasets)
            curr_dataset_dir = os.path.join(dataset_dir, subfolder)
            for j in range(n_synthetic_datasets):
                path = os.path.join(curr_dataset_dir, f"gen{j}.csv")
                fake_train = pd.read_csv(path)
                if remove_unnamed_cols:
                    unnamed_columns_bool = fake_train.columns.str.contains("Unnamed: ")
                    fake_train = fake_train.loc[:, np.logical_not(unnamed_columns_bool)]
                metrics_curr = fit_and_evaluate_xgboost(fake_train, data_test, categories = categories)
                accuracy_vec[j] = metrics_curr["accuracy"]
                auc_vec[j] = metrics_curr["auc"]
                f1_vec[j] = metrics_curr["f1"]
                f1_0_vec[j] = metrics_curr["f1_0"]
                f1_1_vec[j] = metrics_curr["f1_1"]
                pbar.update(1)
            if print_all_accuracy:
                print(subfolder, accuracy_vec)
            accuracy = np.mean(accuracy_vec)
            auc = np.mean(auc_vec)
            f1 = np.mean(f1_vec)
            f1_0_std = np.std(f1_0_vec)
            f1_1_std = np.std(f1_1_vec)
            accuracy_std = np.std(accuracy_vec)
            auc_std = np.std(auc_vec)
            f1_std = np.std(f1_vec)
            f1_0 = np.mean(f1_0_vec)
            f1_1 = np.mean(f1_1_vec)
            result.iloc[i, 1:] = [accuracy, auc, f1, f1_0, f1_1, accuracy_std, auc_std, f1_std, f1_0_std, f1_1_std]
        unique_values_response = np.sort(np.unique(data_train["income"]))
        if "F1_0" in metrics:
            metrics = [metric.replace("F1_0", "F1 score: " + unique_values_response[0]) for metric in metrics]
            result.columns = result.columns.str.replace("F1_0", "F1 score: " + unique_values_response[0])
        if "F1_1" in metrics:
            metrics = [metric.replace("F1_1", "F1 score: " + unique_values_response[1]) for metric in metrics]
            result.columns = result.columns.str.replace("F1_1", "F1 score: " + unique_values_response[1])
        return result