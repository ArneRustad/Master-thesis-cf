from tqdm.auto import tqdm
import pandas as pd
from helpers.eval.eval_xgboost_model import fit_and_evaluate_xgboost

def eval_tabular_GAN_ml_efficacy(data_train, data_test, dataset_dir, subfolders, n_synthetic_datasets,
                                       name_true_train_dataset = "Train dataset",
                                       eval_sd_true_dataset = False, print_all_accuracy = False):
    progress_bar_total = len(subfolders)*n_synthetic_datasets + (n_synthetic_datasets if eval_sd_true_dataset else 1)
    with tqdm(total=progress_bar_total) as pbar:
        models = [name_true_train_dataset] + subfolders
        result = pd.DataFrame({"Dataset" : models, "Test Accuracy" : None, "Test AUC" : None,
                               "SD Accuracy" : None, "SD AUC" : None})
        if eval_sd_true_dataset:
            accuracy_vec = np.zeros(n_synthetic_datasets)
            auc_vec = np.zeros(n_synthetic_datasets)
            for j in range(n_synthetic_datasets):
                accuracy_curr, auc_curr, categories = fit_and_evaluate_xgboost(data_train, data_test, retcats=True)
                accuracy_vec[j] = accuracy_curr
                auc_vec[j] = auc_curr
                pbar.update(1)
            accuracy = np.mean(accuracy_vec)
            auc = np.mean(auc_vec)
            accuracy_std = np.std(accuracy_vec)
            auc_std = np.std(auc_vec)
        else:
            accuracy, auc, categories = fit_and_evaluate_xgboost(data_train, data_test, retcats = True)
            accuracy_std, auc_std = 0, 0
            pbar.update(1)
        result.iloc[0, 1:] = [accuracy, auc, accuracy_std, auc_std]


        for i, subfolder in enumerate(subfolders, start=1):
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
            if print_all_accuracy:
                print(subfolder, accuracy_vec)
            accuracy = np.mean(accuracy_vec)
            auc = np.mean(auc_vec)
            accuracy_std = np.std(accuracy_vec)
            auc_std = np.std(auc_vec)
            result.iloc[i,1:] = [accuracy, auc, accuracy_std, auc_std]
        return result