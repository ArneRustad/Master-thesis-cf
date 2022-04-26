from tqdm.auto import tqdm
import matplotlib.pyplot as plt
import os
from helpers.eval.eval_xgboost_model import fit_and_evaluate_xgboost
import pandas as pd
import numpy as np
import plotnine as p9

def evaluate_n_epochs_through_prediction(data_train, data_test, dataset_dir, n_epochs_vec, n_synthetic_datasets,
                                         save_dir = None, save_path = None, figsize = [14,8], legend_pos="best",
                                         subfolder=None, incl_comparison_folder=True,
                                         drop_na=False,
                                         report_na=None,
                                         print_csv_file_paths=False,
                                         plot_separate=False,
                                         epoch_split=None):

    if report_na is None:
        report_na=drop_na

    if not subfolder is None:
        dataset_dir = os.path.join(dataset_dir, subfolder)
    if incl_comparison_folder:
        dataset_dir = os.path.join(dataset_dir, "n_epochs_comparison/")
    subfolders = [f"Epochs{n_epochs}" for n_epochs in n_epochs_vec]
    with tqdm(total=len(subfolders)*n_synthetic_datasets, leave=False) as pbar:
        models = subfolders
        result = pd.DataFrame({"n_epochs": n_epochs_vec,
                              "Value Accuracy": 0, "Value AUC": 0,
                               "SD Accuracy": 0, "SD AUC": 0})
        accuracy, auc, categories = fit_and_evaluate_xgboost(data_train, data_test, retcats = True)

        for i, subfolder in enumerate(subfolders):
            accuracy_vec = np.zeros(n_synthetic_datasets)
            auc_vec = np.zeros(n_synthetic_datasets)
            curr_dataset_dir = os.path.join(dataset_dir, subfolder)
            for j in range(n_synthetic_datasets):
                path = os.path.join(curr_dataset_dir, f"gen{j}.csv")
                fake_train = pd.read_csv(path, index_col=0)
                if (report_na or drop_na) and (fake_train.isna().sum().sum() > 0):
                    fake_train_wo_nan = fake_train.dropna()
                    if report_na:
                        print(f"{fake_train.shape[0] - fake_train_wo_nan.shape[0]} NA found at path: {path}")
                    if drop_na:
                        fake_train=fake_train_wo_nan
                eval_result = fit_and_evaluate_xgboost(fake_train, data_test, categories = categories)
                accuracy_vec[j] = eval_result[0]
                auc_vec[j] = eval_result[1]
                pbar.update(1)
            accuracy = np.mean(accuracy_vec)
            auc = np.mean(auc_vec)
            accuracy_std = np.std(accuracy_vec)
            auc_std = np.std(auc_vec)
            result.iloc[i, 1:] = [accuracy, auc, accuracy_std, auc_std]

    result_long = pd.wide_to_long(result, stubnames=["Value", "SD"], i=["n_epochs"], j="Metric",
                                  sep=" ", suffix="(AUC|Accuracy)").reset_index()
    if epoch_split is not None:
        result_long["epoch_split"] = np.where(result_long["n_epochs"] <= epoch_split,
                                              f"Epoch 1-{epoch_split}",
                                              f"Epoch {epoch_split+1}-{max(n_epochs_vec)}")
    if plot_separate:
        color_mapping = {}
        fill_mapping = {}
    else:
        color_mapping = {"color": "Metric"}
        fill_mapping = {"fill": "Metric"}

    plot = (p9.ggplot(result_long) + p9.aes(x="n_epochs") +
     p9.geom_line(mapping=p9.aes(y="Value", **color_mapping)) +
     p9.geom_point(mapping=p9.aes(y="Value", **color_mapping)) +
     p9.geom_ribbon(mapping=p9.aes(ymin="Value - SD", ymax="Value + SD", **fill_mapping), alpha=0.5) +
     p9.xlab("Epoch")
     )

    theme_kwargs = {}
    if epoch_split and plot_separate:
        plot += p9.facet_wrap("~ Metric + epoch_split", scales="free", ncol=2)
        theme_kwargs = {"subplots_adjust": {'wspace': 0.15}}
    elif epoch_split:
        plot += p9.facet_wrap("~ epoch_split", scales="free", ncol=2)
        theme_kwargs = {"subplots_adjust": {'wspace': 0.15}}
    elif plot_separate:
        plot += p9.facet_wrap("Metric", scales="free")
        theme_kwargs = {"subplots_adjust": {'wspace': 0.15}}
    plot += p9.theme(figure_size=figsize, **theme_kwargs)

    plot.draw()
    # fig, ax = plt.subplots(1, figsize = figsize)
    # color_accuracy = next(ax._get_lines.prop_cycler)['color']
    # color_auc = next(ax._get_lines.prop_cycler)['color']
    # plt.plot(n_epochs_vec, result["Accuracy"], label = "Accuracy", color = color_accuracy)
    # plt.scatter(n_epochs_vec, result["Accuracy"], color = color_accuracy)
    # plt.fill_between(n_epochs_vec, result["Accuracy"] - result["SD Accuracy"],
    #                  result["Accuracy"] + result["SD Accuracy"],
    #                  label = r"Accuracy $\pm$ SD Accuracy", alpha = 0.5, color=color_accuracy)
    # plt.plot(n_epochs_vec, result["AUC"], label = "AUC", color=color_auc)
    # plt.scatter(n_epochs_vec, result["AUC"], color=color_auc)
    # plt.fill_between(n_epochs_vec, result["AUC"] - result["SD AUC"], result["AUC"] + result["SD AUC"],
    #                  label = r"AUC $\pm$ SD AUC", alpha = 0.5, color=color_auc)
    # plt.legend(loc=legend_pos)
    # if not save_path is None:
    #     if not save_dir is None:
    #         save_path = os.path.join(save_dir, save_path)
    #         os.makedirs(save_dir, exist_ok=True)
    #     fig.savefig(save_path)
    return result