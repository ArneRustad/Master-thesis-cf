import os
import matplotlib.pyplot as plt
from tqdm.auto import tqdm
from helpers.eval.eval_xgboost_model import fit_and_evaluate_xgboost
import pandas as pd
import numpy as np
from matplotlib.lines import Line2D
import warnings
import plotnine as p9


def evaluate_hyperparams_through_prediction(data_train, data_test, dataset_dir, hyperparams_vec, n_synthetic_datasets,
                                            save_dir=None, save_path=None, figsize=[14, 8], legend_pos=None,
                                            plot_separate=True, subfolder=None,
                                            plot_observations=False, plot_observation_marker="x", plot_sd=None,
                                            hyperparams_name="hyperparam",
                                            hyperparams_subname=None,
                                            x_scale="linear",
                                            incl_comparison_folder=True,
                                            allow_not_complete_hp_vec=False,
                                            legend_title=None,
                                            plot_type="line",
                                            only_separate_by_color=False,
                                            separate_legends=False,
                                            result_table_split_hps=False,
                                            label_x_axis=None,
                                            bool_x_axis=False,
                                            str_x_axis=None,
                                            drop_na=False,
                                            report_na=None,
                                            print_csv_file_paths=False,
                                            remove_unnamed_cols=True,
                                            metrics=["Accuracy", "AUC", "F1_0", "F1_1"]):
    if report_na is None:
        report_na=drop_na

    hyperparams_vec = sorted(hyperparams_vec)
    if not subfolder is None:
        dataset_dir = os.path.join(dataset_dir, subfolder)
    if incl_comparison_folder:
        dataset_dir = os.path.join(dataset_dir, f"{hyperparams_name}_comparison")

    hyperparams_abbreviation_vec = []
    combined_hp_tuning = False
    n_hyperparams = None
    for hyperparams in hyperparams_vec:
        if isinstance(hyperparams, (list, tuple)):
            combined_hp_tuning=True
            if n_hyperparams is None:
                n_hyperparams = len(hyperparams)
            elif n_hyperparams != len(hyperparams):
                raise ValueError("Length of hyperparams combinations must constant")

            if hyperparams_subname is None:
                hyperparams_abbreviation_vec.append("".join("_" + str(s) for s in hyperparams))
            else:
                if len(hyperparams_subname) != len(hyperparams):
                    raise ValueError("Length of hyperparams_subname vector must either be of equal length to hyperparams vector or hyperparams_subname must be equal to None")
                hyperparams_abbreviation_vec.append("".join("_" + str(n) + "_" + str(s) for n, s in zip(hyperparams_subname, hyperparams)))
        else:
            if combined_hp_tuning:
                raise ValueError("The number of hyperparameters given must be equal for all combinations.")
            hyperparams_abbreviation_vec.append("_" + str(hyperparams))
            n_hyperparams = 1
    
    if str_x_axis is None:
        if n_hyperparams == 1:
            str_x_axis = isinstance(hyperparams_vec[0], str)
        else:
            str_x_axis = isinstance(hyperparams_vec[0][0], str)
        

    # Create legend_pos variable if not entered as a parameter. Value depends on the bool separate_legends
    # and number of hyperparameters
    if legend_pos is None:
        if separate_legends:
            legend_pos = ["lower left", "lower center"][:n_hyperparams-1]
        else:
            legend_pos = "best"

    if plot_sd is None:
        if n_hyperparams == 1:
            plot_sd = True
        else:
            plot_sd = False

    if legend_title is None:
        if n_hyperparams == 1:
            legend_title = [hyperparams_name.replace('_', ' ').capitalize()]
        else:
            if hyperparams_subname is None:
                raise ValueError("If hyperparams_subname is None then legend_title can't also be None")
            else:
                if only_separate_by_color:
                    legend_title = [
                        hyperparams_subname[0].replace('_', ' ').capitalize(),
                        fr"({', '.join(s.replace('_', ' ').capitalize() for s in hyperparams_subname[1:])}) $=$"
                    ]
                else:
                    legend_title = [fr"{s.replace('_', ' ').capitalize()} $=$" for s in hyperparams_subname[0:]]

    # Asserting valid input parameters (not yet complete)
    if n_hyperparams > 1:
        if (not only_separate_by_color) and separate_legends:
            if (not isinstance(legend_pos, (list, tuple))) or len(legend_pos) != (n_hyperparams - 1):
                raise ValueError(f"When separate_legends=True, then length of legend_pos must be equal to number of wanted legends ({n_hyperparams - 1})")
            if legend_title is None:
                legend_title = [None] * (n_hyperparams - 1)
            elif (not isinstance(legend_title, (list, tuple))) or len(legend_title) != (n_hyperparams - 1):
                    raise ValueError(f"When separate_legends=True, then legend_title must either be equal to None or a list of length equal to number of wanted legends ({n_hyperparams - 1})")
        if only_separate_by_color:
            if not isinstance(legend_pos, str):
                raise ValueError("When using multiple types of hyperparameters in combination with only_separate_by_color=True, then parameter legend_pos must be a string.")
            if not isinstance(legend_title, str):
                raise ValueError("When using multiple types of hyperparameters in combination with only_separate_by_color=True, then parameter legend_title must be a string.")
    else:
        if separate_legends:
            raise ValueError("When number of hyperparameters is equal to 1, then separate_legend must be set equal to False")
    if n_hyperparams > 4 and not only_separate_by_color:
        raise ValueError("Method not yet implemented for tuning more than four hyperparameters simultaneously if only_separate_by_color=False")

    if plot_observations and n_hyperparams > 1:
        warnings.warn("plot_observations is not yet implemented for more than one hyperparameter. This parameter will therefore be ignored")

    plot_types_implemented = ["line", "bar"]
    if plot_type == "line":
            plot = lambda ax, *args, **kwargs: ax.plot(*args, **kwargs)
    elif plot_type == "bar":
        def plot(ax, *args, **kwargs):
            kwargs.pop("marker", None)
            ax.bar(*args, **kwargs)
    else:
        if plot_type not in plot_types_implemented:
            raise ValueError("The parameter plot_type is only implemented for the following types: "
                             "'line', 'bar'")
    if label_x_axis is None:
        if n_hyperparams == 1:
            label_x_axis = hyperparams_name.replace("_", " ").capitalize()
        elif hyperparams_subname is not None:
            label_x_axis = hyperparams_subname[0].replace("_", " ").capitalize()

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
        result = pd.DataFrame({"Hyperparameters": models, "Value Accuracy": 0, "Value AUC": 0,
                               "Value F1": 0, "Value F1_0": 0, "Value F1_1": 0,
                               "SD Accuracy": 0, "SD AUC": 0, "SD F1": 0, "SD F1_0": 0, "SD F1_1": 0})
        metrics_result, categories = fit_and_evaluate_xgboost(data_train, data_test, retcats=True)

        if plot_observations:
            accuracy_obs = np.empty((len(subfolders), n_synthetic_datasets))
            auc_obs = np.empty((len(subfolders), n_synthetic_datasets))
            f1_obs = np.empty((len(subfolders), n_synthetic_datasets))
            f1_0_obs = np.empty((len(subfolders), n_synthetic_datasets))
            f1_1_obs = np.empty((len(subfolders), n_synthetic_datasets))

        for i, subfolder in enumerate(subfolders):
            accuracy_vec = np.zeros(n_synthetic_datasets)
            auc_vec = np.zeros(n_synthetic_datasets)
            f1_vec = np.zeros(n_synthetic_datasets)
            f1_0_vec = np.zeros(n_synthetic_datasets)
            f1_1_vec = np.zeros(n_synthetic_datasets)
            curr_dataset_dir = os.path.join(dataset_dir, subfolder)
            for j in range(n_synthetic_datasets):
                path = os.path.join(curr_dataset_dir, f"gen{j}.csv")
                if print_csv_file_paths:
                    print(path)
                fake_train = pd.read_csv(path)
                if remove_unnamed_cols:
                    unnamed_columns_bool = fake_train.columns.str.contains("Unnamed: ")
                    fake_train = fake_train.loc[:, np.logical_not(unnamed_columns_bool)]
                if (report_na or drop_na) and (fake_train.isna().sum().sum() > 0):
                    fake_train_wo_nan = fake_train.dropna()
                    if report_na:
                        print(f"{fake_train.shape[0] - fake_train_wo_nan.shape[0]} NA found at path: {path}")
                    if drop_na:
                        fake_train=fake_train_wo_nan
                metrics_result = fit_and_evaluate_xgboost(fake_train, data_test, categories=categories)
                accuracy_vec[j] = metrics_result["accuracy"]
                auc_vec[j] = metrics_result["auc"]
                f1_vec[j] = metrics_result["f1"]
                f1_0_vec[j] = metrics_result["f1_0"]
                f1_1_vec[j] = metrics_result["f1_1"]
                pbar.update(1)
            if plot_observations:
                accuracy_obs[i, :] = accuracy_vec
                auc_obs[i, :] = auc_vec
            accuracy = np.mean(accuracy_vec)
            auc = np.mean(auc_vec)
            f1 = np.mean(f1_vec)
            f1_0 = np.mean(f1_0_vec)
            f1_1 = np.mean(f1_1_vec)

            accuracy_std = np.std(accuracy_vec)
            auc_std = np.std(auc_vec)
            f1_std = np.std(f1_vec)
            f1_0_std = np.std(f1_0_vec)
            f1_1_std = np.std(f1_1_vec)
            result.iloc[i, 1:] = [accuracy, auc, f1, f1_0, f1_1, accuracy_std, auc_std, f1_std, f1_0_std, f1_1_std]


    result_split_hps = pd.DataFrame(data=hyperparams_vec, columns=legend_title)
    result_split_hps = pd.concat((
        result_split_hps,
        result.filter(regex="(Value|SD)")
    ), axis=1)
    result_long = pd.wide_to_long(result_split_hps, stubnames=["Value", "SD"], i=legend_title, j="Metric",
                                  sep=" ", suffix="(AUC|Accuracy|F1|F1_0|F1_1)").reset_index()
    unique_values_response = np.sort(np.unique(data_train["income"]))
    result_long = result_long.query("Metric in @metrics")
    if "F1_0" in metrics:
        result_long["Metric"] = result_long["Metric"].str.replace("F1_0", "F1 score: " + unique_values_response[0])
        metrics = [metric.replace("F1_0", "F1 score: " + unique_values_response[0]) for metric in metrics]
    if "F1_1" in metrics:
        result_long["Metric"] = result_long["Metric"].str.replace("F1_1", "F1 score: " + unique_values_response[1])
        metrics = [metric.replace("F1_1", "F1 score: " + unique_values_response[1]) for metric in metrics]
    result_long["Metric"] = pd.Categorical(result_long["Metric"], categories=metrics)
    plot = p9.ggplot(result_long) + p9.aes(x=legend_title[0])
    extra_mapping = {}
    fill_mapping = {}
    if n_hyperparams == 1 and not plot_separate:
        extra_mapping["color"] = "Metric"
        fill_mapping["fill"] = "Metric"
    else:
        plot += p9.facet_wrap("Metric", scales="free_y")
        if n_hyperparams >= 2:
            extra_mapping["color"] = legend_title[1]
            fill_mapping["fill"] = legend_title[2]
        if n_hyperparams >= 3 and not only_separate_by_color:
            extra_mapping["linetype"] = legend_title[2]
        if n_hyperparams >= 4 and not only_separate_by_color:
            extra_mapping["size"] = legend_title[3]
    plot += p9.geom_line(mapping=p9.aes(y="Value", **extra_mapping))
    plot += p9.geom_point(mapping=p9.aes(y="Value", **extra_mapping))

    if plot_sd:
        plot += p9.geom_ribbon(mapping=p9.aes(ymin="Value - SD", ymax="Value + SD", **fill_mapping), alpha=0.5)
    if x_scale == "log":
        plot += p9.scale_x_log10()
    theme_kwargs = {}
    if np.unique(result_long[["Metric"]]).shape[0] > 1:
        theme_kwargs["subplots_adjust"] = {'wspace': 0.15}
    plot += p9.theme(figure_size=figsize, **theme_kwargs)
    plot.draw()
    return result_split_hps
