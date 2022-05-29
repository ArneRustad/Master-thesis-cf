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
                                            save_dir=None, save_path=None, figsize=[14, 8],
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
                                            label_x_axis=None,
                                            x_tick_angle=0,
                                            drop_na=False,
                                            report_na=None,
                                            print_csv_file_paths=False,
                                            remove_unnamed_cols=True,
                                            metrics=["Accuracy", "AUC", "F1_0", "F1_1"],
                                            force_lines_for_bool_or_string_x_axis=True):
    if report_na is None:
        report_na = drop_na

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
                    legend_title = [fr"{s.replace('_', ' ').capitalize()}" for s in hyperparams_subname]

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

    metrics_all = ["Accuracy", "AUC", "F1_0", "F1_1", "F1"]
    with tqdm(total=len(subfolders) * n_synthetic_datasets) as pbar:
        models = subfolders
        result = pd.DataFrame({"Hyperparameters": models, "Value Accuracy": 0, "Value AUC": 0,
                               "Value F1": 0, "Value F1_0": 0, "Value F1_1": 0,
                               "SD Accuracy": 0, "SD AUC": 0, "SD F1": 0, "SD F1_0": 0, "SD F1_1": 0})
        metrics_result, categories = fit_and_evaluate_xgboost(data_train, data_test, retcats=True, metrics=metrics_all)

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
                metrics_result = fit_and_evaluate_xgboost(fake_train, data_test, categories=categories,
                                                          metrics=metrics_all)
                accuracy_vec[j] = metrics_result["accuracy"]
                auc_vec[j] = metrics_result["auc"]
                f1_vec[j] = metrics_result["f1"]
                f1_0_vec[j] = metrics_result["f1_0"]
                f1_1_vec[j] = metrics_result["f1_1"]
                pbar.update(1)
            if plot_observations:
                accuracy_obs[i, :] = accuracy_vec
                auc_obs[i, :] = auc_vec
                f1_obs[i, :] = f1_vec
                f1_0_obs[i, :] = f1_0_vec
                f1_1_obs[i, :] = f1_1_vec
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
    if n_hyperparams > 1:
        result_split_hps.iloc[:, 1:] = result_split_hps.iloc[:, 1:].astype("str")
    bool_x_axis = False
    str_x_axis = False
    if result_split_hps.dtypes.values[0] == "bool" and force_lines_for_bool_or_string_x_axis:
        bool_x_axis = True
        result_split_hps.iloc[:, 0] = result_split_hps.iloc[:, 0].astype(int)
    elif result_split_hps.dtypes.values[0] == "object" and force_lines_for_bool_or_string_x_axis:
        str_x_axis = True
        unique_str_x_values = np.sort(np.unique(result_split_hps.iloc[:, 0]))
        dict_str_to_int = {unique_str_x_values[i]: i for i in range(unique_str_x_values.shape[0])}
        dict_int_to_str = {i: unique_str_x_values[i] for i in range(unique_str_x_values.shape[0])}
        result_split_hps.iloc[:, 0] = result_split_hps.iloc[:, 0].map(lambda x: dict_str_to_int[x])

    if plot_observations:
        list_dfs = []
        obs_dataset_columns = [f"Observation{i}" for i in range(n_synthetic_datasets)]
        for metric, metric_matrix in zip(["Accuracy", "AUC", "F1", "F1_0", "F1_1"],
                                         [accuracy_obs, auc_obs, f1_obs, f1_0_obs, f1_1_obs]):
            result_obs_single_metric = pd.DataFrame(metric_matrix,
                                                    columns=obs_dataset_columns)
            result_obs_single_metric = pd.concat((
                result_split_hps,
                result_obs_single_metric
            ), axis=1)
            result_obs_single_metric["Metric"] = metric
            list_dfs.append(result_obs_single_metric)
        result_obs = pd.concat(list_dfs, ignore_index=True)
        result_obs_long = pd.wide_to_long(result_obs, stubnames="Observation", i=legend_title + ["Metric"],
                                          j="Dataset", sep="").reset_index().query("Metric in @metrics")

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
        if plot_observations:
            result_obs_long["Metric"] = result_obs_long["Metric"].str.replace("F1_0", "F1 score: " +
                                                                              unique_values_response[0])
    if "F1_1" in metrics:
        result_long["Metric"] = result_long["Metric"].str.replace("F1_1", "F1 score: " + unique_values_response[1])
        metrics = [metric.replace("F1_1", "F1 score: " + unique_values_response[1]) for metric in metrics]
        if plot_observations:
            result_obs_long["Metric"] = result_obs_long["Metric"].str.replace("F1_1", "F1 score: " +
                                                                      unique_values_response[1])
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
            fill_mapping["fill"] = legend_title[1]
        if n_hyperparams >= 3 and not only_separate_by_color:
            extra_mapping["linetype"] = legend_title[2]
        if n_hyperparams >= 4 and not only_separate_by_color:
            extra_mapping["size"] = legend_title[3]
    plot += p9.geom_line(mapping=p9.aes(y="Value", **extra_mapping))
    plot += p9.geom_point(mapping=p9.aes(y="Value", **extra_mapping))

    if plot_sd:
        plot += p9.geom_ribbon(mapping=p9.aes(ymin="Value - SD", ymax="Value + SD", **fill_mapping), alpha=0.5)
    if bool_x_axis:
        plot += p9.scale_x_continuous(breaks=[0, 1], labels=["False", "True"], expand=[0.1, 0.1])
    if str_x_axis:
        plot += p9.scale_x_continuous(breaks=[i for i in range(unique_str_x_values.shape[0])],
                                      labels=unique_str_x_values, expand=[0.1, 0.1])
    if x_scale == "log":
        plot += p9.scale_x_log10()
    theme_kwargs = {"axis_text_x": p9.element_text(angle=x_tick_angle)}
    if np.unique(result_long[["Metric"]]).shape[0] > 1:
        theme_kwargs["subplots_adjust"] = {'wspace': 0.15}
    if plot_observations:
        plot += p9.geom_point(mapping=p9.aes(y="Observation", **extra_mapping), data=result_obs_long,
                              shape=plot_observation_marker)
    plot += p9.theme(figure_size=figsize, **theme_kwargs)

    if save_path is not None:
        if save_dir is not None:
            save_path = os.path.join(save_dir, save_path)
        plot.save(filename=save_path, width=figsize[0], height=figsize[1], units="in")
    plot.draw()

    if bool_x_axis:
        result_split_hps.iloc[:, 0] = result_split_hps.iloc[:, 0].astype(bool)
    elif str_x_axis:
        result_split_hps.iloc[:, 0] = result_split_hps.iloc[:, 0].map(lambda x: dict_int_to_str[x])
    return result_split_hps
