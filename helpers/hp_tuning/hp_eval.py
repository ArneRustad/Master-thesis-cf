import os
import matplotlib.pyplot as plt
from tqdm.auto import tqdm
from helpers.eval.eval_xgboost_model import fit_and_evaluate_xgboost
import pandas as pd
import numpy as np
from matplotlib.lines import Line2D
import warnings


def evaluate_hyperparams_through_prediction(data_train, data_test, dataset_dir, hyperparams_vec, n_synthetic_datasets,
                                            save_dir=None, save_path=None, figsize=[14, 8], legend_pos=None,
                                            plot_separate=False, subfolder=None,
                                            plot_observations=False, plot_observation_marker="x", plot_sd=True,
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
                                            drop_na=False,
                                            report_na=None,
                                            print_csv_file_paths=False,
                                            remove_unnamed_cols=True):
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
                raise ValueError("Method not yet implemented for tuning more than three hyperparameters simultaneously")

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

    # Create legend_pos variable if not entered as a parameter. Value depends on the bool separate_legends
    # and number of hyperparameters
    if legend_pos is None:
        if separate_legends:
            legend_pos = ["lower left", "lower center"][:n_hyperparams-1]
        else:
            legend_pos = "best"

    if legend_title is None and hyperparams_subname is not None:
        if n_hyperparams == 2:
            legend_title = fr"{hyperparams_subname[1].replace('_', ' ').capitalize()} $=$"
            if only_separate_by_color is None:
                legend_title = [legend_title]
        elif only_separate_by_color:
            legend_title = fr"({', '.join(s.replace('_', ' ').capitalize() for s in hyperparams_subname[1:])}) $=$"
        else:
            legend_title = [fr"{s.replace('_', ' ').capitalize()} $=$" for s in hyperparams_subname]

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
    if n_hyperparams > 3:
        raise ValueError("Method not yet implemented for tuning more than three hyperparameters simultaneously")

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
        result = pd.DataFrame({"Hyperparameters": models, "Accuracy": 0, "AUC": 0, "SD Accuracy": 0, "SD AUC": 0})
        accuracy, auc, categories = fit_and_evaluate_xgboost(data_train, data_test, retcats=True)

        if plot_observations:
            accuracy_obs = np.empty((len(subfolders), n_synthetic_datasets))
            auc_obs = np.empty((len(subfolders), n_synthetic_datasets))

        for i, subfolder in enumerate(subfolders):
            accuracy_vec = np.zeros(n_synthetic_datasets)
            auc_vec = np.zeros(n_synthetic_datasets)
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
                eval_result = fit_and_evaluate_xgboost(fake_train, data_test, categories=categories)
                accuracy_vec[j] = eval_result[0]
                auc_vec[j] = eval_result[1]
                pbar.update(1)
            if plot_observations:
                accuracy_obs[i, :] = accuracy_vec
                auc_obs[i, :] = auc_vec
            accuracy = np.mean(accuracy_vec)
            auc = np.mean(auc_vec)
            accuracy_std = np.std(accuracy_vec)
            auc_std = np.std(auc_vec)
            result.iloc[i, 1:] = [accuracy, auc, accuracy_std, auc_std]

    if plot_observations:
        obs_metric_dict = {"Accuracy": accuracy_obs, "AUC": auc_obs}
    if combined_hp_tuning:
        fig, axes = plt.subplots(nrows=1, ncols=2, figsize=figsize)
        metric_list = ["Accuracy", "AUC"]
        hp_main_vec = [hp_comb[0] for hp_comb in hyperparams_vec]
        hp_sub_combs_vec = [tuple(hp_comb[i] for i in range(len(hp_comb)) if i != 0) for hp_comb in hyperparams_vec]
        hp_unique_sub_combs_vec = sorted(set(hp_sub_combs_vec))

        if only_separate_by_color:
            if n_hyperparams == 2:
                labels_vec = [labels[0] for labels in hp_unique_sub_combs_vec]
            else:
                labels_vec = hp_unique_sub_combs_vec

            for i, curr_hp_sub_combs in enumerate(hp_unique_sub_combs_vec):
                curr_color = next(axes[0]._get_lines.prop_cycler)['color']
                curr_rows = [curr_hp_sub_combs == hp_sub_combs for hp_sub_combs in hp_sub_combs_vec]
                curr_hp_main_vec = [hp_main for hp_main, bool in zip(hp_main_vec, curr_rows) if bool]
                for j, ax in enumerate(axes):
                    plot(ax, curr_hp_main_vec, result.loc[curr_rows, metric_list[j]],
                            label=labels_vec[i],
                            color=curr_color, marker="o")
            plt.plot()
        else:
            hp_sub_vecs = np.swapaxes(hp_sub_combs_vec, 0, 1)
            color_dict = {hp_sub1: next(axes[0]._get_lines.prop_cycler)['color']
                          for hp_sub1 in np.unique(hp_sub_vecs[0,])}
            linestyles = ['-', '--', ':', '-.']
            if hp_sub_vecs.shape[0] >= 2:
                linestyle_dict = {hp_sub2: linestyle
                                  for hp_sub2, linestyle in zip(np.unique(hp_sub_vecs[1,]), linestyles)}
            for i, curr_hp_sub_combs in enumerate(hp_unique_sub_combs_vec):
                curr_rows = [curr_hp_sub_combs == hp_sub_combs for hp_sub_combs in hp_sub_combs_vec]
                curr_hp_main_vec = [hp_main for hp_main, bool in zip(hp_main_vec, curr_rows) if bool]

                if hp_sub_vecs.shape[0] >= 2:
                    curr_linestyle = linestyle_dict[curr_hp_sub_combs[1]]
                else:
                    curr_linestyle = linestyle[0]

                for j, ax in enumerate(axes):
                    plot(ax, curr_hp_main_vec, result.loc[curr_rows, metric_list[j]],
                            label=curr_hp_sub_combs,
                            color=color_dict[curr_hp_sub_combs[0]],
                            linestyle=curr_linestyle,
                            marker="o")
        for i, ax in enumerate(axes):
            ax.set_xscale(x_scale)
            ax.set_title(metric_list[i])

            if bool_x_axis:
                ax.set_xticks([0, 1])
                ax.set_xticklabels(["False", "True"])

            if separate_legends and not only_separate_by_color:
                custom_lines_color = [Line2D([0], [0], color=color)
                                     for color in color_dict.values()]
                legend_color = ax.legend(custom_lines_color, color_dict.keys(), title=legend_title[0],
                                         loc=legend_pos[0])
                if hp_sub_vecs.shape[1] >= 2:
                    custom_lines_linestyle = [Line2D([0], [0], color="black", linestyle=linestyle)
                                             for linestyle in linestyle_dict.values()]
                    legend_linestyle = ax.legend(custom_lines_linestyle, linestyle_dict.keys(),
                                                 title=legend_title[1], loc=legend_pos[1])
                    ax.add_artist(legend_color)
            else:
                ax.legend(loc=legend_pos, title=legend_title)
        plt.plot()

        if result_table_split_hps and not hyperparams_subname is None:
            result_split_hps = pd.DataFrame(data=hyperparams_vec, columns=hyperparams_subname)
            result_split_hps = pd.concat((
                result_split_hps,
                result[["Accuracy", "AUC", "SD Accuracy", "SD AUC"]]
            ), axis=1)
            return result_split_hps
    else:
        if plot_separate:
            fig, axes = plt.subplots(nrows=1, ncols=2, figsize=figsize)
            ax_accuracy, ax_auc = axes
        else:
            fig, ax = plt.subplots(1, figsize=figsize)
            ax_accuracy = ax_auc = ax

        color_accuracy = next(ax_accuracy._get_lines.prop_cycler)['color']
        color_auc = next(ax_auc._get_lines.prop_cycler)['color']

        for ax, metric, col in zip([ax_accuracy, ax_auc], ["Accuracy", "AUC"], [color_accuracy, color_auc]):
            ax.set_xscale(x_scale)
            ax.set_xlabel(label_x_axis)
            plot(ax, hyperparams_vec, result[metric], label=metric, color=col, marker="o")
            if plot_observations:
                ax.scatter(np.repeat(hyperparams_vec, n_synthetic_datasets), obs_metric_dict[metric].flatten(),
                           label=metric + " obs", color=col, marker=plot_observation_marker)
            if plot_sd:
                ax.fill_between(hyperparams_vec, result[metric] - result["SD " + metric],
                                result[metric] + result["SD " + metric], label=fr"{metric} $\pm$ SD {metric}",
                                alpha=0.5, color=color_accuracy)
            if bool_x_axis:
                print("hei")
                ax.set_xticks([0, 1], ["False", "True"])
            if plot_separate:
                ax.set_ylabel(metric)
            ax.legend(loc=legend_pos)
        if not save_path is None:
            if not save_dir is None:
                save_path = os.path.join(save_dir, save_path)
            fig.savefig(save_path)

        if result_table_split_hps and hyperparams_name is not None:
            result_split_hps = pd.DataFrame(data=hyperparams_vec, columns=hyperparams_subname)
            result_split_hps = pd.concat((
                result_split_hps,
                result[["Accuracy", "AUC", "SD Accuracy", "SD AUC"]]
            ), axis=1)
            return result_split_hps
    return result
