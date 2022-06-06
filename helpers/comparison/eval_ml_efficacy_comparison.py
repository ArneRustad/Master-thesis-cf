from tqdm.auto import tqdm
import pandas as pd
from helpers.eval.eval_xgboost_model import fit_and_evaluate_xgboost
import numpy as np
import os
from src import constants as const

RESPONSE_DICT = {
    "covtype_edited": "Cover_Type",
    "creditcard_edited": "Class",
    "news_edited": "shares"
}

METRIC_DICT = {
    "covtype_edited": ["Accuracy", "F1"],
    "creditcard_edited": ["Accuracy", "AUPRC", "F1"],
    "news_edited": ["Accuracy", "AUC", "F1"]
}


def transform_news_dataset(news):
    return news.assign(shares=lambda x: np.where(x.shares > 3395, ">3395 (Popular)", "<=3395 (Unpopular)"))


DATA_TRANSFORM_DICT = {
    "covtype_edited": None,
    "creditcard_edited": None,
    "news_edited": transform_news_dataset
}

DATA_TASK_IS_CLASSIFICATION_DICT = {
    "covtype_edited": True,
    "creditcard_edited": True,
    "news_edited": True
}


def eval_ml_efficacy_for_synthesizers(synthesizer_names,
                                      datasets=["covtype_edited", "creditcard_edited", "news_edited"],
                                      n_synthetic_datasets=5,
                                      name_true_train_dataset="Train dataset",
                                      dataset_dir=const.dir.data_comparison(),
                                      gen_dataset_dir=const.dir.data_comparison_gen(),
                                      remove_unnamed_cols=False,  # not implemented yet
                                      metrics_dict=None,
                                      metrics_same_all=None,
                                      metric_evals=["Mean", "Median"],
                                      progress_bar_tasks=True, progress_bar_leave=True,
                                      progress_bar_models=True,
                                      progress_bar_each_model=True,
                                      ret_count_nan=False,
                                      allow_fewer_synthetic_datasets=False):
    models = [name_true_train_dataset] + synthesizer_names
    result_datasets = {}
    metric_evals_lower = [metric_eval.lower() for metric_eval in metric_evals]
    n_models = len(models)
    df_count_nan = pd.DataFrame({**{"Models": models},
                                 **{dataset: np.zeros(n_models) for dataset in datasets}})
    if metrics_dict is None:
        if metrics_same_all is not None:
            metrics_dict = {dataset_task: metrics_same_all for dataset_task in datasets}
        else:
            metrics_dict = METRIC_DICT
    with tqdm(total=len(datasets), desc=f"Dataset tasks", leave=progress_bar_leave, disable=not progress_bar_tasks) \
            as pbar_tasks:
        for dataset_task in datasets:
            curr_dataset = pd.read_csv(os.path.join(dataset_dir, dataset_task + ".csv"))
            curr_dir_dataset_train_indices = os.path.join(dataset_dir, "indices", dataset_task)
            curr_metrics = metrics_dict[dataset_task]
            curr_metrics_lower = [curr_metric.lower() for curr_metric in curr_metrics]

            curr_data_transform = DATA_TRANSFORM_DICT[dataset_task]
            if curr_data_transform is not None:
                curr_dataset = curr_data_transform(curr_dataset)

            result_dataset = pd.DataFrame(
                {**{"Model": models},
                 **{f"{metric_eval} {metric}": None
                    for metric in curr_metrics
                    for metric_eval in metric_evals}
                 }
            )
            dict_arr_results = {metric: np.empty(shape=(len(models), n_synthetic_datasets), dtype=np.float64)
                                for metric in curr_metrics_lower}
            params_extra_xgboost = {"response_col": RESPONSE_DICT[dataset_task],
                                    "metrics": curr_metrics,
                                    "data_transform": None,
                                    "classification": DATA_TASK_IS_CLASSIFICATION_DICT[dataset_task]}
            if DATA_TASK_IS_CLASSIFICATION_DICT[dataset_task]:
                unique_classification_values = curr_dataset[RESPONSE_DICT[dataset_task]].unique()
            else:
                unique_classification_values = None
            if allow_fewer_synthetic_datasets:
                dict_count_n_synthetic_datasets = {model: 0 for model in models}
            for j in tqdm(range(n_synthetic_datasets), desc="Synthetic datasets", leave=False):
                curr_boolean_train_indices = np.load(os.path.join(curr_dir_dataset_train_indices,
                                                                  f"bool_indices_{j}.npy"))
                curr_data_train = curr_dataset.loc[curr_boolean_train_indices, :]
                curr_data_test = curr_dataset.loc[np.logical_not(curr_boolean_train_indices), :]

                for i, model in enumerate(tqdm(models, desc="Models", leave=False)):
                    if model == name_true_train_dataset:
                        result_curr_model_and_dataset, curr_cats = fit_and_evaluate_xgboost(
                            data_train=curr_data_train,
                            data_test=curr_data_test,
                            categories="auto",
                            retcats=True,
                            **params_extra_xgboost
                        )
                        if allow_fewer_synthetic_datasets:
                            dict_count_n_synthetic_datasets[model] += 1
                    else:
                        curr_dataset_path = os.path.join(gen_dataset_dir, model, dataset_task, f"gen{j}.csv")
                        if allow_fewer_synthetic_datasets and not os.path.exists(curr_dataset_path):
                            if j == 0:
                                raise RuntimeError(f"Model {model} has zero synthetic datasets for dataset task "
                                                   f"{dataset_task}")
                            elif j == dict_count_n_synthetic_datasets[model]:
                                print(f"Model {model} only has {dict_count_n_synthetic_datasets[model]} synthetic "
                                      f"datasets for dataset task {dataset_task}")
                                result_curr_model_and_dataset = {metric: np.nan for metric in curr_metrics_lower}
                        else:
                            fake_train_dataset = pd.read_csv(curr_dataset_path)
                            if curr_data_transform is not None:
                                fake_train_dataset = curr_data_transform(fake_train_dataset)
                            if unique_classification_values is not None:
                                fake_train_unique_classification_values = list(
                                    fake_train_dataset[RESPONSE_DICT[dataset_task]].unique())

                            if unique_classification_values is None or \
                                    all([str(class_val) in fake_train_unique_classification_values
                                         for class_val in unique_classification_values]):
                                result_curr_model_and_dataset = fit_and_evaluate_xgboost(
                                    data_train=fake_train_dataset,
                                    data_test=curr_data_test,
                                    categories=curr_cats,
                                    retcats=False,
                                    **params_extra_xgboost
                                )
                                if allow_fewer_synthetic_datasets:
                                    dict_count_n_synthetic_datasets[model] += 1
                            else:
                                result_curr_model_and_dataset = {metric: np.nan for metric in curr_metrics_lower}
                                df_count_nan.loc[i, dataset_task] += 1
                    for metric in curr_metrics_lower:
                        dict_arr_results[metric][i, j] = result_curr_model_and_dataset[metric]
            print(dict_arr_results)
            for metric, metric_lower in zip(curr_metrics, curr_metrics_lower):
                for metric_eval, metric_eval_lower in zip(metric_evals, metric_evals_lower):
                    if metric_eval_lower == "mean":
                        metric_eval_func = np.mean
                    elif metric_eval_lower == "median":
                        metric_eval_func = np.median
                    elif metric_eval_lower[:10] == "percentile":
                        metric_eval_func = lambda x, axis: np.percentile(x, q=float(metric_eval_lower[10:]), axis=axis)
                    else:
                        raise ValueError(f"All metric evals must be one of 'mean', 'median' or 'percentile[d]', "
                                         "where [d] is a float between 0 and 100. You entered as metric: {metric}")
                    if allow_fewer_synthetic_datasets:
                        for i, model in enumerate(models):
                            result_dataset.loc[i, f"{metric_eval} {metric}"] = metric_eval_func(
                                dict_arr_results[metric_lower][i, 0:(dict_count_n_synthetic_datasets[model])],
                                axis=None
                            )
                    else:
                        result_dataset[f"{metric_eval} {metric}"] = metric_eval_func(
                            dict_arr_results[metric_lower], axis=1
                        )

            pbar_tasks.update()
            result_datasets[dataset_task] = result_dataset
    if ret_count_nan:
        result_datasets, df_count_nan
    else:
        return result_datasets


def tidy_comparison_ml_efficacy_output(dict_result_datasets,
                                       metrics_dict=None,
                                       extract_median_and_percentile=False, percentiles=None,
                                       round_decimals=3, remove_edited_from_dataset_task_name=True,
                                       add_test_to_metric_name=True):
    dataset_tasks = list(dict_result_datasets.keys())
    if extract_median_and_percentile:
        if percentiles is None:
            df = dict_result_datasets[dataset_tasks[0]]
            percentile_cols = df.columns[df.columns.str.contains(pat="Percentile")]
            percentiles_unique = percentile_cols.str.replace("Percentile", "").str.extract('^(\d+)')[0].unique().astype(
                float)
            if len(percentiles_unique) == 2:
                percentiles = [round(percentile) for percentile in sorted(percentiles_unique)]
            else:
                raise ValueError(
                    f"Can't determine which percentiles should be default. Found {len(percentiles)} unique "
                    f"percentiles in the first dataframe: {percentiles_unique}")
        elif len(percentiles) != 2:
            raise ValueError(f"The parameter percentiles must be of length 2 if used. You entered {percentiles}")
    cols = [("", "Model")]
    for i, dataset_task in enumerate(dataset_tasks):
        curr_dataset = dict_result_datasets[dataset_task]
        if extract_median_and_percentile:
            if metrics_dict is None:
                curr_metrics_all = [splitted_col[1]
                                    for splitted_col in curr_dataset.drop("Model", axis=1).columns.str.split()]
                curr_metrics = []
                for metric in curr_metrics_all:
                    if metric not in curr_metrics:
                        curr_metrics.append(metric)
            else:
                curr_metrics = metrics_dict[dataset_task]
            cols_median_and_percentile = [f"{metric_eval} {metric}"
                                          for metric in curr_metrics
                                          for metric_eval in ["Median", f"Percentile{percentiles[0]}",
                                                              f"Percentile{percentiles[1]}"]
                                          ]
            curr_dataset = curr_dataset[["Model"] + cols_median_and_percentile]
            curr_dataset.loc[:, cols_median_and_percentile] = curr_dataset.loc[:, cols_median_and_percentile].round(
                decimals=round_decimals)
            new_curr_dataset = pd.DataFrame({**{"Model": curr_dataset["Model"]},
                                             **{metric: None for metric in curr_metrics}})
            for metric in curr_metrics:
                new_curr_dataset.loc[:, metric] = (curr_dataset[f"Median {metric}"].astype(str) + " (" +
                                                   curr_dataset[f"Percentile{percentiles[0]} {metric}"].astype(
                                                       str) + ", " +
                                                   curr_dataset[f"Percentile{percentiles[1]} {metric}"].astype(
                                                       str) + ")"
                                                   )
            if add_test_to_metric_name:
                new_curr_dataset.columns = [f"Test {col}" if col != "Model" else col
                                            for col in new_curr_dataset.columns
                                            ]
            curr_dataset = new_curr_dataset
        metric_columns = [col for col in curr_dataset.columns if col != "Model"]
        tidyed_dataset_task = dataset_task.title()
        if remove_edited_from_dataset_task_name:
            tidyed_dataset_task = tidyed_dataset_task.replace("_Edited", "")
        cols += [(tidyed_dataset_task, col) for col in metric_columns]
        if i == 0:
            combined_dataset = curr_dataset
        else:
            combined_dataset = combined_dataset.merge(curr_dataset,
                                                      how='inner', on="Model",
                                                      left_index=False, right_index=False, sort=False,
                                                      suffixes=(
                                                      "" if i > 1 else f"_{dataset_tasks[0]}", f"_{dataset_task}")
                                                      )
    combined_dataset.columns = pd.MultiIndex.from_tuples(cols)
    return combined_dataset
