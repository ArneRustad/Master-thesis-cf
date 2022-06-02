import numpy as np
import pandas as pd
from xgboost import XGBClassifier, XGBRegressor
from sklearn.metrics import (normalized_mutual_info_score, adjusted_mutual_info_score,
                             accuracy_score, roc_auc_score, f1_score, mean_squared_error, mean_absolute_error
                             )
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.linear_model import LinearRegression, Lasso, LassoCV, ElasticNetCV
import tensorflow as tf

def extract_numeric_and_discrete_columns(X):
    columns_discrete_bool = [str(b) in ["object", "category"] for b in X.dtypes]
    discrete_columns_of_X = X.columns[columns_discrete_bool]
    numeric_columns_of_X = X.columns[np.logical_not(columns_discrete_bool)]
    return numeric_columns_of_X, discrete_columns_of_X

def fit_and_evaluate_xgboost(data_train, data_test, categories = "auto", retcats = False, response_col = "income",
                             tree_method=None, metrics=["accuracy", "auc", "f1"], classification=None,
                             model_type="xgboost", model_params={},
                             data_transform=None):
    if data_transform is not None:
        data_train = data_transform(data_train)
        data_test = data_transform(data_test)
    if classification is None:
        if pd.api.types.is_numeric_dtype(data_train[response_col]):
            classification = False
        else:
            classification = True

    metrics = [metric.lower() for metric in metrics]
    if tree_method is None:
        if len(tf.config.list_physical_devices("gpu")) > 0:
            tree_method = "gpu_hist"
        else:
            tree_method = "hist"

    X_train = data_train.iloc[:, data_train.columns != response_col]
    Y_train = data_train[response_col]
    X_test = data_test.iloc[:, data_train.columns != response_col]
    Y_test = data_test[response_col]

    numeric_columns_of_X, discrete_columns_of_X = extract_numeric_and_discrete_columns(X_train)

    X_train[discrete_columns_of_X].astype("category")
    X_test[discrete_columns_of_X].astype("category")

    if classification:
        label_encoder = LabelEncoder()
        Y_train = label_encoder.fit_transform(Y_train)
        Y_test = label_encoder.transform(Y_test)

    oh_encoder = OneHotEncoder(categories=categories, sparse=False)
    oh_encoder.fit(X_train[discrete_columns_of_X])
    X_train = np.concatenate((X_train[numeric_columns_of_X].to_numpy(),
                              oh_encoder.fit_transform(X_train[discrete_columns_of_X])),
                             axis=1)
    X_test = np.concatenate((X_test[numeric_columns_of_X].to_numpy(),
                             oh_encoder.transform(X_test[discrete_columns_of_X])),
                            axis=1)
    if model_type == "xgboost":
        if classification:
            mod = XGBClassifier(
                tree_method=tree_method, enable_categorical=False, use_label_encoder=False, eval_metric="logloss",
                **model_params
            )
        else:
            mod = XGBRegressor(tree_method=tree_method, **model_params)
    elif model_type.lower() in ["lm", "linear model", "linear_model"]:
        mod = LinearRegression(**model_params)
    elif model_type == "lasso":
        mod = Lasso(**model_params)
    elif model_type.lower() in ["lassocv", "lasso_cv", "lasso cv"]:
        mod = LassoCV(**model_params)
    elif model_type.lower() in ["elasticnetcv", "elastic net cv"]:
        mod = ElasticNetCV(**model_params)
    else:
        raise ValueError(f"The model_type {model_type} is not implemented. Try setting the parameter to a different value.")
    # X is the dataframe we created in previous snippet
    mod.fit(X_train, Y_train)
    # # Must use JSON for serialization, otherwise the information is lost
    # mod.save_model("categorical-model.json")

    metrics_result = {}
    if "accuracy" in metrics:
        metrics_result["accuracy"] = accuracy_score(Y_test, mod.predict(X_test))
    if "auc" in metrics:
        metrics_result["auc"] = roc_auc_score(Y_test, mod.predict_proba(X_test)[:, 1])
    if "f1" in metrics:
        metrics_result["f1"] = f1_score(Y_test, mod.predict(X_test), average="macro", labels=[0, 1])
    if "f1_0" in metrics or "f1_1" in metrics:
        f1_both = f1_score(Y_test, mod.predict(X_test), average=None, labels=[0, 1])
        if "f1_0" in metrics:
            metrics_result["f1_0"] = f1_both[0]
        if "f1_1" in metrics:
            metrics_result["f1_1"] = f1_both[1]
    if "mse" in metrics or "rmse" in metrics:
        mse = mean_squared_error(Y_test, mod.predict(X_test))
        metrics_result["mse"] = mse
        metrics_result["rmse"] = np.sqrt(mse)
    if "mae" in metrics:
        metrics_result["mae"] = mean_absolute_error(Y_test, mod.predict(X_test))
        mean_absolute_error

    if retcats:
        return metrics_result, oh_encoder.categories_
    else:
        return metrics_result