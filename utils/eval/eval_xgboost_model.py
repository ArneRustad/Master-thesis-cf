import numpy as np
from xgboost import XGBClassifier
from sklearn.metrics import normalized_mutual_info_score, adjusted_mutual_info_score, accuracy_score, roc_auc_score
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
import tensorflow as tf

def fit_and_evaluate_xgboost(data_train, data_test, categories = "auto", retcats = False, response_col = "income",
                             tree_method=None):
    if tree_method is None:
        if len(tf.config.list_physical_devices("gpu")) > 0:
            tree_method = "gpu_hist"
        else:
            tree_method = "hist"

    X_train = data_train.iloc[:,data_train.columns != response_col]
    Y_train = data_train[response_col]
    X_test = data_test.iloc[:,data_train.columns != response_col]
    Y_test = data_test[response_col]

    columns_discrete_bool = [str(b) in ["object", "category"] for b in X_train.dtypes]
    discrete_columns_of_X = X_train.columns[columns_discrete_bool]
    numeric_columns_of_X = X_train.columns[np.logical_not(columns_discrete_bool)]


    X_train[discrete_columns_of_X].astype("category")
    X_test[discrete_columns_of_X].astype("category")

    label_encoder = LabelEncoder()
    Y_train = label_encoder.fit_transform(Y_train)
    Y_test = label_encoder.transform(Y_test)

    #categories = OneHotEncoder().fit(X_train[discrete_columns_of_X]).categories_
    oh_encoder = OneHotEncoder(categories = categories, sparse = False)
    X_train_discr = oh_encoder.fit_transform(X_train[discrete_columns_of_X])
    X_train = np.concatenate((X_train[numeric_columns_of_X].to_numpy(),
                              oh_encoder.fit_transform(X_train[discrete_columns_of_X])),
                             axis = 1)
    X_test = np.concatenate((X_test[numeric_columns_of_X].to_numpy(),
                             oh_encoder.transform(X_test[discrete_columns_of_X])),
                            axis = 1)

    clf = XGBClassifier(
        tree_method=tree_method, enable_categorical=False, use_label_encoder=False, eval_metric = "logloss"
    )
    # X is the dataframe we created in previous snippet
    clf.fit(X_train, Y_train)
    # Must use JSON for serialization, otherwise the information is lost
    clf.save_model("categorical-model.json")
    clf.predict(X_test)

    accuracy = accuracy_score(Y_test, clf.predict(X_test))
    auc = roc_auc_score(Y_test, clf.predict_proba(X_test)[:,1])

    if retcats:
        return accuracy, auc, oh_encoder.categories_
    else:
        return accuracy, auc