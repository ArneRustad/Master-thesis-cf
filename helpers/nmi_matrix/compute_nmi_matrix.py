import pandas as pd
import numpy as np

def compute_nmi_matrix(tgan = None, dataset = None, bins = None, n_q_bins = 40
                       , generated_data = True, retbins = False, average_method = "arithmetic",
                       n_samples = None):
    if n_samples is None:
        n_samples = tgan.data.shape[0]

    if dataset is None:
        if (generated_data):
            data_binned = tgan.generate_data(n_samples)
        else:
            data_binned = tgan.data.copy()
    else:
        data_binned = dataset.copy()
    if (retbins):
        bins_curr = {}
    for col_num in tgan.columns_num:
        if bins == None:
            cut_series, cut_bins = pd.qcut(data_binned[col_num] , q = n_q_bins, retbins=True, duplicates = "drop")
        else:
            cut_series, cut_bins = pd.cut(data_binned[col_num], bins = bins[col_num], retbins = True,
                                          include_lowest = True)
        data_binned[col_num] = cut_series
        if (retbins):
            bins_curr[col_num] = cut_bins

    if average_method == "arithmetic":
        average_func = lambda x,y : np.mean([x,y])
    elif average_method == "max":
        average_func = lambda x, y : np.max([x,y])
    elif average_method == "min":
        average_func = lambda x, y : np.min([x,y])
    elif average_method == "geometric":
        average_func = lambda x, y : np.sqrt(x * y)
    else:
        raise ValueError("Average_method given as input is not implemented")

    probs_dict = {}
    entropy = np.zeros(tgan.n_columns)
    for i,col in enumerate(tgan.columns):
        col_category_fractions = data_binned[col].value_counts(normalize = True)
        probs_dict[col] = col_category_fractions.to_dict()
        entropy[i] = np.sum(- col_category_fractions * np.log(col_category_fractions))

    nmi_matrix = np.zeros([tgan.n_columns, tgan.n_columns])
    for i,col1 in enumerate(tgan.columns):
        for j,col2 in enumerate(tgan.columns):
            if j < i:
                continue
            elif i == j:
                nmi_matrix[i,j] = 1
                continue
            df_curr_cols = data_binned[[col1,col2]].copy()
            df_curr_cols_fraction = df_curr_cols.groupby([col1,col2]).size().reset_index().rename(columns={0:"Prob.both"})
            df_curr_cols_fraction["Prob.both"] /= data_binned.shape[0]
            df_curr_cols_fraction["Prob.col1"] = df_curr_cols_fraction[col1].map(probs_dict[col1]).astype(float)
            df_curr_cols_fraction["Prob.col2"] = df_curr_cols_fraction[col2].map(probs_dict[col2]).astype(float)
            df_curr_cols_fraction["NMI"] = df_curr_cols_fraction["Prob.both"] * np.log(df_curr_cols_fraction["Prob.both"]/
                                                                                       (df_curr_cols_fraction["Prob.col1"]*df_curr_cols_fraction["Prob.col2"]))
            nmi_matrix[i,j] = nmi_matrix[j,i] = np.sum(df_curr_cols_fraction["NMI"]) / average_func(entropy[i], entropy[j])

    if retbins:
        return nmi_matrix, bins_curr
    else:
        return nmi_matrix
