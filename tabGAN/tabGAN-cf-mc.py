import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, OneHotEncoder, QuantileTransformer
from sklearn.compose import ColumnTransformer

from .tabGAN import tableGAN
from .fast_nondominated_sort import fast_non_dominated_sort

class TabGANcfmc(tableGAN):
    def use_critic_on_data(self, data):
        """
        Internal function for preprocessing data and then fetching it to the critic. Mostly used for debugging purposes.
        """
        num_data = data[self.columns_num]
        discrete_data = data[self.columns_discrete]
        num_data_scaled = self.scaler_num.transform(num_data)
        discrete_data_oh = self.oh_encoder.transform(discrete_data)
        queries_batch = tf.zeros([data.shape[0], self.n_columns_discrete_oh], dtype=tf.dtypes.float32)
        return (self.critic.predict([num_data_scaled, discrete_data_oh, queries_batch]))

    def generate_counterfactuals(self, n_to_keep, pred_func, x_obs, wanted_range=None, n_to_generate=None,
                                 add_plausibility_objective=False, epsilon_num_percent=0.005, return_objectives=True):
        """
        Function for generating counterfactuals
        """
        if n_to_generate is None:
            n_to_generate = n_to_keep * 1000
        if wanted_range is None:
            wanted_label = 1 - np.round(pred_func(x_obs))
            wanted_range = np.sort([0.5, wanted_label])

        gen_data = self.generate_data(n_to_generate)
        pred_gen_data = pred_func(gen_data)
        gen_data = gen_data.loc[(pred_gen_data >= wanted_range[0]) & (pred_gen_data <= wanted_range[1])].reset_index(
            drop=True)
        if gen_data.shape[0] == 0:
            raise RuntimeError("None of the generated observations had a prediction value in the wanted range")

        n_objectives = 2
        objective_names = ["Gower distance", "Number changed"]
        if add_plausibility_objective:
            n_objectives += 1
            objective_names += ["Plausibility"]
        objectives = np.zeros([gen_data.shape[0], n_objectives])
        range_num_values_dict = {}
        for i, col_num in enumerate(self.columns_num):
            range_num_values_dict[col_num] = np.max(self.data[col_num]) - np.min(self.data[col_num])

        for i, col_num in enumerate(self.columns_num):
            objectives[:, 0] += np.abs(gen_data[col_num].to_numpy() - x_obs[col_num].to_numpy()) / \
                                range_num_values_dict[col_num]
            objectives[:, 1] += np.where(np.isclose(gen_data[col_num], x_obs[col_num],
                                                    atol=epsilon_num_percent * range_num_values_dict[col_num]), 0, 1)

        for i, col_discrete in enumerate(self.columns_discrete):
            binary_is_cat_changed = np.where(gen_data[col_discrete].to_numpy() == x_obs[col_discrete].to_numpy(), 0, 1)
            objectives[:, 0] += binary_is_cat_changed
            objectives[:, 1] += binary_is_cat_changed

        objectives[:, 0] /= self.n_columns

        if add_plausibility_objective:
            raise ValueError("Not yet implemented")

        rank_list = fast_non_dominated_sort(objectives, minimize=True)
        n_each_rank = [len(rank_group) for rank_group in rank_list]
        cumsum_each_rank = np.cumsum(n_each_rank)
        if cumsum_each_rank[-1] < n_to_keep:
            raise RuntimeError(
                "Not enough valid counterfactuals were generated," \
                "increase the parameter n_to_generate to decrease the chance of this happening."
            )
        last_needed_rank = np.where(np.greater_equal(cumsum_each_rank, n_to_keep))[0][0]
        rank_list = rank_list[:last_needed_rank + 1]

        ranking = [index for rank_group in rank_list for index in rank_group]

        gen_data = gen_data.iloc[ranking].head(n_to_keep).reset_index(drop=True)
        if return_objectives:
            df_objectives = pd.DataFrame(objectives[ranking[:n_to_keep],], columns=objective_names)
            gen_data = gen_data.join(df_objectives)
        return gen_data
