import pandas as pd
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import warnings
import os

from sklearn.preprocessing import StandardScaler, OneHotEncoder, QuantileTransformer
from sklearn.compose import ColumnTransformer

from tensorflow.keras.models import Model
from tensorflow.keras.layers import (Input, Dense, Flatten, concatenate, Activation)

from .tabGAN import TabGAN
from .fast_nondominated_sort import fast_non_dominated_sort

import utils.plot

class TabGANcf(TabGAN):
    def __init__(self, data_train, classifier, only_gen_class=None, only_want_class=None,
                 query_critic_instance=True,
                 query_critic_classifier_label=False, query_generator_classifier_label=False,
                 classifier_adjusted_wgan_loss_term=False,
                 critic_loss_mix_real_correct_samples_with_generated=False,
                 critic_loss_mix_real_correct_samples_with_generated_coef=1,
                 critic_loss_mix_real_wrong_samples_with_generated=False,
                 critic_loss_mix_real_wrong_samples_with_generated_coef=1,
                 generator_1norm_num_loss_coef=1,
                 generator_2norm_num_loss_coef=0,
                 generator_add_bernoulli_new_vs_query=False,
                 **kwargs):

        super().__init__(data=data_train,
                         use_query=True, ctgan=False,
                         **kwargs
                         )
        if only_gen_class is not None and only_gen_class not in [0, 1]:
            raise ValueError("The parameter only_gen_class only allows values None, 0 or 1."
                             f" You entered {only_gen_class}")
        if only_want_class is not None and only_want_class not in [0, 1]:
            raise ValueError("The parameter only_want_class only allows values None, 0 or 1."
                             f" You entered {only_want_class}")

        if only_want_class is not None and only_gen_class is not None:
            raise ValueError("The parameter only_want_class!=None should not be used in combination with parameter" 
                             " only_gen_class!=None")
        if only_want_class is not None and not classifier_adjusted_wgan_loss_term:
            import logging
            warnings.warn("The parameter only_want_class will have little effect in learning when not used in" 
            " combination with classifier_adjusted_wgan_loss_term=True")

        self.n_columns_query = self.n_columns_num + self.n_columns_discrete_oh
        self.classifier = classifier
        self.query_critic_classifier_label = query_critic_classifier_label
        self.query_generator_classifier_label = query_generator_classifier_label
        self.query_critic_instance = query_critic_instance
        self.only_gen_class = only_gen_class
        self.classifier_adjusted_wgan_loss_term = classifier_adjusted_wgan_loss_term
        self.only_want_class = 1
        self.critic_loss_mix_real_correct_samples_with_generated = critic_loss_mix_real_correct_samples_with_generated
        self.critic_loss_mix_real_wrong_samples_with_generated = critic_loss_mix_real_wrong_samples_with_generated

        self.classifier_label = tf.round(tf.reshape(self.classifier(self.data), shape=(self.nrow, 1)))
        self.data_ix_classifier_0 = np.where(self.classifier_label <= 0.5)[0].flatten()
        self.data_ix_classifier_1 = np.where(self.classifier_label > 0.5)[0].flatten()
        # self.data_ix_classifier_0 = tf.squeeze(tf.where(self.classifier_label <= 0.5))
        # self.data_ix_classifier_1 = tf.squeeze(tf.where(self.classifier_label > 0.5))
        self.wanted_classifier_label = 1 - self.classifier_label
        self.wanted_classifier_label = self.wanted_classifier_label.numpy()
        pos_queries_used_by_critic = np.where([self.query_critic_instance, self.query_critic_classifier_label])[0]
        self.pos_queries_used_by_critic = pos_queries_used_by_critic
        print(self.pos_queries_used_by_critic)

        self.generator_1norm_num_loss_coef = generator_1norm_num_loss_coef
        self.generator_2norm_num_loss_coef = generator_2norm_num_loss_coef
        self.generator_add_bernoulli_new_vs_query = generator_add_bernoulli_new_vs_query
        self.critic_loss_mix_real_correct_samples_with_generated_coef = \
            critic_loss_mix_real_correct_samples_with_generated_coef
        self.critic_loss_mix_real_wrong_samples_with_generated_coef = \
            critic_loss_mix_real_wrong_samples_with_generated_coef


        if len(self.pos_queries_used_by_critic) == 0:
            self.critic_use_query_input = False



    def generate_queries(self, n, plot2D_test_samples=False):
        if plot2D_test_samples and self.only_want_class is not None:
            only_gen_class = self.only_want_class
        else:
            only_gen_class = self.only_gen_class

        if only_gen_class is None:
            ix = tf.random.uniform(shape=[n], minval=0, maxval=self.data.shape[0], dtype=tf.int64)
        elif only_gen_class == 0:
            ix_indices = tf.random.uniform(shape=[n], minval=0, maxval=self.data_ix_classifier_1.shape[0],
                                        dtype=tf.int64)
            ix = tf.squeeze(tf.py_function(np.vectorize(lambda ixs: self.data_ix_classifier_1[ixs]),
                                           inp=[ix_indices], Tout=tf.int64))
        elif only_gen_class == 1:
            ix_indices = tf.random.uniform(shape=[n], minval=0, maxval=self.data_ix_classifier_0.shape[0],
                                           dtype=tf.int64)
            ix = tf.squeeze(tf.py_function(np.vectorize(lambda ixs: self.data_ix_classifier_0[ixs]),
                                           inp=[ix_indices], Tout=tf.int64))
        else:
            raise ValueError(f"The parameter only_gen_class must be one of'None', 1 or 2. You entered {only_gen_class}")

        # queries = tf.py_function(lambda ixs: [self.data_num_scaled[ixs, ], self.data_discrete_oh[ixs, ]],
        #                          inp=[ix], Tout=[tf.float32, tf.float32])
        # ix = tf.random.uniform(shape=[n], minval=0, maxval=self.data_ix_classifier_0.shape[0], dtype=tf.int64)
        # ix = tf.py_function(lambda ixs: self.data_ix_classifier_0[ixs.numpy()],
        #                     inp=[ix], Tout=[tf.float32])
        queries = tf.concat(
            tf.py_function(lambda ixs: [self.data_num_scaled[ixs, ], self.data_discrete_oh[ixs, ]],
                           inp=[ix], Tout=[tf.float32, tf.float32]),
            axis=1
        )
        #if self.query_critic_classifier_label or self.query_generator_classifier_label:
        wanted_classifier_label = tf.reshape(
            tf.py_function(np.vectorize(lambda ixs: self.wanted_classifier_label[ixs]), inp=[ix], Tout=tf.float32),
            shape=(n, 1))
        return [queries, wanted_classifier_label]

    def _python_get_numpy_data_batch_real_from_queries(self, queries, wanted_classifier_label):
        n_batch_1 = np.sum(wanted_classifier_label > 0.5)
        n_batch_0 = np.sum(wanted_classifier_label < 0.5)

        ix_batch_0 = np.random.choice(self.data_ix_classifier_0, n_batch_0)
        ix_batch_1 = np.random.choice(self.data_ix_classifier_1, n_batch_1)

        ix = np.empty(shape=queries.shape[0], dtype=int)
        ix[wanted_classifier_label <= 0.5] = ix_batch_0
        ix[wanted_classifier_label > 0.5] = ix_batch_1
        ix = ix.astype(int)
        
        return_object = [self.data_num_scaled[ix, ], self.data_discrete_oh[ix, ]]
        return return_object

    def get_numpy_data_batch_real_from_queries(self, queries):
        return tf.py_function(self._python_get_numpy_data_batch_real_from_queries,
                              inp=[queries[0], tf.squeeze(queries[1])], Tout=[tf.float32, tf.float32])

    def calc_loss_critic(self, real_output, fake_output, data_batch_real, queries_real, critic_tape):
        if self.classifier_adjusted_wgan_loss_term:
            # Classifier adjusted loss term for WGAN
            with critic_tape.stop_recording():
                # classifier_score_real_data = tf.squeeze(queries_real[1])
                classifier_score_real_data = self.eval_classifier_on_transformed_data(
                    data_num_scaled=data_batch_real[0], data_discrete_oh=data_batch_real[1])
                # tf.print(classifier_score_real_data)
                # tf.print(tf.squeeze(queries_real[1]))
                if self.only_want_class is None:
                    classifier_score_real_data = tf.where(wanted_classifier_label <= 0.5,
                                                       1 - classifier_score_real_data, classifier_score_real_data)
                elif self.only_want_class == 0:
                    classifier_score_real_data = 1 - classifier_score_real_data
                elif self.only_want_class == 1:
                    pass
                else:
                    raise ValueError(f"Not valid input for parameter only_want_class: {self.only_want_class}")
                #tf.print(classifier_val_gen_data)
                sum_classifier_score_real_data = tf.reduce_sum(classifier_score_real_data)
            loss = - tf.reduce_sum(classifier_score_real_data * tf.squeeze(real_output) / sum_classifier_score_real_data)
        else:
            # Standard WGAN loss term
            loss = - tf.reduce_mean(real_output)

        critic_loss_gen_part_normalizer = 1
        loss_from_generator_samples = 0
        if self.critic_loss_mix_real_wrong_samples_with_generated:
            loss_from_generator_samples += (
                    self.critic_loss_mix_real_wrong_samples_with_generated_coef *
                    tf.reduce_mean(self.critic([self.split_transformed_data(queries_real[0]),
                                                [tf.concat(data_batch_real, axis=1),
                                                 queries_real[1]]
                                                ]))
                     )
            critic_loss_gen_part_normalizer += self.critic_loss_mix_real_wrong_samples_with_generated_coef

        if self.critic_loss_mix_real_correct_samples_with_generated:
            with critic_tape.stop_recording():
                extra_correct_samples = self.get_numpy_data_batch_real_from_queries(queries_real)
            loss_from_generator_samples += (self.critic_loss_mix_real_correct_samples_with_generated_coef *
                     tf.reduce_mean(self.critic([extra_correct_samples, queries_real]))
                     )
            critic_loss_gen_part_normalizer += self.critic_loss_mix_real_correct_samples_with_generated_coef
        loss_from_generator_samples += tf.reduce_mean(fake_output)
        loss += loss_from_generator_samples / critic_loss_gen_part_normalizer
        return loss

    def calc_loss_generator(self, fake_output, gen_data_num, gen_data_discrete, queries, gen_tape):
        query_instances_num = queries[0][:, :self.n_columns_num]
        query_instances_discrete = queries[0][:, self.n_columns_num:]

        # Standard WGAN loss term
        loss = - tf.reduce_mean(fake_output)
        loss_vec = [loss]
        # Distance 1 norm numeric loss term
        loss += (self.generator_1norm_num_loss_coef *
                 tf.reduce_mean(tf.reduce_mean(tf.abs(query_instances_num - gen_data_num), axis=1)))
        # Distance 2 norm numeric loss term
        loss += (self.generator_2norm_num_loss_coef *
                 tf.reduce_mean(tf.sqrt(tf.reduce_mean((query_instances_num - gen_data_num)**2, axis=1))))
        # Discrete columns loss term
        loss += tf.reduce_mean(tf.reduce_sum(tf.abs(query_instances_discrete - gen_data_discrete), axis=1))
        loss_vec = tf.concat((loss_vec, [loss - loss_vec[0]]), axis=-1)
        #tf.print(loss_vec)
        return loss

    def _eval_classifier_on_transformed_data(self, data_num_scaled, data_discrete_oh):
        data = self.inv_data_transform(data_num_scaled, data_discrete_oh)
        return tf.cast(self.classifier(data), tf.float32)

    def eval_classifier_on_transformed_data(self, data_num_scaled, data_discrete_oh):
        # return tf.reshape(
        #     tf.py_function(self._eval_classifier_on_transformed_data,
        #                    inp=[data_num_scaled, data_discrete_oh], Tout=tf.float32),
        #     shape=[data_num_scaled.shape[0]])
        return tf.py_function(self._eval_classifier_on_transformed_data,
                           inp=[data_num_scaled, data_discrete_oh], Tout=tf.float32)



    def train_step_func(self, n_batch, ret_loss=False):
        # prev_time = tf.timestamp()
        for i in range(self.n_critic):
            queries = self.generate_queries(n_batch)
            #curr_time = tf.timestamp(); tf.print("Fetching queries:", curr_time - prev_time); prev_time=curr_time
            data_batch_real = self.get_numpy_data_batch_real_from_queries(queries)
            #curr_time = tf.timestamp(); tf.print("Fetching data real:", curr_time - prev_time); prev_time=curr_time
            self.train_step_critic(data_batch_real, n_batch, queries=queries)
            #curr_time = tf.timestamp(); tf.print("Performing train step critic:", curr_time - prev_time); prev_time=curr_time

        queries = self.generate_queries(n_batch)
        #curr_time = tf.timestamp(); tf.print("Fetching queries:", curr_time - prev_time); prev_time=curr_time
        self.train_step_generator(n_batch, queries=queries)
        #curr_time = tf.timestamp(); tf.print("Performing train step generator:", curr_time - prev_time); prev_time=curr_time

        if ret_loss:
            return 0, 0
        else:
            return None, None

    def create_critic(self):
        """
        Internal function for creating the critic neural network. Uses input parameters given to TabGAN to decide
        between different critic architectures
        """
        input_numeric = Input(shape=(self.n_columns_num * self.pac), name="Numeric_input")
        input_discrete = Input(shape=(self.n_columns_discrete_oh * self.pac), name="Discrete_input")
        input_instance = Input(shape=((self.n_columns_num + self.n_columns_discrete_oh) * self.pac), name="Query_instance")
        input_classifier_label = Input(shape=(1 * self.pac), name="Query_classifier_label")
        queries = [input_instance, input_classifier_label]
        queries_combined = []
        if self.query_critic_instance:
            queries_combined += [input_instance]

        if self.query_critic_classifier_label:
            queries_combined += [input_classifier_label]

        if len(queries_combined) == 0:
            combined1 = concatenate([input_numeric, input_discrete], name="Combining_input")
        else:
            if len(queries_combined) > 1:
                queries_combined = concatenate(queries_combined, name="Combining_queries")
            else:
                queries_combined = queries_combined[0]
            combined1 = concatenate([input_numeric, input_discrete, queries_combined], name="Combining_input")
        inputs = [[input_numeric, input_discrete], queries]
        hidden = combined1
        if 0 in self.add_dropout_critic:
            hidden = Dropout(rate=self.dropout_rate_critic, name=f"Dropout0")(hidden)
        for i in range(self.n_hidden_critic_layers):
            hidden = Dense(self.dim_hidden_critic[i], activation=self.activation_function,
                           name=f"hidden{i+1}")(hidden)
            if (i+1) in self.add_dropout_critic:
                hidden = Dropout(rate=self.dropout_rate_critic, name=f"Dropout{i+1}")(hidden)
        output = Dense(1, name="output_critic")(hidden)
        model = Model(inputs=inputs, outputs=output)
        return model

    def create_generator(self):
        """
        Internal function for creating the generator neural network. Uses input parameters given to TabGAN to decide
        between different critic architectures. Also uses the number of discrete columns to decide between architectures
        """
        if self.n_columns_num == 0:
            raise ValueException("TabGAN not yet implemented for zero numerical columns")

        latent = Input(shape=self.dim_latent, name="Latent")
        input_instance = Input(shape=((self.n_columns_num + self.n_columns_discrete_oh) * self.pac), name="Query_instance")
        input_classifier_label = Input(shape=(1 * self.pac), name="Query_classifier_label")
        queries = [input_instance, input_classifier_label]
        combined_queries = [input_instance]

        if self.query_generator_classifier_label:
            combined_queries += [input_classifier_label]
        else:
            combined_queries = queries

        if self.query_generator_classifier_label:
            combined_queries = concatenate(combined_queries, name="Concatenate_queries")
        else:
            combined_queries = combined_queries[0]
        combined1 = concatenate([latent, combined_queries], name="Concatenate_input")
        inputs = [latent, queries]

        hidden = combined1
        if 0 in self.add_dropout_generator:
            hidden = Dropout(rate=self.dropout_rate_generator, name=f"Dropout0")(hidden)
        for i in range(self.n_hidden_generator_layers):
            hidden = Dense(self.dim_hidden_generator[i], activation=self.activation_function,
                           name=f"hidden{i+1}")(hidden)
            if (i+1) in self.add_dropout_generator:
                hidden = Dropout(rate=self.dropout_rate_generator, name=f"Dropout{i+1}")(hidden)

        if not self.add_connection_discrete_to_num:
            output_numeric = Dense(self.n_columns_num, name="Numeric_output")(hidden)
        if self.add_connection_num_to_discrete:
            if self.dim_hidden_layer_num_to_discrete > 0:
                numeric_to_discrete = Dense(self.dim_hidden_layer_num_to_discrete,
                                            name="Hidden_numeric_to_discrete")(output_numeric)
            else:
                numeric_to_discrete = output_numeric
            potential_concat_hidden_and_num = concatenate((hidden, numeric_to_discrete),
                                                          name="Concatenate_hidden_and_numeric")
        else:
            potential_concat_hidden_and_num = hidden

        if self.use_query and self.add_connection_query_to_discrete:
            potential_concat_hidden_and_num = concatenate((potential_concat_hidden_and_num,
                                                           query),
                                                          name="Concatenate_hidden_and_query")

        if self.n_columns_discrete == 0:
            raise ValueException("TabGAN not yet implemented for zero discrete columns")
        else:
            output_discrete_sep = []
            for i in range(self.n_columns_discrete):
                output_discrete_i = Dense(self.categories_len[i],
                                          name="%s_output" % self.columns_discrete[i])(potential_concat_hidden_and_num)
                output_discrete_sep.append(
                    Activation("gumbel_softmax", name="Gumbel_softmax%d" % (i + 1))(output_discrete_i))

            if self.n_columns_discrete > 1:
                output_discrete = concatenate(output_discrete_sep, name="Discrete_output")
            else:
                output_discrete = output_discrete_sep[0]


        if self.add_connection_discrete_to_num:
            if self.dim_hidden_layer_discrete_to_num > 0:
                discrete_to_numeric = Dense(self.dim_hidden_layer_discrete_to_num,
                                            name="Hidden_discrete_to_numeric")(output_discrete)
            else:
                discrete_to_numeric = output_discrete
            concatenate_hidden_and_discrete = concatenate((hidden, discrete_to_numeric),
                                                          name="Concatenate_hidden_and_discrete")
            output_numeric = Dense(self.n_columns_num, name="Numeric_output")(concatenate_hidden_and_discrete)

        if self.generator_add_bernoulli_new_vs_query:
            pass

        model = Model(inputs=inputs, outputs=[output_numeric, output_discrete])
        return model

    def _sample_interpret_input(self, n, queries, n_repeat):
        if n_repeat is not None:
            raise ValueError("n_repeat different from 1 is not yet implemented for tabGAN-cf")

        if n is None:
            if queries is None:
                n = self.nrow
            else:
                n = queries[0].shape[0]
        else:
            if queries is not None and n != queries.shape[0]:
                raise ValueError("If parameter n is set in addition to queries and/or n_repeat, "
                                 "then n must be equal to length of queries times n_repeat (if n_repeat set)")
        return n, queries

    def _sample_get_gen_input(self, n, queries):
        noise = self.generate_latent(n)
        if queries is None:
            queries = self.generate_queries(n)
        gen_input = [noise, queries]
        return gen_input

    def plot2D_axis_update(self, ax, latent_vec, num_cols, discrete_col=None, queries=None, inv_scale=True,
                           legend=True, color_opacity=1, color_map=None):
        gen_data_num, gen_data_discrete = self.generator([latent_vec, queries] if self.use_query
                                                         else [latent_vec])
        color_gen = None
        color_queries = None

        x1_lim = np.array([np.min(self.data_num.loc[:, num_cols[0]]),
                           np.max(self.data_num.loc[:, num_cols[0]])])
        x2_lim = np.array([np.min(self.data_num.loc[:, num_cols[0]]),
                           np.max(self.data_num.loc[:, num_cols[0]])])
        utils.plot.heatmap2D_classifier(ax, self.classifier, x1_lim=x1_lim, x2_lim=x2_lim)

        if inv_scale:
            gen_data = self.inv_data_transform(gen_data_num, gen_data_discrete)
            queries = self.inv_data_transform(*self.split_transformed_data(queries[0]))
            if discrete_col is None:
                color_dict = {"all": next(ax._get_lines.prop_cycler)['color']}
            else:
                labels_unique = np.sort(np.unique([gen_data[discrete_col], queries[discrete_col]]))
                if color_map is None:
                    colors_unique = [next(ax._get_lines.prop_cycler)['color'] for label in labels_unique]
                else:
                    cmap = matplotlib.cm.get_cmap(color_map)
                    colors_unique = [color for color in cmap(np.linspace(0, 1, len(labels_unique)))]
                color_dict = {label: color for label, color in zip(labels_unique, colors_unique)}
                color_gen = gen_data[discrete_col].map(color_dict)
                color_queries = queries[discrete_col].map(color_dict)

                if legend:
                    markers = [plt.Line2D([0, 0], [0, 0], color=color, marker='o', linestyle='', alpha=0.5)
                               for color in color_dict.values()]
                    ax.legend(markers, color_dict.keys(), numpoints=1)

            class_gen_data = np.where(self.classifier(gen_data) > 0.5, 1, 0).astype(int)
            class_queries = np.where(self.classifier(queries) > 0.5, 1, 0).astype(int)
        else:
            gen_data = pd.DataFrame(gen_data_num, columns=self.columns_num)
            queries = pd.DataFrame(self.split_transformed_data(queries[0])[0], columns=self.columns_num)
        ax.scatter(queries[num_cols[0]], queries[num_cols[1]], c=color_queries, alpha=color_opacity,
                   edgecolors="black")
        ax.scatter(gen_data[num_cols[0]], gen_data[num_cols[1]], c=color_gen, alpha=color_opacity)
        ax.quiver(queries[num_cols[0]], queries[num_cols[1]],
                  gen_data[num_cols[0]] - queries[num_cols[0]],
                  gen_data[num_cols[1]] - queries[num_cols[1]],
                  angles='xy', scale_units='xy', scale=1,
                  color=np.where(class_gen_data != class_queries, "yellow", "black"))

    def plot2D_classifier(self, numeric_cols, add_observations=False, add_counterfactuals=False,
                          n_counterfactuals=None, n_observations=None, seed=None,
                          counterfactual_queries=None,
                          save_dir=None, save_path=None, observation_opacity=1,
                          figsize=[12, 8], expand_limits=0.2,
                          print_original_observations=False, print_counterfactuals=False):
        if n_counterfactuals is None:
            if counterfactual_queries is None:
                n_counterfactuals = 10
            else:
                n_counterfactuals = queries_counterfactuals[0].shape[0]

        if n_observations is None:
            n_observations = self.data.shape[0]

        x1_lim = np.array([np.min(self.data_num.loc[:, numeric_cols[0]]),
                           np.max(self.data_num.loc[:, numeric_cols[0]])])
        x2_lim = np.array([np.min(self.data_num.loc[:, numeric_cols[1]]),
                           np.max(self.data_num.loc[:, numeric_cols[1]])])

        def expand_fraction(limits, fraction):
            avg_lim, half_range_lim = np.mean(limits), (limits[1] - limits[0]) / 2
            return [avg_lim - half_range_lim * (1 + fraction), avg_lim + half_range_lim * (1 + fraction)]


        fig, ax = plt.subplots(figsize=figsize)
        utils.plot.heatmap2D_classifier(ax, self.classifier, x1_lim=expand_fraction(x1_lim, expand_limits),
                                        x2_lim=expand_fraction(x2_lim, expand_limits))

        if add_observations:
            data_num_subset = self.data_num.loc[:n_observations,:]
            obs_classifier_vals = self.classifier(data_num_subset.loc[:, numeric_cols])
            plt.scatter(data_num_subset.loc[:, numeric_cols[0]], data_num_subset.loc[:, numeric_cols[1]],
                        c=np.where(obs_classifier_vals > 0.5, "maroon", "darkblue"),
                        alpha=observation_opacity)

        if add_counterfactuals:
            if counterfactual_queries is None:
                queries_counterfactuals = self.generate_queries(n=n_counterfactuals, plot2D_test_samples=True)
            else:
                queries_counterfactuals = [counterfactual_queries[0][:n_counterfactuals, :],
                                           counterfactual_queries[1][:n_counterfactuals, :]
                                           ]
            counterfactuals = self.sample(queries=queries_counterfactuals)
            counterfactuals_classifier_vals = self.classifier(counterfactuals)
            counterfactuals_classifier_class = np.where(counterfactuals_classifier_vals > 0.5, 1, 0)
            original_observations = self.inv_data_transform(*self.split_transformed_data(queries_counterfactuals[0]))
            original_observations_classifiers_vals = self.classifier(original_observations)
            original_observations_classifiers_class = np.where(original_observations_classifiers_vals > 0.5, 1, 0)
            ax.quiver(original_observations[numeric_cols[0]], original_observations[numeric_cols[1]],
                      counterfactuals[numeric_cols[0]] - original_observations[numeric_cols[0]],
                      counterfactuals[numeric_cols[1]] - original_observations[numeric_cols[1]],
                      angles='xy', scale_units='xy', scale=1, width=0.005,
                      color=np.where(counterfactuals_classifier_class != original_observations_classifiers_class,
                                     "yellow", "magenta")
                      )

            if print_original_observations:
                print(original_observations.to_latex())
            if print_counterfactuals:
                print(counterfactuals.to_latex())
            # tf.print(np.sum(original_observations["dummy"] == counterfactuals["dummy"]))

        if save_path is not None:
            if save_dir is not None:
                save_path = os.path.join(save_dir, save_path)
            fig.savefig(save_path)



