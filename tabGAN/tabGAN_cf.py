import pandas as pd
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler, OneHotEncoder, QuantileTransformer
from sklearn.compose import ColumnTransformer

from tensorflow.keras.models import Model
from tensorflow.keras.layers import (Input, Dense, Flatten, concatenate, Activation)

from .tabGAN import TabGAN
from .fast_nondominated_sort import fast_non_dominated_sort

class TabGANcf(TabGAN):
    def __init__(self, data_train, classifier, only_gen_class=None,
                 query_critic_instance=True,
                 query_critic_classifier_label=False, query_generator_classifier_label=False,
                 **kwargs):

        super().__init__(data=data_train,
                         use_query=True, ctgan=False,
                         **kwargs
                         )
        if only_gen_class is not None and only_gen_class not in [0, 1]:
            raise ValueError("The parameter only_gen_class only allows values None, 0 or 1."
                             f" You entered {only_gen_class}")
        self.n_columns_query = self.n_columns_num + self.n_columns_discrete_oh
        self.classifier = classifier
        self.query_critic_classifier_label = query_critic_classifier_label
        self.query_generator_classifier_label = query_generator_classifier_label
        self.query_critic_instance = query_critic_instance

        self.classifier_label = np.array(self.classifier(self.data))
        print(self.classifier_label)
        self.data_ix_classifier_0 = np.where(self.classifier_label <= 0.5)[0]
        self.data_ix_classifier_1 = np.where(self.classifier_label > 0.5)[0]

    def generate_queries(self, n):
        ix = tf.random.uniform(shape=[n], minval=0, maxval=self.data.shape[0], dtype=tf.int64)
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
        if self.query_critic_classifier_label:
            classifier_label = tf.py_function(lambda ixs: self.classifier_label[ixs],
                                              inp=[ix], Tout=[tf.float32, tf.float32])
            queries = [queries, classifier_label]
        else:
            queries = [queries]
        return queries

    def _python_get_numpy_data_batch_real_from_queries(self, queries):
        classifier_values = self.classifier(
            self.inv_data_transform(*self.split_transformed_data(queries[0]))
        )
        classifier_label = np.round(classifier_values).astype(np.int64)

        n_batch_1 = np.sum(classifier_label == 1)
        n_batch_0 = np.sum(classifier_label == 0)

        ix_batch_0 = np.random.choice(self.data_ix_classifier_0, n_batch_0)
        ix_batch_1 = np.random.choice(self.data_ix_classifier_1, n_batch_1)

        ix = np.empty(shape=queries[0].shape[0], dtype=np.int64)
        ix[classifier_label == 0] = ix_batch_0
        ix[classifier_label == 1] = ix_batch_1
        ix = ix.astype(np.int64)

        return [self.data_num_scaled[ix, ], self.data_discrete_oh[ix, ]]

    def get_numpy_data_batch_real_from_queries(self, queries):
        return tf.py_function(self._python_get_numpy_data_batch_real_from_queries,
                              inp=[queries], Tout=[tf.float32, tf.float32])

    def calc_loss_generator(self, fake_output, gen_data_num, gen_data_discrete, queries):
        query_instances_num = queries[0][:, :self.n_columns_num]
        query_instances_discrete = queries[0][:, self.n_columns_num:]
        # Standard WGAN loss term
        loss = - tf.reduce_mean(fake_output)
        loss_vec = [loss]
        # Distance 1 norm loss term
        loss += tf.reduce_mean(tf.reduce_mean(tf.abs(query_instances_num - gen_data_num), axis=1))
        loss += tf.reduce_mean(tf.reduce_mean(tf.abs(query_instances_discrete - gen_data_discrete), axis=1))
        loss_vec = tf.concat((loss_vec, [loss - loss_vec[0]]), axis=-1)
        # Classifier loss term
        # queries_inv_transformed = tf.py_function(lambda query_num, query_discrete:
        #                                          self.inv_data_transform(query_num, query_discrete),
        #                                          inp=[query_instances_num, uery_instances_discrete],
        #                                          Tout=[tf.float32, tf.float32])
        # class_queries = tf.cast(
        #     tf.where(self.eval_classifier_on_transformed_data(query_instances_num, query_instances_discrete) < 0.5, 0, 1),
        #     tf.int32
        # )
        # classifier_val_gen_data =  self.eval_classifier_on_transformed_data(gen_data_num, gen_data_discrete)
        # classifier_val_gen_data = tf.where(class_queries == 1, 1 - classifier_val_gen_data, 0)
        # loss += tf.reduce_sum(classifier_val_gen_data * fake_output / classifier_val_gen_data)
        loss_vec = tf.concat((loss_vec, [loss - tf.reduce_sum(loss_vec)]), axis=-1)

        tf.print(loss_vec)
        return loss

    def _eval_classifier_on_transformed_data(self, data_num_scaled, data_discrete_oh):
        data = self.inv_data_transform(data_num_scaled, data_discrete_oh)
        return self.classifier(data)

    def eval_classifier_on_transformed_data(self, data_num_scaled, data_discrete_oh):
        return tf.py_function(self._eval_classifier_on_transformed_data,
                              inp=[data_num_scaled, data_discrete_oh],
                              Tout=[tf.float32])



    def train_step_func(self, n_batch, ret_loss=False):

        for i in range(self.n_critic):
            queries = self.generate_queries(n_batch)
            data_batch_real = self.get_numpy_data_batch_real_from_queries(queries)
            self.train_step_critic(data_batch_real, n_batch, queries=queries)

        queries = self.generate_queries(n_batch)
        self.train_step_generator(n_batch, queries=queries)

        if ret_loss:
            return 0, 0
        else:
            return

    def create_critic(self):
        """
        Internal function for creating the critic neural network. Uses input parameters given to TabGAN to decide
        between different critic architectures
        """
        input_numeric = Input(shape=(self.n_columns_num * self.pac), name="Numeric_input")
        input_discrete = Input(shape=(self.n_columns_discrete_oh * self.pac), name="Discrete_input")
        input_instance = Input(shape=((self.n_columns_num + self.n_columns_discrete_oh) * self.pac), name="Query_instance")
        queries = [input_instance]
        queries_combined = []
        if self.query_critic_instance:
            queries_combined += [input_instance]

        if self.query_critic_classifier_label:
            input_classifier_label = Input(shape=(1 * self.pac), name="Query_classifier_label")
            queries += [input_classifier_label]
            queries_combined += [input_classifier_label]

        if len(queries_combined) > 1:
            queries_combined = concatenate(queries_combined, name="Combining_queries")
        else:
            queries_combined = queries[0]

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
        queries = [input_instance]

        if self.query_generator_classifier_label:
            input_classifier_label = Input(shape=(1 * self.pac), name="Query_classifier_label")
            queries += [input_classifier_label]
            combined_queries = concatenate(queries, axis=1, name="Combining_queries")
        else:
            combined_queries = queries[0]

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

        model = Model(inputs=inputs, outputs=[output_numeric, output_discrete])
        return model

    def plot2D_axis_update(self, ax, latent_vec, num_cols, discrete_col=None, queries=None, inv_scale=True,
                           legend=True, color_opacity=1):
        gen_data_num, gen_data_discrete = self.generator([latent_vec, queries] if self.use_query
                                                         else [latent_vec])
        color_gen = None
        color_queries = None
        if inv_scale:
            gen_data = self.inv_data_transform(gen_data_num, gen_data_discrete)
            queries = self.inv_data_transform(*self.split_transformed_data(queries[0]))
            if discrete_col is None:
                color_dict = {"all": next(ax._get_lines.prop_cycler)['color']}
            else:
                labels_unique = np.sort(np.unique([gen_data[discrete_col], queries[discrete_col]]))
                colors_unique = [next(ax._get_lines.prop_cycler)['color'] for label in labels_unique]
                color_dict = {label: color for label, color in zip(labels_unique, colors_unique)}
                color_gen = gen_data[discrete_col].map(color_dict)
                color_queries = queries[discrete_col].map(color_dict)

                if legend:
                    markers = [plt.Line2D([0, 0], [0, 0], color=color, marker='o', linestyle='', alpha=0.5)
                               for color in color_dict.values()]
                    ax.legend(markers, color_dict.keys(), numpoints=1)

                class_gen_data = np.where(self.classifier(gen_data) > 0.5, 1, 0).astype(np.int)
                class_queries = np.where(self.classifier(queries) > 0.5, 1, 0).astype(np.int)
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
                  color=np.where(class_gen_data != class_queries, "yellow", "red"))
