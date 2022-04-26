import pandas as pd
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler, OneHotEncoder, QuantileTransformer
from sklearn.compose import ColumnTransformer

from .tabGAN import TabGAN
from .fast_nondominated_sort import fast_non_dominated_sort

class TabGANcf(TabGAN):
    def __init__(self, data_train, classifier, **kwargs):

        super().__init__(data=data_train,
                         use_query=True, ctgan=False,
                         **kwargs
                         )
        self.n_columns_query = self.n_columns_num + self.n_columns_discrete_oh
        self.classifier = classifier
        self.data_ix_classifier_0 = np.where(classifier(self.data) <= 0.5)[0]
        self.data_ix_classifier_1 = np.where(classifier(self.data) > 0.5)[0]
        print(self.data_ix_classifier_1[tf.convert_to_tensor(np.array([1,2,3])).numpy()])

    def generate_queries(self, n):
        ix = tf.random.uniform(shape=[n], minval=0, maxval=self.data.shape[0], dtype=tf.int64)
        # queries = tf.py_function(lambda ixs: [self.data_num_scaled[ixs, ], self.data_discrete_oh[ixs, ]],
        #                          inp=[ix], Tout=[tf.float32, tf.float32])
        # ix = tf.random.uniform(shape=[n], minval=0, maxval=self.data_ix_classifier_0.shape[0], dtype=tf.int64)
        # ix = tf.py_function(lambda ixs: self.data_ix_classifier_0[ixs.numpy()],
        #                     inp=[ix], Tout=[tf.float32])
        queries = tf.py_function(lambda ixs: [self.data_num_scaled[ixs, ], self.data_discrete_oh[ixs, ]],
                                 inp=[ix], Tout=[tf.float32, tf.float32])
        return tf.concat(queries, axis=1)

    def _python_get_numpy_data_batch_real_from_queries(self, queries):
        classifier_values = self.classifier(
            self.inv_data_transform(*self.split_transformed_data(queries))
        )
        classifier_label = np.round(classifier_values).astype(np.int64)

        n_batch_1 = np.sum(classifier_label == 1)
        n_batch_0 = np.sum(classifier_label == 0)

        ix_batch_0 = np.random.choice(self.data_ix_classifier_0, n_batch_0)
        ix_batch_1 = np.random.choice(self.data_ix_classifier_1, n_batch_1)

        ix = np.empty(shape=queries.shape[0], dtype=np.int64)
        ix[classifier_label == 0] = ix_batch_0
        ix[classifier_label == 1] = ix_batch_1
        ix = ix.astype(np.int64)

        return [self.data_num_scaled[ix, ], self.data_discrete_oh[ix, ]]

    def get_numpy_data_batch_real_from_queries(self, queries):
        return tf.py_function(self._python_get_numpy_data_batch_real_from_queries,
                              inp=[queries], Tout=[tf.float32, tf.float32])

    def calc_loss_generator(self, fake_output, gen_data_num, gen_data_discrete, queries):
        queries_num = queries[:, :self.n_columns_num]
        queries_discrete = queries[:, self.n_columns_num:]
        # Standard WGAN loss term
        loss = - tf.reduce_mean(fake_output)
        loss_vec = [loss]
        # Distance 1 norm loss term
        loss += tf.reduce_mean(tf.reduce_mean(tf.abs(queries_num - gen_data_num), axis=1))
        loss += tf.reduce_mean(tf.reduce_mean(tf.abs(queries_discrete - gen_data_discrete), axis=1))
        loss_vec = tf.concat((loss_vec, [loss - loss_vec[0]]), axis=-1)
        # Classifier loss term
        # queries_inv_transformed = tf.py_function(lambda query_num, query_discrete:
        #                                          self.inv_data_transform(query_num, query_discrete),
        #                                          inp=[queries_num, queries_discrete],
        #                                          Tout=[tf.float32, tf.float32])
        # class_queries = tf.cast(
        #     tf.where(self.eval_classifier_on_transformed_data(queries_num, queries_discrete) < 0.5, 0, 1),
        #     tf.int32
        # )
        # classifier_val_gen_data =  self.eval_classifier_on_transformed_data(gen_data_num, gen_data_discrete)
        # classifier_val_gen_data = tf.where(class_queries == 1, 1 - classifier_val_gen_data, 0)
        # loss += tf.reduce_sum(classifier_val_gen_data * fake_output / classifier_val_gen_data)
        loss_vec = tf.concat((loss_vec, [loss - tf.reduce_sum(loss_vec)]), axis=-1)

        tf.print(loss_vec)
        return loss

        tf.print(loss)

    def _eval_classifier_on_transformed_data(self, data_num_scaled, data_discrete_oh):
        data = self.inv_data_transform(data_num_scaled, data_discrete_oh)
        return self.classifier(data)

    def eval_classifier_on_transformed_data(self, data_num_scaled, data_discrete_oh):
        return tf.py_function(self._eval_classifier_on_transformed_data,
                              inp=[data_num_scaled, data_discrete_oh],
                              Tout=[tf.float32])



    def train_step_func(self, n_batch):

        for i in range(self.n_critic):
            queries = self.generate_queries(n_batch)
            data_batch_real = self.get_numpy_data_batch_real_from_queries(queries)
            self.train_step_critic(data_batch_real, n_batch, queries=queries)

        self.train_step_generator(n_batch)

        return 0, 0

    def plot2D_axis_update(self, ax, latent_vec, num_cols, discrete_col=None, queries=None, inv_scale=True,
                           legend=True, color_opacity=1):
        gen_data_num, gen_data_discrete = self.generator([latent_vec, queries] if self.use_query
                                                         else [latent_vec])
        color_gen = None
        color_queries = None
        if inv_scale:
            gen_data = self.inv_data_transform(gen_data_num, gen_data_discrete)
            queries = self.inv_data_transform(*self.split_transformed_data(queries))
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
            queries = pd.DataFrame(self.split_transformed_data(queries)[0], columns=self.columns_num)
        ax.scatter(queries[num_cols[0]], queries[num_cols[1]], c=color_queries, alpha=color_opacity,
                   edgecolors="black")
        ax.scatter(gen_data[num_cols[0]], gen_data[num_cols[1]], c=color_gen, alpha=color_opacity)
        ax.quiver(queries[num_cols[0]], queries[num_cols[1]],
                  gen_data[num_cols[0]] - queries[num_cols[0]],
                  gen_data[num_cols[1]] - queries[num_cols[1]],
                  angles='xy', scale_units='xy', scale=1,
                  color=np.where(class_gen_data != class_queries, "yellow", "red"))
