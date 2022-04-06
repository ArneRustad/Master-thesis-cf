import pandas as pd
import numpy as np
import tensorflow as tf
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

    def generate_queries(self, n):
        ix = tf.random.uniform(shape=[n], minval=0, maxval=self.data.shape[0],
                               dtype=tf.int64)
        queries = tf.py_function(lambda ixs: [self.data_num_scaled[ixs, ], self.data_discrete_oh[ixs, ]],
                                 inp=[ix], Tout=[tf.float32, tf.float32])
        return tf.concat(queries, axis=1)

    def _python_get_numpy_data_batch_real_from_queries(self, queries):
        classifier_values = self.classifier(queries)
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

    # def train_step_func(self, n_batch):
    #
    #     for i in range(self.n_critic):
    #         queries = self.generate_queries(n_batch)
    #         data_batch_real = self.get_numpy_data_batch_real_from_queries(queries)
    #         self.train_step_critic(data_batch_real, n_batch, queries=queries)
    #
    #     self.train_step_generator(n_batch)
    #
    #     return 0, 0
