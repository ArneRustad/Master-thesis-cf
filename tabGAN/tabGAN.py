import pandas as pd
import numpy as np
import os
import shutil
import scipy
import tensorflow as tf
import warnings
from math import floor, ceil
from tqdm.auto import tqdm

from tensorflow.keras.models import Model
from tensorflow.keras.layers import (Input, Dense, Flatten, concatenate, LeakyReLU, ReLU,
                                     ELU, Embedding, Activation, Dropout)
from tensorflow.keras.activations import gelu, selu, swish, relu
import tensorflow_probability as tfp
tfd = tfp.distributions

from sklearn.preprocessing import StandardScaler, OneHotEncoder, QuantileTransformer
from sklearn.compose import ColumnTransformer

from IPython.display import clear_output, display, Image, Video
import matplotlib.pyplot as plt

class TabGAN:
    """
    Class for creating a tabular GAN that can generate synthetic datasets.
    """
    def __init__(self, data, batch_size=500, gan_method="WGAN-GP", wgan_lambda=10, pac=1,
                 n_hidden_layers=2, n_hidden_generator_layers=None, n_hidden_critic_layers=None,
                 dim_hidden=256, dim_hidden_generator=None, dim_hidden_critic=None,
                 dim_latent=128, gumbel_temperature=0.5, n_critic=5,
                 quantile_transformation_int=True, max_quantile_share=1, print_quantile_shares=False,
                 qt_distribution="normal",
                 n_quantiles_int=1000, qt_n_subsample=1e5,
                 quantile_rand_transformation=True, qtr_spread=0.4, qtr_lbound_apply=0.05,
                 ctgan=False, ctgan_log_frequency=True, ctgan_binomial_loss=True,
                 ctgan_binomial_distance_floor=0,
                 activation_function="LeakyReLU", leaky_relu_alpha=0.3, gelu_approximate=False,
                 elu_alpha=1.0,
                 add_dropout_critic=[], add_dropout_generator=[],
                 dropout_rate=0, dropout_rate_critic=None, dropout_rate_generator=None,
                 add_connection_discrete_to_num=False, add_connection_num_to_discrete=False,
                 dim_hidden_layer_discrete_to_num=0, dim_hidden_layer_num_to_discrete=0,
                 add_connection_query_to_discrete=False,
                 optimizer="adam", opt_lr=0.0002, adam_beta1=0, adam_beta2=0.999, sgd_momentum=0.0,
                 adam_amsgrad=False, sgd_nesterov=False,
                 rmsprop_rho=0.9, rmsprop_momentum=0, rmsprop_centered=False,
                 ckpt_dir=None, ckpt_every=None, ckpt_max_to_keep=None, ckpt_name="ckpt_epoch",
                 noise_discrete_unif_max=0, use_query=None, np_data_fix=False,
                 tf_data_use=None, tf_data_shuffle=True, tf_data_prefetch=True, tf_data_cache=False,
                 tf_make_graph=True, tf_make_critic_step_graph=None, tf_make_gen_step_graph=None,
                 tf_make_train_step_graph=None, tf_make_numpy_data_step_graph=None,
                 tf_make_em_distance_graph=None, tf_make_generate_latent_graph=None,
                 jit_compile=False, jit_compile_critic_step=None, jit_compile_gen_step=None,
                 jit_compile_numpy_data_step=None, jit_compile_generate_latent=None, jit_compile_em_distance=None,
                 default_epochs_to_train=None):
        # Create variable defaults if needed
        if n_quantiles_int > data.shape[0]:
            n_quantiles_int = data.shape[0]

        if isinstance(ctgan_binomial_distance_floor, str):
            if ctgan_binomial_distance_floor == "noise_discrete_unif_max":
                ctgan_binomial_distance_floor = noise_discrete_unif_max
            else:
                ValueError("ctgan_binomial_distance_floor must be a float between 0 and 1 or the"
                           "string 'noise_discrete_unif_max'. No other input to the parameter is allowed)")


        activation_function = activation_function.lower()
        dict_activation_function = {
            "leakyrelu": LeakyReLU(alpha=leaky_relu_alpha),
            "relu": ReLU(),
            "gelu": lambda x: gelu(x, approximate=gelu_approximate),
            "selu": selu,
            "swish": swish,
            "elu": ELU(alpha=elu_alpha),
            "squaredrelu": lambda x: tf.math.square(relu(x)),
            "leakysquaredrelu": lambda x: tf.where(x > 0, tf.math.square(relu(x)), leaky_relu_alpha * x)
        }
        if activation_function in dict_activation_function.keys():
            activation_function = dict_activation_function[activation_function]
        else:
            raise ValueError(f"The activation function {activation_function} is not (yet) implemented")

        if tf_data_use is None:
            if ctgan:
                tf_data_use = False
            else:
                tf_data_use = True

        if use_query is None:
            use_query = True if ctgan else False
        if n_hidden_generator_layers is None:
            n_hidden_generator_layers = n_hidden_layers
        if n_hidden_critic_layers is None:
            n_hidden_critic_layers = n_hidden_layers
        if dim_hidden_generator is None:
            dim_hidden_generator = dim_hidden
        if dim_hidden_critic is None:
            dim_hidden_critic = dim_hidden
        if isinstance(dim_hidden_generator, list):
            assert len(dim_hidden_generator) == n_hidden_generator_layers
        else:
            dim_hidden_generator = [dim_hidden_generator] * n_hidden_generator_layers
        if isinstance(dim_hidden_critic, list):
            assert len(dim_hidden_critic) == n_hidden_critic_layers
        else:
            dim_hidden_critic = [dim_hidden_critic] * n_hidden_critic_layers

        if tf_make_critic_step_graph is None:
            tf_make_critic_step_graph = tf_make_graph
        if tf_make_gen_step_graph is None:
            tf_make_gen_step_graph = tf_make_graph
        if tf_make_train_step_graph is None:
            tf_make_train_step_graph = tf_make_graph
        if tf_make_numpy_data_step_graph is None:
            tf_make_numpy_data_step_graph = tf_make_graph
        if tf_make_em_distance_graph is None:
            tf_make_em_distance_graph = tf_make_graph
        if tf_make_generate_latent_graph is None:
            tf_make_generate_latent_graph = tf_make_graph
        
        if jit_compile_critic_step is None:
            jit_compile_critic_step = jit_compile
        if jit_compile_gen_step is None:
            jit_compile_gen_step = jit_compile
        if jit_compile_numpy_data_step is None:
            jit_compile_numpy_data_step = jit_compile
        if jit_compile_generate_latent is None:
            jit_compile_generate_latent = jit_compile
        if jit_compile_em_distance is None:
            jit_compile_em_distance = jit_compile

        if dropout_rate_critic is None:
            dropout_rate_critic = dropout_rate
        if dropout_rate_generator is None:
            dropout_rate_generator = dropout_rate

        # Assert correct input
        assert not (ctgan and not use_query)
        gan_methods = ["WGAN-GP", "WGAN-SGP"]
        if gan_method not in gan_methods:
            raise ValueError("Wrong input to parameter gan_method. Currently only implemented options:", gan_methods)
        if batch_size % pac != 0:
            raise ValueError(f"Batch size ({batch_size}) must be a multiple of pac size ({pac})." 
                             " Please change one or both of these parameters")
        if ctgan and tf_data_use:
            raise ValueError("tf_data_use=True is not yet implemented in combination with ctgan=True")
        if ctgan_binomial_distance_floor > 1 or ctgan_binomial_distance_floor < 0:
            raise ValueError("ctgan_binomial_distance_floor must be an float between (and including) 0 and 1")

        # Initialize variables
        self.data = data
        self.batch_size = batch_size
        self.columns = data.columns
        self.n_columns = len(self.columns)
        self.nrow = data.shape[0]
        self.batch_size = batch_size
        self.gan_method = gan_method
        self.wgan_lambda = wgan_lambda
        self.pac = pac
        self.n_hidden_generator_layers = n_hidden_generator_layers
        self.n_hidden_critic_layers = n_hidden_critic_layers
        self.dim_latent = dim_latent
        self.dim_hidden_generator = dim_hidden_generator
        self.dim_hidden_critic = dim_hidden_critic
        self.gumbel_temperature = gumbel_temperature
        self.optimizer = optimizer
        self.n_critic = n_critic
        self.opt_lr = opt_lr
        self.adam_beta1 = adam_beta1
        self.adam_beta2 = adam_beta2
        self.adam_amsgrad = adam_amsgrad
        self.sgd_momentum = sgd_momentum
        self.sgd_nesterov = sgd_nesterov
        self.rmsprop_rho = rmsprop_rho
        self.rmsprop_momentum = rmsprop_momentum
        self.rmsprop_centered = rmsprop_centered
        self.ckpt_dir = ckpt_dir
        self.ckpt_every = ckpt_every
        self.ckpt_max_to_keep = ckpt_max_to_keep
        self.ckpt_name = ckpt_name
        self.ckpt_prefix = os.path.join(self.ckpt_dir, self.ckpt_name) if not self.ckpt_dir is None else None
        self.noise_discrete_unif_max = noise_discrete_unif_max
        self.quantile_transformation_int = quantile_transformation_int
        self.max_quantile_share = max_quantile_share
        self.print_quantile_shares = print_quantile_shares
        self.quantile_rand_transformation = quantile_rand_transformation
        self.qt_distribution = qt_distribution
        self.n_quantiles_int = n_quantiles_int
        self.qt_n_subsample = qt_n_subsample
        self.initialized_gan = False
        self.qtr_spread = qtr_spread
        self.qtr_lbound_apply = qtr_lbound_apply
        self.ctgan = ctgan
        self.ctgan_log_frequency = ctgan_log_frequency
        self.ctgan_binomial_loss = ctgan_binomial_loss
        self.ctgan_binomial_distance_floor = ctgan_binomial_distance_floor
        self.activation_function = activation_function
        self.leaky_relu_alpha = leaky_relu_alpha
        self.add_dropout_critic = add_dropout_critic
        self.add_dropout_generator = add_dropout_generator
        self.dropout_rate_critic = dropout_rate_critic
        self.dropout_rate_generator = dropout_rate_generator
        self.add_connection_discrete_to_num = add_connection_discrete_to_num
        self.add_connection_num_to_discrete = add_connection_num_to_discrete
        self.add_connection_query_to_discrete = add_connection_query_to_discrete
        self.dim_hidden_layer_discrete_to_num = dim_hidden_layer_discrete_to_num
        self.dim_hidden_layer_num_to_discrete = dim_hidden_layer_num_to_discrete
        self.use_query = use_query
        self.tf_data_use = tf_data_use
        self.tf_data_shuffle = tf_data_shuffle
        self.tf_data_prefetch = tf_data_prefetch
        self.tf_data_cache = tf_data_cache
        self.np_data_fix = np_data_fix
        self.tf_make_graph = tf_make_graph
        self.tf_make_critic_step_graph = tf_make_critic_step_graph
        self.tf_make_gen_step_graph = tf_make_gen_step_graph
        self.tf_make_train_step_graph = tf_make_train_step_graph
        self.tf_make_numpy_data_step_graph = tf_make_numpy_data_step_graph
        self.tf_make_em_distance_graph = tf_make_em_distance_graph
        self.tf_make_generate_latent_graph = tf_make_generate_latent_graph
        self.jit_compile = jit_compile
        self.jit_compile_critic_step = jit_compile_critic_step
        self.jit_compile_gen_step = jit_compile_gen_step
        self.jit_compile_numpy_data_step = jit_compile_numpy_data_step
        self.jit_compile_generate_latent = jit_compile_generate_latent
        self.jit_compile_em_distance = jit_compile_em_distance
        self.jit_compile_arg_func = None # Will be initialized later depending on tensorflow version being less than 2.5 or above
        self.default_epochs_to_train = default_epochs_to_train

        self.np_ix_start = tf.convert_to_tensor(0)
        self.np_ix_list = np.array([], dtype=np.int)
        self.np_ix_iter = None

        # Separate numeric data, fit numeric scaler and scale numeric data. Store numeric column names.
        self.data_num = data.select_dtypes(include=np.number)
        self.columns_num = self.data_num.columns
        self.columns_num_mask = [col in self.columns_num for col in self.columns]
        self.n_columns_num = len(self.data_num.columns)
        self.columns_num_int_mask = self.data_num.dtypes.astype(str).str.contains("int")
        self.columns_int = self.columns_num[self.columns_num_int_mask]
        self.columns_num_int_pos = np.arange(len(self.columns_num))[self.columns_num_int_mask]
        self.columns_float = self.columns_num[np.logical_not(self.columns_num_int_mask)]
        if self.quantile_transformation_int:
            self.scaler_num = ColumnTransformer(transformers=[("float", StandardScaler(), self.columns_float), (
            "int", QuantileTransformer(n_quantiles=n_quantiles_int, output_distribution=self.qt_distribution,
                                       subsample=self.qt_n_subsample),
            self.columns_int)])
            self.data_num_scaled = self.scaler_num.fit_transform(self.data_num)

            if self.max_quantile_share < 1:
                self.fix_quantile_share()

            if self.quantile_rand_transformation:
                self.data_num_scaled = self.randomize_quantile_transformation(self.data_num_scaled)
        else:
            self.scaler_num = StandardScaler()
            self.data_num_scaled = self.scaler_num.fit_transform(self.data_num)
            self.columns_num_int_mask = None
            self.columns_int = None
            self.columns_float = None
            self.columns_num_int_pos = None

        # Separate discrete data, fit one-hot-encoder, perform one hot encoding. Store discrete column names
        # and store the number of categories for each discrete variable
        self.data_discrete = data.select_dtypes(exclude=np.number)
        self.columns_discrete = self.data_discrete.columns
        self.columns_discrete_mask = [col in self.columns_discrete for col in self.columns]
        self.n_columns_discrete = len(self.columns_discrete)

        self.oh_encoder = OneHotEncoder(sparse=False)
        self.data_discrete_oh = self.oh_encoder.fit_transform(self.data_discrete)
        self.n_columns_discrete_oh = self.data_discrete_oh.shape[1]

        # Store number of categories for each discrete column
        self.column_discrete_n_cats = [len(col_cat_vec) for col_cat_vec in self.oh_encoder.categories_]
        self.column_discrete_n_cats_cumulative = np.cumsum(self.column_discrete_n_cats)
        if self.noise_discrete_unif_max > 0:
            noise_discrete = np.random.uniform(low = 0, high = self.noise_discrete_unif_max,
                                              size = self.data_discrete_oh.shape)
            self.data_discrete_oh += noise_discrete * np.where(self.data_discrete_oh > 0.5, -1, 1)

        self.categories_len = [len(i) for i in self.oh_encoder.categories_]

        # Create Gumbel-activation function
        tf.keras.utils.get_custom_objects().update({'gumbel_softmax': Activation(self.gumbel_softmax)})

        # To accomodate for that jit_compile was referred to as experimental compile in tensorflow versions before 2.5
        if tf.__version__ < "2.5":
            self.jit_compile_arg_func = lambda bool: {"experimental_compile": bool}
        else:
            self.jit_compile_arg_func = lambda bool: {"jit_compile": bool}

        # Create generator and critic objects as well as critic and generator optimizer
        #self.initialize_gan() #Not needed now, will be done when initiating training
        # If needed create checkpoint manager
        if (self.ckpt_dir != None):
            self.initialize_cptk()

        # Create either tf dataset or numpy dataset in float32
        if self.tf_data_use:
            self.data_processed = tf.data.Dataset.zip(
                (tf.data.Dataset.from_tensor_slices(tf.cast(self.data_num_scaled, dtype=tf.float32)),
                 tf.data.Dataset.from_tensor_slices(tf.cast(self.data_discrete_oh, dtype=tf.float32))
                 )
            )
            if self.tf_data_shuffle:
                self.data_processed = self.data_processed.shuffle(buffer_size=self.data.shape[0])
            self.data_processed = self.data_processed.repeat().batch(self.batch_size)
            if self.tf_data_prefetch:
                #self.data_processed = self.data_processed.apply(tf.data.experimental.prefetch_to_device("/gpu:0"))
                self.data_processed = self.data_processed.prefetch(tf.data.AUTOTUNE)
            if self.tf_data_cache:
                self.data_processed = self.data_processed.cache()
            #self.data_processed = self.data_processed.apply(tf.data.experimental.prefetch_to_device("/gpu:0"))
            self.data_processed_iter = iter(self.data_processed)

            self.data_num_scaled_cast = None
            self.data_discrete_oh_cast = None
        else:
            self.data_num_scaled_cast = self.data_num_scaled.astype(np.float32)
            self.data_discrete_oh_cast = self.data_discrete_oh.astype(np.float32)

            self.data_processed = None
            self.data_processed_iter = None

        if self.ctgan:
            self.query_probs = np.empty(shape=(self.n_columns_discrete_oh), dtype=np.float32)
            self.query_original_probs = np.empty(shape=(self.n_columns_discrete_oh), dtype=np.float32)
            self.n_columns_query = self.n_columns_discrete_oh
            #self.map_query_id_to_mask_indices = {}
            start_idx = 0
            for i, col_discrete in enumerate(self.columns_discrete):
                end_idx = self.column_discrete_n_cats_cumulative[i]
                category_freq = np.sum(self.data_discrete_oh[:, start_idx:end_idx], axis=0)
                self.query_original_probs[start_idx:end_idx] = category_freq / np.sum(category_freq)
                if self.ctgan_log_frequency:
                    category_freq = np.log(1 + category_freq)
                self.query_probs[start_idx:end_idx] = category_freq / np.sum(category_freq)
                #self.map_query_id_to_mask_indices.update({i: (start_idx, end_idx) for i in range(start_idx, end_idx)})
                start_idx = end_idx

            self.query_original_probs /= self.n_columns_discrete
            self.query_probs /= self.n_columns_discrete
            self.queries_all = np.identity(self.n_columns_discrete_oh, dtype=np.float32)

            self.map_query_id_to_indices_list = {}
            self.map_query_id_to_n_indices = np.empty(shape=self.n_columns_discrete_oh, dtype=np.float32)
            for i in range(self.n_columns_discrete_oh):
                curr_indices = np.flatnonzero(self.data_discrete_oh[:, i])
                self.map_query_id_to_indices_list[i] = curr_indices
                self.map_query_id_to_n_indices[i] = curr_indices.shape[0]

            self.map_query_id_and_indices_idx_to_obs_idx = np.vectorize(
                lambda query_id, indices_idx: self.map_query_id_to_indices_list[query_id][indices_idx]
            )

            # def ctgan_cond_loss_func(queries, discrete_vec, query_id):
            #     mask_indices = self.map_query_id_to_mask_indices[query_id]
            #     return tf.keras.losses.CategoricalCrossentropy(
            #         reduction=tf.keras.losses.Reduction.SUM_OVER_BATCH_SIZE,
            #         from_logits=False)(queries[mask_indices], discrete_vec[mask_indices])
    def initialize_gan(self, tf_make_graph=True):
        """
        Internal function used for initializing the GAN architecture
        """
        # Create generator and critic objects
        self.generator = self.create_generator()
        self.critic = self.create_critic()

        # Create optimizers for generator and critic
        if self.optimizer.lower() == "adam":
            self.generator_optimizer = tf.keras.optimizers.Adam(learning_rate=self.opt_lr, beta_1=self.adam_beta1, beta_2=self.adam_beta2, amsgrad=self.adam_amsgrad)
            self.critic_optimizer = tf.keras.optimizers.Adam(learning_rate=self.opt_lr, beta_1=self.adam_beta1, beta_2=self.adam_beta2, amsgrad=self.adam_amsgrad)
        elif self.optimizer.lower() == "sgd":
            self.generator_optimizer = tf.keras.optimizers.SGD(learning_rate=self.opt_lr, momentum=self.sgd_momentum, nesterov=self.sgd_nesterov)
            self.critic_optimizer = tf.keras.optimizers.SGD(learning_rate=self.opt_lr, momentum=self.sgd_momentum, nesterov=self.sgd_nesterov)
        elif self.optimizer.lower() == "rmsprop":
            self.generator_optimizer = self.tf.keras.optimizers.RMSprop(learning_rate=self.opt_lr, rho=self.rmsprop_rho, momentum=self.rmsprop_rho, centered=self.rmsprop_centered)
            self.critic_optimizer = self.tf.keras.optimizers.RMSprop(learning_rate=self.opt_lr, rho=self.rmsprop_rho, momentum=self.rmsprop_rho, centered=self.rmsprop_centered)
        else:
            raise ValueError("Optimizer name not recognized. Currently only implemented optimizers: adam, sgd and rmsprop")
        self.start_epoch = 0
        
        if self.tf_make_generate_latent_graph and not (self.tf_make_graph and self.jit_compile_generate_latent):
            self.generate_latent = tf.function(self.generate_latent_func, **self.jit_compile_arg_func(self.jit_compile_generate_latent))
        else:
            self.generate_latent = self.generate_latent_func
        if self.tf_make_numpy_data_step_graph and not self.tf_data_use:
            self.get_numpy_data_batch_real = tf.function(self.get_numpy_data_batch_real_func, **self.jit_compile_arg_func(self.jit_compile_numpy_data_step))
        else:
            self.get_numpy_data_batch_real = self.get_numpy_data_batch_real_func
        if self.tf_make_critic_step_graph:
            self.train_step_critic = tf.function(self.train_step_critic_func, **self.jit_compile_arg_func(self.jit_compile_critic_step))
        else:
            self.train_step_critic = self.train_step_critic_func
        if self.tf_make_em_distance_graph:
            self.compute_em_distance = tf.function(self.compute_em_distance_func, **self.jit_compile_arg_func(self.jit_compile_em_distance))
        else:
            self.compute_em_distance = self.compute_em_distance_func
        if self.tf_make_gen_step_graph:
            self.train_step_generator = tf.function(self.train_step_generator_func, **self.jit_compile_arg_func(self.jit_compile_gen_step))
        else:
            self.train_step_generator = self.train_step_generator_func
        if self.tf_make_train_step_graph:
            self.train_step = tf.function(self.train_step_func)
        else:
            self.train_step = self.train_step_func

        self.initialized_gan = True

    def fix_quantile_share(self):
        qt_transformer = self.scaler_num.named_transformers_["int"]
        n_quantiles = qt_transformer.n_quantiles_
        max_n_quantiles_per_value = floor(n_quantiles * self.max_quantile_share)
        if max_n_quantiles_per_value == 0:
            warnings.warn(f"You have chose a max_share={self.max_quantile_share} along with "
                          f"n_quantiles={self.n_quantiles_int} such that the maximum number of" 
                          "categories per unique value is less than 1. This is automatically changed"
                          "such that maximum number of categories per unique value is equal to 1.")
            max_n_quantiles_per_value = 1
        for col_qt_idx, col_idx in enumerate(self.columns_num_int_pos):
            col_unique_values, col_value_counts = np.unique(qt_transformer.quantiles_[:, col_qt_idx], return_counts=True)
            col_percentages = col_value_counts / sum(col_value_counts)
            if max(col_percentages) > self.max_quantile_share:
                if self.print_quantile_shares:
                    print(f"Column {self.columns_num[col_idx]} quantile percentages:", col_percentages)
                indices_too_common_values = np.where(col_percentages > self.max_quantile_share)[0]
                # Freeing up available quantile spaces from values that are too common
                # (have a larger share than permitted by the parameter self.max_quantile_share)
                col_bool_remaining_values = np.ones(self.data_num.iloc[:, col_idx].shape[0], dtype=bool)
                for curr_common_value_idx in indices_too_common_values:
                    curr_common_value = col_unique_values[curr_common_value_idx]
                    curr_common_value_count = col_value_counts[curr_common_value_idx]
                    curr_avail_quantile_indices = np.nonzero(
                        np.isclose(qt_transformer.quantiles_[:, col_qt_idx], curr_common_value)
                    )[0][max_n_quantiles_per_value:]
                    qt_transformer.quantiles_[curr_avail_quantile_indices, col_qt_idx] = np.nan
                    col_bool_remaining_values *= np.logical_not(np.isclose(self.data_num.iloc[:, col_idx], curr_common_value))

                # Reset the quantile spaces for values with shares smaller than the parameter max_share,
                # before a redraw
                indices_less_common = np.where(col_percentages <= self.max_quantile_share)
                values_less_common = col_unique_values[indices_less_common]
                for curr_less_common_value in values_less_common:
                    qt_transformer.quantiles_[np.where(np.isclose(qt_transformer.quantiles_, curr_less_common_value))[0], col_qt_idx] = np.nan

                available_quantile_indices = np.where(np.isnan(qt_transformer.quantiles_))[0]
                n_available_quantiles = len(available_quantile_indices)
                col_remaining_values = self.data_num.iloc[:, col_idx].loc[col_bool_remaining_values].to_numpy()

                if col_remaining_values.shape[0] != n_available_quantiles:
                    new_quantiles = np.nanpercentile(col_remaining_values,
                                                     q=np.linspace(0, 100, n_available_quantiles))
                    # Due to floating-point precision error in `np.nanpercentile`,
                    # make sure that quantiles are monotonically increasing.
                    # Upstream issue in numpy: https://github.com/numpy/numpy/issues/14685
                    new_quantiles = np.maximum.accumulate(new_quantiles)
                    qt_transformer.quantiles_[available_quantile_indices, col_qt_idx] = new_quantiles
                else:
                    qt_transformer.quantiles_[available_quantile_indices, col_qt_idx] = col_remaining_values
                qt_transformer.quantiles_[:, col_qt_idx] = np.sort(qt_transformer.quantiles_[:, col_qt_idx])

    def randomize_quantile_transformation(self, data):
        """
        Internal function for performing the randomized quantile transformation
        """
        qt_transformer = self.scaler_num.named_transformers_["int"]
        references = np.copy(qt_transformer.references_)
        quantiles = np.copy(qt_transformer.quantiles_)
        lower_bound_references = 1e-7
        references[[0, -1]] = [lower_bound_references, 1 - lower_bound_references]
        for i, col in enumerate(self.columns_num_int_pos):
            quantiles_curr = quantiles[:, i]
            quantiles_unique_integer = np.unique(quantiles_curr)
            quantiles_unique_integer = quantiles_unique_integer[np.isclose(np.mod(quantiles_unique_integer, 1), 0)]
            for integer in quantiles_unique_integer:
                curr_references = references[np.isclose(quantiles_curr, integer)]
                n_curr_references = curr_references.shape[0]
                if n_curr_references >= max((self.qtr_lbound_apply * self.n_quantiles_int, 2)):
                    mask = self.data_num[self.columns_int[i]] == integer
                    n_obs_curr = np.sum(mask)
                    curr_reference_range = curr_references[-1] - curr_references[0]
                    low = curr_references[0] + curr_reference_range * (0.5 - self.qtr_spread / 2)
                    high = curr_references[0] + curr_reference_range * (0.5 + self.qtr_spread / 2)
                    if self.qt_distribution == "normal":
                        data[mask, col] = scipy.stats.norm.ppf(np.random.uniform(low=low, high=high, size=n_obs_curr))
                    elif self.qt_distribution == "uniform":
                        data[mask, col] = np.random.uniform(low=low, high=high, size=n_obs_curr)
                    else:
                        raise ValueError("qt_distribution must be equal to normal or uniform")
        return data

    def initialize_cptk(self):
        """
        Internal function for initializing checkpoint manager used to save the progress of the model.
        """
        os.makedirs(self.ckpt_dir, exist_ok=True)
        self.ckpt = tf.train.Checkpoint(epoch=tf.Variable(0), generator_opt=self.generator_optimizer,
                                        critic_opt=self.critic_optimizer, generator=self.generator,
                                        critic=self.critic)
        self.ckpt_manager = tf.train.CheckpointManager(self.ckpt, self.ckpt_dir, max_to_keep=self.ckpt_max_to_keep,
                                                       checkpoint_name=self.ckpt_name)

    def split_transformed_data(self, transformed_data):
        data_num_scaled = transformed_data[:, :self.n_columns_num]
        data_discrete_oh = transformed_data[:, self.n_columns_num:]
        return data_num_scaled, data_discrete_oh

    def inv_data_transform(self, data_num_scaled, data_discrete_oh):
        """
        Internal function used for inverting the data transformation done in preprocessing
        """
        data_discrete = pd.DataFrame(self.oh_encoder.inverse_transform(data_discrete_oh), columns=self.columns_discrete)
        if self.quantile_transformation_int:
            if len(self.columns_float) > 0:
                data_float = pd.DataFrame(self.scaler_num.named_transformers_["float"].inverse_transform(
                    data_num_scaled[:, np.logical_not(self.columns_num_int_mask)]), columns=self.columns_float)
            else:
                data_float = None

            if len(self.columns_int) > 0:
                data_int_scaled = pd.DataFrame(data_num_scaled[:, self.columns_num_int_mask], columns=self.columns_int)
                data_int = pd.DataFrame(self.scaler_num.named_transformers_["int"].inverse_transform(data_int_scaled),
                    columns=self.columns_int)
            else:
                data_int = None
        else:
            data_float = pd.DataFrame(self.scaler_num.inverse_transform(data_num_scaled), columns=self.columns_num)
            data_int = None
        return (pd.concat([data_float, data_int, data_discrete], axis=1)[self.columns])


    def generate_queries(self, n, ret_query_id=False, original_probs=True):
        """
        Generate n queries. Implemented for ctgan parameter. Else a dummy function
        """
        if self.ctgan:
            if original_probs:
                query_probs = self.query_original_probs
            else:
                query_probs = self.query_probs
            query_ids = tf.random.categorical(
                logits=tf.math.log(tf.expand_dims(query_probs, axis=0)),
                num_samples=n, dtype=tf.int64
            )
            query_ids = tf.squeeze(query_ids, axis=0)
            drawn_queries = tf.py_function(lambda ids: self.queries_all[ids, :],
                                           inp=[query_ids], Tout=tf.float32)
            if ret_query_id:
                return drawn_queries, query_ids
            else:
                return drawn_queries
        # Return a dummy query consisting of only zeros
        else:
            return tf.zeros([n, self.n_columns_discrete_oh])

    def get_numpy_data_batch_real_from_queries(self, query_ids):
        return tf.py_function(self._get_numpy_data_batch_real_from_queries,
                              inp=[query_ids], Tout=[tf.float32, tf.float32])

    def _get_numpy_data_batch_real_from_queries(self, query_ids):
        u = np.random.uniform(low=0, high=1, size=query_ids.shape[0])
        indices_idx = np.round(u * (self.map_query_id_to_n_indices[query_ids] - 1)).astype(np.int)
        idx = self.map_query_id_and_indices_idx_to_obs_idx(query_ids, indices_idx)
        return [self.data_num_scaled_cast[idx], self.data_discrete_oh_cast[idx]]

    def _sample_interpret_input(self, n, queries, n_repeat):
        if queries is not None:
            queries = np.array(queries, ndmin=2)
            if n_repeat is not None:
                queries = np.repeat(queries, repeats=n_repeat)

        if n is None:
            if queries is None:
                n = self.nrow
            else:
                n = queries.shape[0]
        else:
            if n != queries.shape[0]:
                raise ValueError("If parameter n is set in addition to queries and/or n_repeat, "
                                 "then n must be equal to length of queries times n_repeat (if n_repeat set)")
        return n, queries

    def _sample_get_gen_input(self, n, queries):
        noise = self.generate_latent(n)
        if self.use_query:
            if queries is None:
                if self.ctgan:
                    queries = self.generate_queries(n, original_probs=True)
                else:
                    queries = self.generate_queries(n)
            gen_input = [noise, queries]
        else:
            gen_input = [noise]
        return gen_input

    def sample(self, n=None, queries=None, n_repeat=None):
        """
        Function for generating data using the data synthesizer
        """
        n, queries = self._sample_interpret_input(n=n, queries=queries, n_repeat=n_repeat)
        gen_input = self._sample_get_gen_input(n=n, queries=queries)
        gen_data_num_scaled, gen_data_discrete_oh = self.generator.predict(gen_input)
        return self.inv_data_transform(gen_data_num_scaled, gen_data_discrete_oh)

    def sample_scaled(self, n=None, queries=None, n_repeat=None):
        """
        Function for generating data directly from the generator without inverting any transformations. Mostly used for debugging.
        """
        n, queries = self._sample_interpret_input(n=n, queries=queries, n_repeat=n_repeat)
        gen_input = self._sample_get_gen_input(n=n, queries=queries)

        gen_data_num_scaled, gen_data_discrete_oh = self.generator.predict(gen_input)
        columns_discrete_oh = []
        for i, col in enumerate(self.columns_discrete):
            for category in self.oh_encoder.categories_[i]:
                columns_discrete_oh.append(col + ":" + category)
        return pd.concat((pd.DataFrame(gen_data_num_scaled, columns=self.columns_num),
                          pd.DataFrame(gen_data_discrete_oh, columns=columns_discrete_oh)), axis=1)

    def create_critic(self):
        """
        Internal function for creating the critic neural network. Uses input parameters given to TabGAN to decide
        between different critic architectures
        """
        input_numeric = Input(shape=(self.n_columns_num * self.pac), name="Numeric_input")
        input_discrete = Input(shape=(self.n_columns_discrete_oh * self.pac), name="Discrete_input")
        if self.use_query:
            query = Input(shape=(self.n_columns_query * self.pac), name="Categorical Query" if self.ctgan else "Query")
            combined1 = concatenate([input_numeric, input_discrete, query], name="Combining_input")
            inputs = [input_numeric, input_discrete, query]
        else:
            combined1 = concatenate([input_numeric, input_discrete], name="Combining_input")
            inputs = [input_numeric, input_discrete]
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
        if self.use_query:
            query = Input(shape=self.n_columns_query, name="Categorical Query" if self.ctgan else "Query")
            combined1 = concatenate([latent, query], name="Concatenate_input")
            inputs = [latent, query]
        else:
            combined1 = latent
            inputs = [latent]

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

    def generate_latent_func(self, n):
        """
        Internal function for generating latent noise as input for generator
        """
        return tf.random.normal([n, self.dim_latent])

    def gumbel_softmax(self, logits):
        """
        Internal function for used in creating of gumbel softmax layers
        """
        return tfd.RelaxedOneHotCategorical(temperature=self.gumbel_temperature, logits=logits).sample()

    def train_step_func(self, n_batch, ret_loss=False):
        """
        Internal function for running training for a single batch.
        """
        queries = None
        for i in range(self.n_critic):
            #data_batch_real = self.get_data_batch_real()
            if self.use_query:
                if self.ctgan:
                    queries, query_ids = self.generate_queries(n_batch, ret_query_id=True,
                                                               original_probs=False)
                    data_batch_real = self.get_numpy_data_batch_real_from_queries(query_ids)
                else:
                    queries = self.generate_queries(n_batch)
                    data_batch_real = self.get_numpy_data_batch_real_from_queries(queries)
            else:
                if self.tf_data_use:
                    data_batch_real = next(self.data_processed_iter)
                else:
                    data_batch_real = self.get_numpy_data_batch_real(n_batch)
            self.train_step_critic(data_batch_real, n_batch, queries=queries)

        if ret_loss:
            em_distance = self.compute_em_distance()

        if self.use_query:
            if self.ctgan:
                queries = self.generate_queries(n_batch, original_probs=False)
            else:
                queries = self.generate_queries(n_batch)
        loss_gen = self.train_step_generator(n_batch, queries=queries)
        if ret_loss:
            return em_distance, loss_gen
        else:
            return None, None


    def train_step_critic_func(self, data_batch_real, n_batch, queries=None):
        noise = self.generate_latent(n_batch)
        noise_and_query = [noise, queries] if self.use_query else noise
        data_batch_gen = self.generator(noise_and_query, training=True)

        n_batch_pac = n_batch // self.pac
        if self.pac > 1:
            if self.use_query:
                queries = tf.reshape(queries, (n_batch_pac, self.n_columns_query * self.pac))
            data_batch_gen = [
                tf.reshape(data_batch_gen[0], (n_batch_pac, self.n_columns_num * self.pac)),
                tf.reshape(data_batch_gen[1], (n_batch_pac, self.n_columns_discrete_oh * self.pac))
            ]
            data_batch_real = [
                tf.reshape(data_batch_real[0], (n_batch_pac, self.n_columns_num * self.pac)),
                tf.reshape(data_batch_real[1], (n_batch_pac, self.n_columns_discrete_oh * self.pac))
            ]


        with tf.GradientTape() as discr_tape:
            output_discr_real = self.critic(
                [data_batch_real[0], data_batch_real[1], queries] if self.use_query else data_batch_real,
                training=True
            )
            output_discr_fake = self.critic(
                [data_batch_gen[0], data_batch_gen[1], queries] if self.use_query else data_batch_gen,
                training=True
            )
            loss_discr = self.calc_loss_discr(real_output=output_discr_real,
                                              fake_output=output_discr_fake)

            epsilon = tf.random.uniform([n_batch_pac, 1])
            combined_data_num = epsilon * data_batch_gen[0] + (1 - epsilon) * data_batch_real[0]
            combined_data_discrete = epsilon * data_batch_gen[1] + (1 - epsilon) * data_batch_real[1]
            if self.use_query:
                combined_data = [combined_data_num, combined_data_discrete, queries]
            else:
                combined_data = [combined_data_num, combined_data_discrete]

            with tf.GradientTape() as discr_tape_comb:
                discr_tape_comb.watch(combined_data)
                loss_discr_combined = self.critic(combined_data, training=True)
            combined_gradients = discr_tape_comb.gradient(loss_discr_combined, combined_data)
            combined_gradients = tf.concat(combined_gradients, axis=1)

            gradient_penalty = tf.norm(combined_gradients, axis=1) - 1
            if self.gan_method == "WGAN-SGP":
                gradient_penalty = tf.maximum(gradient_penalty, 0)
            loss_discr_gradients = self.wgan_lambda * tf.reduce_mean(gradient_penalty ** 2)
            loss_discr_combined = loss_discr + loss_discr_gradients
        gradients_of_critic = discr_tape.gradient(loss_discr_combined,
                                                  self.critic.trainable_variables)
        self.critic_optimizer.apply_gradients(
            zip(gradients_of_critic, self.critic.trainable_variables))

        return None

    def calc_loss_discr(self, real_output, fake_output):
        return - tf.reduce_mean(real_output) + tf.reduce_mean(fake_output)

    def calc_loss_generator(self, fake_output, **kwargs):
        return - tf.reduce_mean(fake_output)

    def compute_em_distance_func(self):
        noise = self.generate_latent(self.data.shape[0])
        if self.use_query:
            if self.ctgan:
                queries = self.generate_queries(self.data.shape[0], original_probs=True)
            else:
                queries = self.generate_queries(self.data.shape[0])
            noise_and_queries = [noise, queries]
        else:
            noise_and_queries = noise
        gen_data_num, gen_data_discrete = self.generator(noise_and_queries, training=True)

        if self.pac > 1:
            n_batch_pac = n_batch // self.pac
            if self.use_query:
                queries = tf.reshape(queries, shape=(n_batch_pac, self.n_columns_query * self.pac))
            gen_data_num = tf.reshape(gen_data_num, shape=(n_batch_pac, self.n_columns_num * self.pac))
            gen_data_discrete = tf.reshape(gen_data_discrete,
                                           shape=(n_batch_pac, self.n_columns_discrete_oh * self.pac))
            data_num_scaled = tf.reshape(self.data_num_scaled, shape=(n_batch_pac, self.n_columns_num * self.pac))
            data_discrete_oh = tf.reshape(self.data_discrete_oh,
                                          shape=(n_batch_pac, self.n_columns_discrete_oh * self.pacto))
        else:
            data_num_scaled = self.data_num_scaled
            data_discrete_oh = self.data_discrete_oh

        output_discr_real = self.critic(
            [data_num_scaled, data_discrete_oh, queries] if self.use_query
            else [data_num_scaled, data_discrete_oh], training=True)
        output_discr_fake = self.critic(
            [gen_data_num, gen_data_discrete, queries] if self.use_query else [gen_data_num, gen_data_discrete],
            training=True)
        em_distance = self.calc_loss_discr(real_output=output_discr_real, fake_output=output_discr_fake)
        return em_distance

    def train_step_generator_func(self, n_batch, queries=None):
        noise = self.generate_latent(n_batch)

        with tf.GradientTape() as gen_tape:
            gen_data_num, gen_data_discrete = self.generator(
                [noise, queries] if self.use_query else noise, training=True
            )
            tf.print(gen_data_num)
            tf.print(gen_data_discrete)

            if self.pac > 1:
                n_batch_pac = n_batch // self.pac
                if self.use_query:
                    queries = tf.reshape(queries, shape=(n_batch_pac, self.n_columns_query * self.pac))
                gen_data_num = tf.reshape(gen_data_num, shape=(n_batch_pac, self.n_columns_num * self.pac))
                gen_data_discrete = tf.reshape(gen_data_discrete,
                                               shape=(n_batch_pac, self.n_columns_discrete_oh * self.pac))

            fake_output = self.critic(
                [gen_data_num, gen_data_discrete, queries] if self.use_query else [gen_data_num, gen_data_discrete],
                training=True)
            loss_gen = self.calc_loss_generator(fake_output=fake_output,
                                                gen_data_num=gen_data_num,
                                                gen_data_discrete=gen_data_discrete,
                                                queries=queries)
            tf.print(loss_gen)
            if self.ctgan and self.ctgan_binomial_loss:
                # tf.print(tf.reduce_sum(queries * gen_data_discrete, axis=1),
                #          "min_val:", tf.math.reduce_min(tf.reduce_sum(queries * gen_data_discrete, axis=1)),
                #          "sum:", tf.math.reduce_sum(tf.math.round(tf.reduce_sum(queries * gen_data_discrete, axis=1))),
                #          tf.math.log(tf.reduce_sum(queries * gen_data_discrete, axis=1)),
                #          "min_val:", tf.math.reduce_min(tf.math.log(tf.reduce_sum(queries * gen_data_discrete, axis=1))),
                #          "total_loss:", tf.reduce_mean(tf.math.log(tf.reduce_sum(queries * gen_data_discrete, axis=1)))
                #          )
                category_same_as_query = tf.reduce_sum(queries * gen_data_discrete, axis=1)
                if self.ctgan_binomial_distance_floor > 0:
                    category_same_as_query = tf.minimum(category_same_as_query, 1 - self.ctgan_binomial_distance_floor)
                loss_gen -= tf.reduce_mean(tf.math.log(category_same_as_query + 1e-22))
                #category_same_as_query = tf.minimum(tf.reduce_sum(queries * gen_data_discrete, axis=1), 0.99)
                #loss_gen -= tf.reduce_mean(tf.math.log(category_same_as_query))


        gradients_of_generator = gen_tape.gradient(loss_gen, self.generator.trainable_variables)
        self.generator_optimizer.apply_gradients(zip(gradients_of_generator, self.generator.trainable_variables))

        return loss_gen

    def get_data_batch_real(self):
        """Currently not in use"""
        return next(self.data_processed_iter)

    def get_numpy_data_batch_real_func(self, n_batch):
        if self.np_data_fix:
            ix = tf.py_function(lambda: next(self.np_ix_iter),
                                inp=[], Tout=tf.int64)
        else:
            ix = tf.random.uniform(shape=[n_batch], minval=0, maxval=self.data.shape[0], dtype=tf.dtypes.int64)
        return tf.py_function(lambda ixs: [self.data_num_scaled_cast[ixs], self.data_discrete_oh_cast[ixs]],
                          inp=[ix], Tout=[tf.float32, tf.float32])

    def train(self, n_epochs=None, batch_size=None, restart_training=False, progress_bar=False, progress_bar_desc=None,
              plot_loss=False, plot2D_image=False, plot2D_image_real_time=False, plot_time=False, plot_loss_type="scatter",
              plot_loss_real_time=False,
              plot_loss_update_every=1, plot_loss_title=None, ckpt_every=None, tf_make_graph=None,
              save_dir=None, filename_plot_loss=None, filename_plot2D=None, save_loss=False, plot_loss_incl_generator_loss=False,
              plot2D_num_cols=[0, 1], plot2D_discrete_col=None, plot2D_color_opacity=0.5, plot2D_n_save_img=20,
              plot2D_save_int=None, plot2D_background_func=None, plot2D_n_img_horiz=5, plot2D_inv_scale=True,
              plot2D_n_test=None,
              tf_profile_train_step_range=[-1, -1], tf_profile_log_dir="log_tabGAN"):
        """
        Function for training the data synthesizer (training the GAN architecture).
        """
        if n_epochs is None:
            if self.default_epochs_to_train is None:
                raise ValueError("Number of epochs to run must be entered as a parameter if no default value is given when creating class object.")
            else:
                n_epochs = self.default_epochs_to_train
        if tf_make_graph is None:
            tf_make_graph = self.tf_make_graph

        if plot_loss_real_time and plot2D_image_real_time:
            raise ValueError("plot_loss_real_time and plot2D_image_real_time can not both be True at the same time")

        if batch_size is None:
            batch_size = self.batch_size
        if batch_size % self.pac != 0:
            raise ValueError(f"Batch size ({batch_size}) must be a multiple of pac size ({self.pac})."
                             " Please change one or both of these parameters")

        if plot2D_save_int != None:
            plot2D_save_epochs = np.arange(0, n_epochs, plot2D_save_int)
            plot2D_n_save_img = len(save_epochs)
        else:
            plot2D_save_epochs = np.linspace(0, n_epochs, plot2D_n_save_img).astype(int)

        if plot2D_n_test == None:
            plot2D_n_test = self.nrow

        if (not self.ckpt_dir is None) and (ckpt_every is None):
            if (self.ckpt_every == None):
                ckpt_every = n_epochs
            else:
                ckpt_every = self.ckpt_every

        if restart_training or not self.initialized_gan:
            self.initialize_gan(tf_make_graph=tf_make_graph)

        if restart_training and self.ckpt_dir is not None:
            shutil.rmtree(self.ckpt_dir)
            os.makedirs(self.ckpt_dir, exist_ok=True)
            self.initialize_cptk()

        if all(isinstance(col, int) for col in plot2D_num_cols):
            plot2D_num_cols = self.columns_num[plot2D_num_cols]

        if isinstance(plot2D_discrete_col, int):
            plot2D_discrete_col = self.columns[plot2D_discrete_col]

        batch_per_epoch = int(self.nrow / batch_size)

        plot2D_n_img_vert = ceil(plot2D_n_save_img / plot2D_n_img_horiz)

        if plot2D_image or plot2D_image_real_time or filename_plot2D:
            hdisplay = display("", display_id=True)
            fig_plot2D, axes_plot2D = plt.subplots(plot2D_n_img_vert, plot2D_n_img_horiz,
                                                   figsize=(16, 16 / plot2D_n_img_horiz * plot2D_n_img_vert),
                                                   squeeze=False)
            noise_test = self.generate_latent(plot2D_n_test)
            if self.use_query:
                queries_test = self.generate_queries(plot2D_n_test)
            else:
                queries_test = None
            plot2D_any = True
            plot2D_img_count = 0
        else:
            plot2D_any = False
        
        if plot_loss or plot_loss_real_time or filename_plot_loss:
            gen_loss_vec = np.zeros(n_epochs)
            em_distance_vec = np.zeros(n_epochs)
            plot_loss_any = True
            if plot_loss_real_time:
                hdisplay = display("", display_id=True)
                fig_loss, ax_loss = plt.subplots(1, 1, figsize=(12, 6))
                ax_loss.hlines(0, self.start_epoch, self.start_epoch + n_epochs, color="black", linestyle="dashed")
        else:
            plot_loss_any = False

        if plot_time:
            time_epoch_vec = np.zeros(n_epochs + 1)

        if self.np_data_fix:
            n_obs_needed = n_epochs * batch_per_epoch * self.n_critic * batch_size
            np_ix_list = []
            for i in range(ceil(n_obs_needed / self.data.shape[0])):
                np_ix_list = np.concatenate((np_ix_list,
                                             np.random.choice(self.data.shape[0], size=self.data.shape[0],
                                                              replace=False)),
                                            axis=0)
            self.np_ix_iter = iter(np_ix_list[:n_obs_needed].reshape((n_epochs * batch_per_epoch *
                                                                           self.n_critic,
                                                                           batch_size)))

        epochs = np.arange(self.start_epoch + 1, self.start_epoch + n_epochs + 1)

        with tqdm(total=n_epochs, leave=False, disable=not progress_bar, desc=progress_bar_desc) as pbar:
            # manually enumerate epochs
            for epoch in range(0, n_epochs + 1):
                if plot_time:
                    time_before_epoch = time.perf_counter()
                if epoch > 0:
                    em_distance = g_loss = 0
                    if tf_profile_train_step_range[0] == epoch:
                        tf.profiler.experimental.start(tf_profile_log_dir)
                    for batch in range(batch_per_epoch):
                        with tf.profiler.experimental.Trace('train_step', step_num=epoch, _r=1):
                            em_distance_batch, gen_loss_batch = self.train_step(batch_size,
                                                                                ret_loss=plot_loss_any)
                        if plot_loss_any:
                            g_loss += gen_loss_batch
                            em_distance += em_distance_batch
                    if plot_loss_any:
                        g_loss /= batch_per_epoch
                        em_distance /= batch_per_epoch
                        gen_loss_vec[epoch - 1] = g_loss
                        em_distance_vec[epoch - 1] = em_distance

                    if tf_profile_train_step_range[1] == epoch:
                        tf.profiler.experimental.stop(tf_profile_log_dir)
                    
                    if plot_loss_real_time:
                        if (epoch % plot_loss_update_every == 0 or epoch in [1, n_epochs]):
                            if (epoch > 1):
                                scatter_critic.remove()
                                if plot_loss_incl_generator_loss:
                                    scatter_gen.remove()

                            scatter_critic = ax_loss.scatter(epochs[:(epoch - 1)], em_distance_vec[:(epoch - 1)],
                                                            color="red", label="EM distance")
                            if plot_loss_incl_generator_loss:
                                scatter_gen = ax_loss.scatter(epochs[:(epoch - 1)], gen_loss_vec[:(epoch - 1)],
                                                              color="blue", label="Generator loss")
                            if (epoch == 1):
                                ax_loss.legend()
                            hdisplay.update(fig_loss)

                    pbar.update(1)

                if not self.ckpt_dir is None:
                    if epoch % ckpt_every == 0 or epoch == n_epochs:
                        if os.path.exists(os.path.join(self.ckpt_dir, "checkpoint")):
                            os.remove(os.path.join(self.ckpt_dir, "checkpoint"))
                        self.ckpt_manager.save(self.ckpt.epoch.numpy())
                    self.ckpt.epoch.assign_add(1)
                if plot2D_any:
                    if epoch in plot2D_save_epochs:
                        ax_plot2D = axes_plot2D[plot2D_img_count // plot2D_n_img_horiz, plot2D_img_count % plot2D_n_img_horiz]

                        if plot2D_background_func is not None:
                            plot2D_background_func(ax_plot2D)

                        self.plot2D_axis_update(ax=ax_plot2D, latent_vec=noise_test, queries=queries_test,
                                                inv_scale=plot2D_inv_scale,
                                                num_cols=plot2D_num_cols, discrete_col=plot2D_discrete_col,
                                                legend=(plot2D_img_count == 0),
                                                color_opacity=plot2D_color_opacity)

                        ax_plot2D.set_title("Epoch %d/%d" % (epoch, n_epochs))
                        if plot2D_img_count == 0:
                            plt.tight_layout()
                        if plot2D_image_real_time:
                            hdisplay.update(fig_plot2D)
                        plot2D_img_count += 1

                if plot_time:
                    time_epoch_vec[epoch] = time.perf_counter() - time_before_epoch

        if plot_loss_real_time:
            plt.close(fig_loss)
        
        if plot_loss or filename_plot_loss:
            fig_loss, ax_loss = plt.subplots(1, 1, figsize=(12, 6))
            ax_loss.hlines(0, self.start_epoch, self.start_epoch + n_epochs, color="black", linestyle="dashed")
            if plot_loss_type == "scatter":
                plot_loss_func = ax_loss.scatter
            elif plot_loss_type == "line":
                plot_loss_func = ax_loss.plot
            else:
                raise Exception("Unknown plot_loss_type. Only scatter and line implemented.")
            ax_loss.scatter(epochs, em_distance_vec, color="red", label="EM distance")
            if plot_loss_incl_generator_loss:
                ax_loss.scatter(epochs, gen_loss_vec, color="blue", label="Generator loss")
            ax_loss.legend()
            plt.close(fig_loss)

        if not (plot_loss or plot_time or plot_time):
            return None

        return_figures = ()

        if plot_loss:
            return_figures += (fig_loss,)

        if plot2D_image_real_time:
            plt.close(fig_plot2D)

        if plot2D_image:
            return_figures += (fig_plot2D,)

        if plot_time:
            epochs = np.arange(self.start_epoch, self.start_epoch + n_epochs + 1)
            fig_plot_time = plt.figure()
            plt.plot(epochs, time_epoch_vec)

            plt.title("Time for each epoch")
            plt.close(fig_plot_time)
            return_figures += (fig_plot_time,)

        if not save_dir is None:
            if not filename_plot2D is None:
                save_path = os.path.join(save_dir, filename_plot2D)
                os.makedirs(save_dir, exist_ok=True)
                fig_plot2D.savefig(save_path)

            if not filename_plot_loss is None:
                save_path = os.path.join(save_dir, filename_plot_loss)
                os.makedirs(save_dir, exist_ok=True)
                fig_loss.savefig(save_path)

        return return_figures

    def plot2D_axis_update(self, ax, latent_vec, num_cols, discrete_col=None, queries=None, inv_scale=True,
                           legend=True, color_opacity=1):
        gen_data_num, gen_data_discrete = self.generator([latent_vec, queries] if self.use_query
                                                         else [latent_vec])

        if inv_scale:
            gen_data = self.inv_data_transform(gen_data_num, gen_data_discrete)
            if discrete_col is None:
                color_dict = {"all": next(ax._get_lines.prop_cycler)['color']}
                color = None
            else:
                labels_unique = np.sort(np.unique(gen_data[discrete_col]))
                colors_unique = [next(ax._get_lines.prop_cycler)['color'] for label in labels_unique]
                color_dict = {label: color for label, color in zip(labels_unique, colors_unique)}
                color = gen_data[discrete_col].map(color_dict)

                if legend:
                    markers = [plt.Line2D([0, 0], [0, 0], color=color, marker='o', linestyle='', alpha=0.5)
                               for color in color_dict.values()]
                    ax.legend(markers, color_dict.keys(), numpoints=1)
        else:
            gen_data = pd.DataFrame(gen_data_num, columns=self.columns_num)
            color = None
        ax.scatter(gen_data[num_cols[0]], gen_data[num_cols[1]], c=color, alpha=color_opacity)

    def restore_checkpoint(self, epoch="latest"):
        """
        Function for restoring saved model checkpoint
        """
        if (epoch == "latest"):
            ckpt_path = self.ckpt_manager.latest_checkpoint
        else:
            ckpt_path = self.ckpt_prefix + "-" + str(epoch)
        if self.initialized_gan:
            self.ckpt.restore(ckpt_path).assert_consumed()
        else:
            self.ckpt.restore(ckpt_path).expect_partial()
        self.start_epoch = self.ckpt.epoch.numpy()

    def use_critic_on_data(self, data):
        """
        Internal function for preprocessing data and then fetching it to the critic. Mostly used for debugging purposes.
        """
        num_data = data[self.columns_num]
        discrete_data = data[self.columns_discrete]
        num_data_scaled = self.scaler_num.transform(num_data)
        discrete_data_oh = self.oh_encoder.transform(discrete_data)
        #queries_batch = tf.zeros([data.shape[0], self.n_columns_discrete_oh], dtype=tf.dtypes.float32)
        return self.critic.predict([num_data_scaled, discrete_data_oh])
