import pandas as pd
import numpy as np
import os
import shutil
import scipy
import tensorflow as tf

from sklearn.preprocessing import StandardScaler, OneHotEncoder, QuantileTransformer
from sklearn.compose import ColumnTransformer

from math import ceil
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, Flatten, concatenate, LeakyReLU, ReLU, Embedding, Activation

from IPython.display import clear_output, display, Image, Video
import matplotlib.pyplot as plt

from tqdm.auto import tqdm

import tensorflow_probability as tfp
tfd = tfp.distributions


class TabGAN:
    """
    Class for creating a tabular GAN that can also generate counterfactual explanations through a post-processing step.
    """
    def __init__(self, data, batch_size=500,
                 n_hidden_layers=2, n_hidden_generator_layers=None, n_hidden_critic_layers=None,
                 dim_hidden=256, dim_hidden_generator=None, dim_hidden_critic=None,
                 dim_latent=128, gumbel_temperature=0.5, n_critic=5, wgan_lambda=10,
                 quantile_transformation_int=True, quantile_rand_transformation=True,
                 n_quantiles_int=1000, qtr_spread=0.4, qtr_lbound_apply=0.05, adam_amsgrad=False,
                 optimizer="adam", opt_lr=0.0002, adam_beta1=0, adam_beta2=0.999, sgd_momentum=0.0, sgd_nesterov=False,
                 rmsprop_rho=0.9, rmsprop_momentum=0, rmsprop_centered=False,
                 ckpt_dir=None, ckpt_every=None, ckpt_max_to_keep=None, ckpt_name="ckpt_epoch",
                 noise_discrete_unif_max=0, use_query=True, tf_make_train_step_graph=True,
                 jit_compile_train_step=False):
        # Create variable defaults if needed
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
        # Initialize variables
        self.data = data
        self.columns = data.columns
        self.n_columns = len(self.columns)
        self.nrow = data.shape[0]
        self.batch_size = batch_size
        self.n_hidden_generator_layers = n_hidden_generator_layers
        self.n_hidden_critic_layers = n_hidden_critic_layers
        self.dim_latent = dim_latent
        self.dim_hidden_generator = dim_hidden_generator
        self.dim_hidden_critic = dim_hidden_critic
        self.gumbel_temperature = gumbel_temperature
        self.optimizer = optimizer
        self.n_critic = n_critic
        self.wgan_lambda = wgan_lambda
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
        self.quantile_rand_transformation = quantile_rand_transformation
        self.n_quantiles_int = n_quantiles_int
        self.initialized_gan = False
        self.qtr_spread = qtr_spread
        self.qtr_lbound_apply = qtr_lbound_apply
        self.use_query = use_query
        self.tf_make_train_step_graph = tf_make_train_step_graph
        self.jit_compile_train_step = jit_compile_train_step

        # Separate numeric data, fit numeric scaler and scale numeric data. Store numeric column names.
        self.data_num = data.select_dtypes(include=np.number)
        self.columns_num = self.data_num.columns
        self.n_columns_num = len(self.data_num.columns)
        self.columns_num_int_mask = self.data_num.dtypes.astype(str).str.contains("int")
        self.columns_int = self.columns_num[self.columns_num_int_mask]
        self.columns_num_int_pos = np.arange(len(self.columns_num))[self.columns_num_int_mask]
        self.columns_float = self.columns_num[np.logical_not(self.columns_num_int_mask)]
        if self.quantile_transformation_int:
            self.scaler_num = ColumnTransformer(transformers=[("float", StandardScaler(), self.columns_float), (
            "int", QuantileTransformer(n_quantiles=n_quantiles_int, output_distribution='normal'), self.columns_int)])
            self.data_num_scaled = self.scaler_num.fit_transform(self.data_num)

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
        self.n_columns_discrete = len(self.columns_discrete)

        self.oh_encoder = OneHotEncoder(sparse=False)
        self.data_discrete_oh = self.oh_encoder.fit_transform(self.data_discrete)
        self.n_columns_discrete_oh = self.data_discrete_oh.shape[1]
        if (self.noise_discrete_unif_max > 0):
            noise_discrete = np.random.uniform(low = 0, high = self.noise_discrete_unif_max,
                                              size = self.data_discrete_oh.shape)
            self.data_discrete_oh += noise_discrete * np.where(self.data_discrete_oh > 0.5, -1, 1)

        self.categories_len = [len(i) for i in self.oh_encoder.categories_]

        # Create Gumbel-activation function
        tf.keras.utils.get_custom_objects().update({'gumbel_softmax': Activation(self.gumbel_softmax)})

        # Create generator and critic objects as well as critic and generator optimizer
        self.initialize_gan()
        # If needed create checkpoint manager
        if (self.ckpt_dir != None):
            self.initialize_cptk()

    def initialize_gan(self, tf_make_train_step_graph=True):
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

        if tf_make_train_step_graph:
            if tf.__version__ < "2.5":
                jit_compile_args = {"experimental_compile" : self.jit_compile_train_step}
            else:
                jit_compile_args = {"jit_compile" : self.jit_compile_train_step}
            self.train_step = tf.function(self.train_step_func, **jit_compile_args)
        else:
            self.train_step = self.train_step_func

        self.initialized_gan = True

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
                if (n_curr_references >= self.qtr_lbound_apply * self.n_quantiles_int):
                    mask = self.data_num[self.columns_int[i]] == integer
                    n_obs_curr = np.sum(mask)
                    curr_reference_range = curr_references[-1] - curr_references[0]
                    low = curr_references[0] + curr_reference_range * (0.5 - self.qtr_spread / 2)
                    high = curr_references[0] + curr_reference_range * (0.5 + self.qtr_spread / 2)
                    data[mask, col] = scipy.stats.norm.ppf(np.random.uniform(low=low, high=high, size=n_obs_curr))
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

    def inv_data_transform(self, data_num_scaled, data_discrete_oh):
        """
        Internal function used for inverting the data transformation done in preprocessing
        """
        data_discrete = pd.DataFrame(self.oh_encoder.inverse_transform(data_discrete_oh), columns=self.columns_discrete)
        if (self.quantile_transformation_int):
            if (len(self.columns_float) > 0):
                data_float = pd.DataFrame(self.scaler_num.named_transformers_["float"].inverse_transform(
                    data_num_scaled[:, np.logical_not(self.columns_num_int_mask)]), columns=self.columns_float)
            else:
                data_float = None

            if (len(self.columns_int) > 0):
                data_int_scaled = pd.DataFrame(data_num_scaled[:, self.columns_num_int_mask], columns=self.columns_int)
                data_int = pd.DataFrame(self.scaler_num.named_transformers_["int"].inverse_transform(data_int_scaled),
                    columns=self.columns_int)
            else:
                data_int = None
        else:
            data_float = pd.DataFrame(self.scaler_num.inverse_transform(data_num_scaled), columns=self.columns_num)
            data_int = None
        return (pd.concat([data_float, data_int, data_discrete], axis=1)[self.columns])

    def generate_queries(self, n):
        """
        Currently a dummy function for generating queries. Query functionality is not yet implemented
        """
        return (tf.zeros([n, self.n_columns_discrete_oh]))

    def generate_data(self, n=None):
        """
        Function for generating data using the data synthesizer
        """
        if (n == None):
            n = self.nrow
        noise = self.generate_latent(n)
        queries = self.generate_queries(n)
        gen_data_num_scaled, gen_data_discrete_oh = self.generator.predict([noise, queries])
        return (self.inv_data_transform(gen_data_num_scaled, gen_data_discrete_oh))

    def generate_data_scaled(self, n=None):
        """
        Function for generating data directly from the generator without inverting any transformations. Mostly used for debugging.
        """
        if (n == None):
            n = self.nrow
        noise = self.generate_latent(n)
        queries = self.generate_queries(n)
        gen_data_num_scaled, gen_data_discrete_oh = self.generator.predict([noise, queries])
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
        input_numeric = Input(shape=(self.n_columns_num), name="Numeric_input")
        input_discrete = Input(shape=(self.n_columns_discrete_oh), name="Discrete_input")
        if self.use_query:
            query = Input(shape=(self.n_columns_discrete_oh), name="Query")
            combined1 = concatenate([input_numeric, input_discrete, query], name="Combining_input")
            inputs = [input_numeric, input_discrete, query]
        else:
            combined1 = concatenate([input_numeric, input_discrete], name="Combining_input")
            inputs = [input_numeric, input_discrete]
        hidden = combined1
        for i in range(self.n_hidden_critic_layers):
            hidden = Dense(self.dim_hidden_critic[i], activation=LeakyReLU(), name=f"hidden{i+1}")(hidden)
        output = Dense(1, name="output_critic")(hidden)
        model = Model(inputs=inputs, outputs=output)
        return (model)

    def create_generator(self):
        """
        Internal function for creating the generator neural network. Uses input parameters given to TabGAN to decide
        between different critic architectures. Also uses the number of discrete columns to decide between architectures
        """
        latent = Input(shape=(self.dim_latent), name="Latent")
        if self.use_query:
            query = Input(shape=(self.n_columns_discrete_oh), name="Query")
            combined1 = concatenate([latent, query], name="Concatenate_input")
            inputs = [latent, query]
        else:
            combined1 = latent
            inputs = [latent]

        hidden = combined1
        for i in range(self.n_hidden_generator_layers):
            hidden = Dense(self.dim_hidden_generator[i], activation=LeakyReLU(), name=f"hidden{i+1}")(hidden)

        if (self.n_columns_discrete == 0):
            raise ValueException("TabGAN not yet implemented for zero discrete columns")
        elif (self.n_columns_discrete == 1):
            output_discrete_i = Dense(self.categories_len[0], name="%s_output" % self.columns_discrete[0])(hidden)
            output_discrete = Activation("gumbel_softmax", name="Gumbel_softmax")(output_discrete_i)
        else:
            output_discrete_sep = []
            for i in range(self.n_columns_discrete):
                output_discrete_i = Dense(self.categories_len[i], name="%s_output" % self.columns_discrete[i])(hidden)
                output_discrete_sep.append(
                    Activation("gumbel_softmax", name="Gumbel_softmax%d" % (i + 1))(output_discrete_i))

            output_discrete = concatenate(output_discrete_sep, name="Discrete_output")

        if self.n_columns_num == 0:
            raise ValueException("TabGAN not yet implemented for zero numerical columns")
        output_numeric = Dense(self.n_columns_num, name="Numeric_output")(hidden)
        model = Model(inputs=inputs, outputs=[output_numeric, output_discrete])
        return (model)

    @tf.function
    def generate_latent(self, n):
        """
        Internal function for generating latent noise as input for generator
        """
        return (tf.random.normal([n, self.dim_latent]))

    def gumbel_softmax(self, logits):
        """
        Internal function for used in creating of gumbel softmax layers
        """
        return (tfd.RelaxedOneHotCategorical(temperature=self.gumbel_temperature, logits=logits).sample())

    def train_step_func(self, n_batch):
        """
        Internal function for running training for a single batch.
        """
        queries_batch = tf.zeros([n_batch, self.n_columns_discrete_oh], dtype=tf.dtypes.float32)

        for i in range(self.n_critic):
            noise = self.generate_latent(n_batch)
            gen_data_num, gen_data_discrete = self.generator([noise, queries_batch], training=True)

            ix = np.random.randint(low=0, high=self.nrow, size=n_batch)
            data_num_batch = self.data_num_scaled[ix]
            data_discrete_oh_batch = self.data_discrete_oh[ix]
            with tf.GradientTape() as discr_tape:
                output_discr_real = self.critic([data_num_batch, data_discrete_oh_batch, queries_batch],
                                                       training=True)
                output_discr_fake = self.critic([gen_data_num, gen_data_discrete, queries_batch], training=True)
                loss_discr = - tf.reduce_mean(output_discr_real) + tf.reduce_mean(output_discr_fake)

                epsilon = tf.random.uniform([n_batch, 1])
                combined_data_num = epsilon * gen_data_num + (1 - epsilon) * data_num_batch
                combined_data_discrete = epsilon * gen_data_discrete + (1 - epsilon) * data_discrete_oh_batch

                with tf.GradientTape() as discr_tape_comb:
                    discr_tape_comb.watch(combined_data_num)
                    discr_tape_comb.watch(combined_data_discrete)
                    discr_tape_comb.watch(queries_batch)
                    loss_discr_combined = self.critic([combined_data_num, combined_data_discrete, queries_batch],
                                                             training=True)
                combined_gradients = discr_tape_comb.gradient(loss_discr_combined,
                                                              [combined_data_num, combined_data_discrete,
                                                               queries_batch])
                combined_gradients = tf.concat(combined_gradients, axis=1)

                loss_discr_gradients = self.wgan_lambda * tf.reduce_mean((tf.norm(combined_gradients, axis=1) - 1) ** 2)
                loss_discr_combined = loss_discr + loss_discr_gradients
            gradients_of_critic = discr_tape.gradient(loss_discr_combined,
                                                             self.critic.trainable_variables)
            self.critic_optimizer.apply_gradients(
                zip(gradients_of_critic, self.critic.trainable_variables))

        queries = tf.zeros([self.data.shape[0], self.n_columns_discrete_oh], dtype=tf.dtypes.float32)
        noise = self.generate_latent(self.data.shape[0])
        gen_data_num, gen_data_discrete = self.generator([noise, queries], training=True)
        output_discr_real = self.critic([self.data_num_scaled, self.data_discrete_oh, queries], training=True)
        output_discr_fake = self.critic([gen_data_num, gen_data_discrete, queries], training=True)
        em_distance = tf.reduce_mean(output_discr_real) - tf.reduce_mean(output_discr_fake)

        noise = self.generate_latent(n_batch)
        with tf.GradientTape() as gen_tape:
            gen_data_num, gen_data_discrete = self.generator([noise, queries_batch], training=True)
            loss_gen = - tf.reduce_mean(
                self.critic([gen_data_num, gen_data_discrete, queries_batch], training=True))

        gradients_of_generator = gen_tape.gradient(loss_gen, self.generator.trainable_variables)
        self.generator_optimizer.apply_gradients(zip(gradients_of_generator, self.generator.trainable_variables))

        return em_distance, loss_gen

    def train(self, n_epochs, batch_size=None, restart_training=False, progress_bar=False, progress_bar_desc=None,
              plot_loss=False, plot2D_image=False, plot_time=False, plot_loss_type="scatter",
              plot_loss_update_every=1, plot_loss_title=None, ckpt_every=None, tf_make_train_step_graph=None,
              save_dir=None, filename_plot_loss=None, filename_plot2D=None, save_loss=False, plot_loss_incl_generator_loss=False,
              plot2D_num_cols=[0, 1], plot2D_discrete_col=None, plot2D_color_opacity=0.5, plot2D_n_save_img=20,
              plot2D_save_int=None, plot2D_background_func=None, plot2D_n_img_horiz=5, plot2D_inv_scale=True,
              plot2D_n_test=None, plot_loss_return=None,
              tf_profile_train_step_range=[-1, -1], tf_profile_log_dir="log_tabGAN"):
        """
        Function for training the data synthesizer (training the GAN architecture).
        """
        if tf_make_train_step_graph is None:
            tf_make_train_step_graph=self.tf_make_train_step_graph
            self.initialized_gan=False

        if plot_loss and plot2D_image:
            raise ValueError("plot_loss and plot2D_image can not both be True at the same time")

        if batch_size == None:
            batch_size = self.batch_size
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
            self.initialize_gan(tf_make_train_step_graph=tf_make_train_step_graph)

        if restart_training and not self.ckpt_dir is None:
            shutil.rmtree(self.ckpt_dir)
            os.makedirs(self.ckpt_dir, exist_ok=True)
            self.initialize_cptk()

        batch_per_epoch = int(self.nrow / batch_size)

        n_img_vert = ceil(plot2D_n_save_img / plot2D_n_img_horiz)

        if plot2D_image:
            fig_plot2D = plt.figure(figsize=(16, 16 / plot2D_n_img_horiz * n_img_vert))
            if not plot2D_title is None:
                fig_plot2D.suptitle(plot2D_title)
            noise_test = self.generate_latent(plot2D_n_test)
            queries_test = tf.zeros([plot2D_n_test, self.n_columns_discrete_oh])

        if plot_loss:
            hdisplay = display("", display_id=True)
            fig_loss, ax_loss = plt.subplots(1, 1, figsize=(12, 6))
            ax_loss.hlines(0, self.start_epoch, self.start_epoch + n_epochs, color="black", linestyle="dashed")
            gen_loss_vec = np.zeros(n_epochs)
            em_distance_vec = np.zeros(n_epochs)

        if plot_time:
            time_epoch_vec = np.zeros(n_epochs + 1)

        epochs = np.arange(self.start_epoch + 1, self.start_epoch + n_epochs + 1)

        img_count = 1
        with tqdm(total=n_epochs, leave=False, disable=not progress_bar, desc=progress_bar_desc) as pbar:
            # manually enumerate epochs
            for epoch in range(0, n_epochs + 1):
                if (plot_time):
                    time_before_epoch = time.perf_counter()
                if (epoch > 0):
                    em_distance = g_loss = 0
                    if tf_profile_train_step_range[0] == epoch:
                        tf.profiler.experimental.start(tf_profile_log_dir)
                    for batch in range(batch_per_epoch):
                        with tf.profiler.experimental.Trace('train_step', step_num=epoch, _r=1):
                            em_distance_batch, gen_loss_batch = self.train_step(batch_size)
                        g_loss += gen_loss_batch
                        em_distance += em_distance_batch
                    g_loss /= batch_per_epoch
                    em_distance /= batch_per_epoch

                    if tf_profile_train_step_range[1] == epoch:
                        tf.profiler.experimental.stop(tf_profile_log_dir)
                    
                    if plot_loss:
                        gen_loss_vec[epoch - 1] = g_loss
                        em_distance_vec[epoch - 1] = em_distance
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

                if self.ckpt_dir != None:
                    if epoch % ckpt_every == 0 or epoch == n_epochs:
                        if os.path.exists(os.path.join(self.ckpt_dir, "checkpoint")):
                            os.remove(os.path.join(self.ckpt_dir, "checkpoint"))
                        self.ckpt_manager.save(self.ckpt.epoch.numpy())
                    self.ckpt.epoch.assign_add(1)
                if plot2D_image:
                    if epoch in plot2D_save_epochs:
                        ax_plot2D = fig_plot2D.add_subplot(n_img_vert, plot2D_n_img_horiz, img_count)
                        gen_data_num, gen_data_discrete = self.generator([noise_test, queries_test])

                        if not plot2D_background_func is None:
                            plot2D_background_func(ax_plot2D)

                        if (plot2D_inv_scale):
                            gen_data_num = self.scaler_num.inverse_transform(gen_data_num)
                            gen_data_discrete = self.oh_encoder.inverse_transform(gen_data_discrete)
                            gen_data_discrete = pd.DataFrame(gen_data_discrete, columns=self.columns_discrete)
                            if plot2D_discrete_col is None:
                                color = None
                            else:
                                labels_unique = np.unique(gen_data_discrete[plot2D_discrete_col])
                                colors_unique = map_str_to_color(labels_unique)
                                color_dict = {label: color for label, color in zip(labels_unique, colors_unique)}
                                color = gen_data_discrete[plot2D_discrete_col].map(color_dict)

                            if img_count == 1:
                                markers = [plt.Line2D([0, 0], [0, 0], color=color, marker='o', linestyle='', alpha=0.5)
                                           for color in color_dict.values()]
                                plt.legend(markers, color_dict.keys(), numpoints=1)
                        ax_plot2D.scatter(gen_data_num[:, plot2D_num_cols[0]], gen_data_num[:, plot2D_num_cols[1]],
                                          c=color, alpha=plot2D_color_opacity)
                        ax_plot2D.set_title("Epoch %d/%d" % (epoch, n_epochs))
                        plt.tight_layout()
                        display(fig_plot2D)
                        clear_output(wait=True)
                        img_count += 1

                if (plot_time):
                    time_epoch_vec[epoch] = time.perf_counter() - time_before_epoch

        if plot_loss:
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

        if not (plot_loss_return or plot_time or plot_time):
            return None

        return_figures = ()

        if plot_loss:
            return_figures += (fig_loss,)

        if plot2D_image:
            plt.close(fig_plot2D)
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
