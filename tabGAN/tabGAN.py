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

from .fast_nondominated_sort import fast_non_dominated_sort


class TableGAN:
    """
    Class for creating a tabular GAN that can also generate counterfactual explanations through a post-processing step.
    """

    def __init__(self, data, dim_latent=128, dim_hidden=256, batch_size=500, gumbel_temperature=0.5, n_critic=5,
                 wgan_lambda=10, adam_lr=0.0002, adam_beta1=0, adam_beta2=0.999, ckpt_dir=None, ckpt_every=None,
                 ckpt_max_to_keep=None, ckpt_name="ckpt_epoch", noise_discrete_unif_max=0,
                 quantile_transformation_int=False, n_quantiles_int=1000, quantile_rand_transformation=True,
                 qtr_fraction=0.4, qtr_apply_lbound=0.05, use_query=True):
        # Initialize variables
        self.data = data
        self.columns = data.columns
        self.n_columns = len(self.columns)
        self.nrow = data.shape[0]
        self.batch_size = batch_size
        self.dim_latent = dim_latent
        self.dim_hidden = dim_hidden
        self.gumbel_temperature = gumbel_temperature
        self.adam_lr = adam_lr
        self.adam_beta1 = adam_beta1
        self.n_critic = n_critic
        self.wgan_lambda = wgan_lambda
        self.adam_lr = adam_lr
        self.adam_beta1 = adam_beta1
        self.adam_beta2 = adam_beta2
        self.ckpt_dir = ckpt_dir
        self.ckpt_every = ckpt_every
        self.ckpt_max_to_keep = ckpt_max_to_keep
        self.ckpt_name = ckpt_name
        self.ckpt_prefix = os.path.join(self.ckpt_dir, self.ckpt_name) if not self.ckpt_dir is None else None
        self.noise_discrete_unif_max = noise_discrete_unif_max
        self.quantile_transformation_int = quantile_transformation_int
        self.quantile_rand_transformation = quantile_rand_transformation
        self.n_quantiles_int = n_quantiles_int
        self.uninitialized_opt_vars = True
        self.qtr_fraction = qtr_fraction
        self.qtr_apply_lbound = qtr_apply_lbound
        self.use_query = use_query

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
        # if (self.noise_discrete_unif_max > 0):
        # noise_discrete = np.random.uniform(low = 0, high = self.noise_discrete_unif_max,
        #                                   size = self.data_discrete_oh.shape)
        # self.data_discrete_oh += noise_discrete * np.where(self.data_discrete_oh > 0.5, -1, 1)

        self.categories_len = [len(i) for i in self.oh_encoder.categories_]

        # Create Gumbel-activation function
        tf.keras.utils.get_custom_objects().update({'gumbel_softmax': Activation(self.gumbel_softmax)})

        # Create generator and discriminator objects as well as discriminator and generator optimizer
        self.initialize_gan()
        # If needed create checkpoint manager
        if (self.ckpt_dir != None):
            self.initialize_cptk()

    def initialize_gan(self):
        """
        Internal function used for initializing the GAN architecture
        """
        # Create generator and discriminator objects
        self.generator = self.create_generator()
        self.discriminator = self.create_discriminator()

        # Create optimizers for generator and discriminator
        self.generator_optimizer = tf.keras.optimizers.Adam(learning_rate=self.adam_lr, beta_1=self.adam_beta1,
                                                            beta_2=self.adam_beta2)
        self.discriminator_optimizer = tf.keras.optimizers.Adam(learning_rate=self.adam_lr, beta_1=self.adam_beta1,
                                                                beta_2=self.adam_beta2)
        self.start_epoch = 0

        self.train_step = tf.function(self.train_step_func)

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
                if (n_curr_references >= self.qtr_apply_lbound * self.n_quantiles_int):
                    mask = self.data_num[self.columns_int[i]] == integer
                    n_obs_curr = np.sum(mask)
                    curr_reference_range = curr_references[-1] - curr_references[0]
                    low = curr_references[0] + curr_reference_range * (0.5 - self.qtr_fraction / 2)
                    high = curr_references[0] + curr_reference_range * (0.5 + self.qtr_fraction / 2)
                    data[mask, col] = scipy.stats.norm.ppf(np.random.uniform(low=low, high=high, size=n_obs_curr))
        return data

    def initialize_cptk(self):
        """
        Internal function for initializing checkpoint mangager used to save the progress of the model.
        """
        os.makedirs(self.ckpt_dir, exist_ok=True)
        self.ckpt = tf.train.Checkpoint(epoch=tf.Variable(0), generator_opt=self.generator_optimizer,
                                        discriminator_opt=self.discriminator_optimizer, generator=self.generator,
                                        discriminator=self.discriminator)
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
        Function for generating data used the data synthesizer
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

    def create_discriminator(self):
        """
        Internal function for creating the critic neural network.
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
        hidden1 = Dense(self.dim_hidden, activation=LeakyReLU(), name="hidden1")(combined1)
        hidden2 = Dense(self.dim_hidden, activation=LeakyReLU(), name="hidden2")(hidden1)
        output = Dense(1, name="output_discriminator")(hidden2)
        model = Model(inputs=inputs, outputs=output)
        return (model)

    def create_generator(self):
        """
        Internal function for creating the generator neural network
        """
        latent = Input(shape=(self.dim_latent), name="Latent")
        if self.use_query:
            query = Input(shape=(self.n_columns_discrete_oh), name="Query")
            combined1 = concatenate([latent, query], name="Concatenate_input")
            inputs = [latent, query]
        else:
            combined1 = latent
            inputs = [latent]
        hidden1 = Dense(self.dim_hidden, activation=LeakyReLU(), name="Hidden1")(combined1)
        hidden2 = Dense(self.dim_hidden, activation=LeakyReLU(), name="Hidden2")(hidden1)

        if (self.n_columns_discrete == 0):
            raise Exception("tableGAN not yet implemented for zero discrete columns")
        elif (self.n_columns_discrete == 1):
            output_discrete_i = Dense(self.categories_len[0], name="%s_output" % self.columns_discrete[0])(hidden2)
            output_discrete = Activation("gumbel_softmax", name="Gumbel_softmax")(output_discrete_i)
        else:
            output_discrete_sep = []
            for i in range(self.n_columns_discrete):
                output_discrete_i = Dense(self.categories_len[i], name="%s_output" % self.columns_discrete[i])(hidden2)
                output_discrete_sep.append(
                    Activation("gumbel_softmax", name="Gumbel_softmax%d" % (i + 1))(output_discrete_i))

            output_discrete = concatenate(output_discrete_sep, name="Discrete_output")

        output_numeric = Dense(self.n_columns_num, name="Numeric_output")(hidden2)
        model = Model(inputs=inputs, outputs=[output_numeric, output_discrete])
        return (model)

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
                output_discr_real = self.discriminator([data_num_batch, data_discrete_oh_batch, queries_batch],
                                                       training=True)
                output_discr_fake = self.discriminator([gen_data_num, gen_data_discrete, queries_batch], training=True)
                loss_discr = - tf.reduce_mean(output_discr_real) + tf.reduce_mean(output_discr_fake)

                epsilon = tf.random.uniform([n_batch, 1])
                combined_data_num = epsilon * gen_data_num + (1 - epsilon) * data_num_batch
                combined_data_discrete = epsilon * gen_data_discrete + (1 - epsilon) * data_discrete_oh_batch

                with tf.GradientTape() as discr_tape_comb:
                    discr_tape_comb.watch(combined_data_num)
                    discr_tape_comb.watch(combined_data_discrete)
                    discr_tape_comb.watch(queries_batch)
                    loss_discr_combined = self.discriminator([combined_data_num, combined_data_discrete, queries_batch],
                                                             training=True)
                combined_gradients = discr_tape_comb.gradient(loss_discr_combined,
                                                              [combined_data_num, combined_data_discrete,
                                                               queries_batch])
                combined_gradients = tf.concat(combined_gradients, axis=1)

                loss_discr_gradients = self.wgan_lambda * tf.reduce_mean((tf.norm(combined_gradients, axis=1) - 1) ** 2)
                loss_discr_combined = loss_discr + loss_discr_gradients
            gradients_of_discriminator = discr_tape.gradient(loss_discr_combined,
                                                             self.discriminator.trainable_variables)
            self.discriminator_optimizer.apply_gradients(
                zip(gradients_of_discriminator, self.discriminator.trainable_variables))

        queries = tf.zeros([self.data.shape[0], self.n_columns_discrete_oh], dtype=tf.dtypes.float32)
        noise = self.generate_latent(self.data.shape[0])
        gen_data_num, gen_data_discrete = self.generator([noise, queries], training=True)
        output_discr_real = self.discriminator([self.data_num_scaled, self.data_discrete_oh, queries], training=True)
        output_discr_fake = self.discriminator([gen_data_num, gen_data_discrete, queries], training=True)
        em_distance = tf.reduce_mean(output_discr_real) - tf.reduce_mean(output_discr_fake)

        noise = self.generate_latent(n_batch)
        with tf.GradientTape() as gen_tape:
            gen_data_num, gen_data_discrete = self.generator([noise, queries_batch], training=True)
            loss_gen = - tf.reduce_mean(
                self.discriminator([gen_data_num, gen_data_discrete, queries_batch], training=True))

        gradients_of_generator = gen_tape.gradient(loss_gen, self.generator.trainable_variables)
        self.generator_optimizer.apply_gradients(zip(gradients_of_generator, self.generator.trainable_variables))

        return em_distance, loss_gen

    def train(self, n_epochs, batch_size=None, restart_training=False, plot2D_image=False, plot2D_num_cols=[0, 1],
              plot2D_discrete_col=None, plot2D_color_opacity=0.5, tot_save_img=20, plot2D_save_int=None,
              plot2D_background_func=None, n_img_horiz=5, plot2D_inv_scale=True, plot_loss=True, plot_loss_return=None,
              loss_plot_type="scatter", loss_plot_update_every=1, save_path=None, title_with_loss=False,
              progress_bar=False, progress_bar_desc=None, n_test=None, ckpt_every=None, time_plot=False, save_dir=None,
              filename_train_loss="train_loss.jpg", filename_plot2D="train_plot2D.jpg", save_loss=False,
              plot_train_loss_both=False):
        """
        Function for training the data synthesizer (training the GAN architecture).
        """
        if plot_loss and plot2D_image:
            raise ValueError("plot_loss and plot2D_image can not both be True at the same time")

        if (batch_size == None):
            batch_size = self.batch_size
        if (plot2D_save_int != None):
            plot2D_save_epochs = np.arange(0, n_epochs, plot2D_save_int)
            tot_save_img = len(save_epochs)
        else:
            plot2D_save_epochs = np.linspace(0, n_epochs, tot_save_img).astype(int)

        if plot_loss_return is None:
            if plot_loss:
                plot_loss_return = True
            else:
                plot_loss_return = False

        if (n_test == None):
            n_test = self.nrow

        if (self.ckpt_dir != None and ckpt_every == None):
            if (self.ckpt_every == None):
                ckpt_every = n_epochs
            else:
                ckpt_every = self.ckpt_every

        if restart_training:
            if not self.uninitialized_opt_vars:
                self.initialize_gan()
            if not self.ckpt_dir is None:
                shutil.rmtree(self.ckpt_dir)
                os.makedirs(self.ckpt_dir, exist_ok=True)
                self.initialize_cptk()

        self.uninitialized_opt_vars = False

        batch_per_epoch = int(self.nrow / batch_size)

        n_img_vert = ceil(tot_save_img / n_img_horiz)

        fig_loss, ax_loss = plt.subplots(1, 1, figsize=(12, 6))
        ax_loss.hlines(0, self.start_epoch, self.start_epoch + n_epochs, color="black", linestyle="dashed")

        if plot2D_image:
            fig_plot2D = plt.figure(figsize=(16, 16 / n_img_horiz * n_img_vert))
            # fig_plot2D.suptitle('Visalization of counterfactual generator training for %d epochs' % n_epochs)
            noise_test = self.generate_latent(n_test)
            queries_test = tf.zeros([n_test, self.n_columns_discrete_oh])

        if plot_loss:
            hdisplay = display("", display_id=True)

        if time_plot:
            time_epoch_vec = np.zeros(n_epochs + 1)

        gen_loss_vec = np.zeros(n_epochs)
        em_distance_vec = np.zeros(n_epochs)
        epochs = np.arange(self.start_epoch + 1, self.start_epoch + n_epochs + 1)

        img_count = 1
        with tqdm(total=n_epochs, leave=False, disable=not progress_bar, desc=progress_bar_desc) as pbar:
            # manually enumerate epochs
            for epoch in range(0, n_epochs + 1):
                if (time_plot):
                    time_before_epoch = time.perf_counter()
                if (epoch > 0):
                    em_distance = g_loss = 0
                    for batch in range(batch_per_epoch):
                        em_distance_batch, gen_loss_batch = self.train_step(batch_size)
                        g_loss += gen_loss_batch
                        em_distance += em_distance_batch
                    g_loss /= batch_per_epoch
                    em_distance /= batch_per_epoch

                    gen_loss_vec[epoch - 1] = g_loss
                    em_distance_vec[epoch - 1] = em_distance

                    if plot_loss:
                        if (epoch % loss_plot_update_every == 0 or epoch in [1, n_epochs]):
                            if (epoch > 1):
                                scatter_discr.remove()
                                if plot_train_loss_both:
                                    scatter_gen.remove()

                            scatter_discr = ax_loss.scatter(epochs[:(epoch - 1)], em_distance_vec[:(epoch - 1)],
                                                            color="red", label="EM distance")
                            if plot_train_loss_both:
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
                        ax_plot2D = fig_plot2D.add_subplot(n_img_vert, n_img_horiz, img_count)
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

                if (time_plot):
                    time_epoch_vec[epoch] = time.perf_counter() - time_before_epoch

        plt.close(fig_loss)
        fig_loss, ax_loss = plt.subplots(1, 1, figsize=(12, 6))
        ax_loss.hlines(0, self.start_epoch, self.start_epoch + n_epochs, color="black", linestyle="dashed")
        if (loss_plot_type == "scatter"):
            ax_loss.scatter(epochs, em_distance_vec, color="red", label="EM distance")
            if plot_train_loss_both:
                ax_loss.scatter(epochs, gen_loss_vec, color="blue", label="Generator loss")
        elif loss_plot_type == "line":
            ax_loss.plot(epochs, em_distance_vec, color="red", label="EM distance")
            if plot_train_loss_both:
                ax_loss.plot(epochs, gen_loss_vec, color="blue", label="Generator loss")
        else:
            raise Exception("Unknown loss_plot_type. Only scatter and line implemented.")
        ax_loss.legend()

        plt.close(fig_loss)

        if not (plot_loss_return or time_plot or time_plot):
            return None

        return_figures = ()

        if plot_loss_return:
            return_figures += (fig_loss,)

        if (plot2D_image):
            plt.close(fig_plot2D)
            return_figures += (fig_plot2D,)

        if (time_plot):
            epochs = np.arange(self.start_epoch, self.start_epoch + n_epochs + 1)
            fig_time_plot = plt.figure()
            plt.plot(epochs, time_epoch_vec)
            plt.title("Time for each epoch")
            plt.close(fig_time_plot)
            return_figures += (fig_time_plot,)

        if not save_dir is None:
            if plot2D_image and (not filename_plot2D is None):
                save_path = os.path.join(save_dir, filename_plot2D)
                os.makedirs(save_dir, exist_ok=True)
                fig_plot2D.savefig(save_path)

            if (plot_loss or save_loss) and not filename_train_loss is None:
                save_path = os.path.join(save_dir, filename_train_loss)
                os.makedirs(save_dir, exist_ok=True)
                fig_loss.savefig(save_path)

        return (return_figures)

    def restore_checkpoint(self, epoch="latest"):
        """
        Function for restoring saved model checkpoint
        """
        if (epoch == "latest"):
            ckpt_path = self.ckpt_manager.latest_checkpoint
        else:
            ckpt_path = self.ckpt_prefix + "-" + str(epoch)
        if self.uninitialized_opt_vars:
            self.ckpt.restore(ckpt_path).expect_partial()
        else:
            self.ckpt.restore(ckpt_path).assert_consumed()
        self.start_epoch = self.ckpt.epoch.numpy()

    def use_critic_on_data(self, data):
        """
        Internal function for preprocessing data and then fetching it to the critic. Mostly used for debugging purposes.
        """
        num_data = data[self.columns_num]
        discrete_data = data[self.columns_discrete]
        num_data_scaled = self.scaler_num.transform(num_data)
        discrete_data_oh = self.oh_encoder.transform(discrete_data)
        queries_batch = tf.zeros([data.shape[0], self.n_columns_discrete_oh], dtype=tf.dtypes.float32)
        return (self.discriminator.predict([num_data_scaled, discrete_data_oh, queries_batch]))

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
                "Not enough valid counterfactuals were generated,"\
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
