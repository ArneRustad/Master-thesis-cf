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

from .tabGAN import TabGAN

class TabGAN2(TabGAN):
    def __init__(self, data, batch_size=500,
                 n_hidden_layers=2, n_hidden_generator_layers=None, n_hidden_critic_layers=None,
                 dim_hidden=256, dim_hidden_generator=None, dim_hidden_critic=None,
                 dim_latent=128, gumbel_temperature=0.5, n_critic=5, wgan_lambda=10,
                 quantile_transformation_int=True, quantile_rand_transformation=True,
                 n_quantiles_int=1000, qtr_spread=0.4, qtr_lbound_apply=0.05, adam_amsgrad=False,
                 optimizer="adam", opt_lr=0.0002, adam_beta1=0, adam_beta2=0.999, sgd_momentum=0.0, sgd_nesterov=False,
                 rmsprop_rho=0.9, rmsprop_momentum=0, rmsprop_centered=False,
                 ckpt_dir=None, ckpt_every=None, ckpt_max_to_keep=None, ckpt_name="ckpt_epoch",
                 noise_discrete_unif_max=0, tf_make_train_step_graph=True,
                 jit_compile_train_step=False):
        super().__init__(data, batch_size=500,
                         n_hidden_layers=n_hidden_layers, n_hidden_generator_layers=n_hidden_generator_layers,
                         n_hidden_critic_layers=n_hidden_critic_layers,
                         dim_hidden=dim_hidden, dim_hidden_generator=dim_hidden_generator, dim_hidden_critic=dim_hidden_critic,
                         dim_latent=dim_latent, gumbel_temperature=gumbel_temperature, n_critic=n_critic, wgan_lambda=wgan_lambda,
                         quantile_transformation_int=quantile_transformation_int, quantile_rand_transformation=quantile_rand_transformation,
                         n_quantiles_int=n_quantiles_int, qtr_spread=qtr_spread, qtr_lbound_apply=qtr_lbound_apply, adam_amsgrad=adam_amsgrad,
                         optimizer=optimizer, opt_lr=opt_lr, adam_beta1=adam_beta1, adam_beta2=adam_beta2, sgd_momentum=sgd_momentum,
                         sgd_nesterov=sgd_nesterov,
                         rmsprop_rho=rmsprop_rho, rmsprop_momentum=rmsprop_momentum, rmsprop_centered=rmsprop_centered,
                         ckpt_dir=ckpt_dir, ckpt_every=ckpt_every, ckpt_max_to_keep=ckpt_max_to_keep, ckpt_name=ckpt_name,
                         noise_discrete_unif_max=noise_discrete_unif_max, use_query=False, tf_make_train_step_graph=tf_make_train_step_graph,
                         jit_compile_train_step=jit_compile_train_step)

        self.data_processed = tf.data.Dataset.zip((tf.data.Dataset.from_tensor_slices(self.data_num),
                                                   tf.data.Dataset.from_tensor_slices(self.data_discrete_oh)
                                                   )
                                                  )

