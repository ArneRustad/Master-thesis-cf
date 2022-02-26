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

