import os
import helpers.hp_tuning.hp_gen
import helpers.hp_tuning.hp_gen
from tabGAN import TabGAN
from src import constants as const
import numpy as np
import pandas as pd
import copy

JIT_COMPILE_TRAIN_STEP = False
N_EPOCHS = 100
BATCH_SIZE = 500

tabgan_args_dict = {
    "batch_size": BATCH_SIZE,
    "jit_compile": JIT_COMPILE_TRAIN_STEP,
    # WGAN parameters
    "n_critic": 10,
    "wgan_lambda": 10,
    # Optimizer parameters
    "optimizer": "adam",
    "opt_lr": 0.0002,
    "adam_beta1": 0.5,
    "adam_beta2": 0.999,
    # Transformation parameters
    "quantile_rand_transformation": True,
    "quantile_transformation_int": True,
    "qtr_spread": 0.8,
    "qtr_lbound_apply": 0.05,
    "max_quantile_share": 1,
    "n_quantiles_int": 1000,
    "qt_n_subsample": 1e5,
    "noise_discrete_unif_max": 0.01,
    # Neural network parameters
    "gumbel_temperature": 0.1,
    "activation_function": "LeakyReLU",
    #"gelu_approximate": True,
    "dim_hidden": 256,
    "dim_latent": 128,
    # Conditional sampling parameters
    "ctgan": False,
    # "ctgan_binomial_loss": True,
    # "ctgan_log_frequency": True,
    # Packing parameters
    "pac": 1
}

ctabgan_args_dict = {
    "batch_size": BATCH_SIZE,
    "jit_compile": JIT_COMPILE_TRAIN_STEP,
    # WGAN parameters
    "n_critic": 10,
    "wgan_lambda": 10,
    # Optimizer parameters
    "optimizer": "adam",
    "opt_lr": 0.0002,
    "adam_beta1": 0.5,
    "adam_beta2": 0.999,
    # Transformation parameters
    "quantile_rand_transformation": True,
    "quantile_transformation_int": True,
    "qtr_spread": 0.8,
    "qtr_lbound_apply": 0.05,
    "max_quantile_share": 1,
    "n_quantiles_int": 1000,
    "qt_n_subsample": 1e5,
    "noise_discrete_unif_max": 0,
    # Neural network parameters
    "gumbel_temperature": 0.5,
    "activation_function": "LeakyReLU",
    #"gelu_approximate": True,
    "dim_hidden": 256,
    "dim_latent": 128,
    # Conditional sampling parameters
    "ctgan": True,
    "ctgan_binomial_loss": True,
    "ctgan_log_frequency": True,
    # Packing parameters
    "pac": 1
}

dataset_train_path = os.path.join(const.dir.data(), "df_adult_edited_train.csv")
data_train = pd.read_csv(dataset_train_path)

def fetch_hp_info(method="ctabGAN-qtr"):
    if method == "tabGAN-qtr":
        method_args_dict = tabgan_args_dict
    elif method == "ctabGAN-qtr":
        method_args_dict = ctabgan_args_dict
    else:
        raise ValueError(f"Entered method name {method} has not yet any hyperparameter tuning info")

    hp_info = {}

    def create_tabGAN_for_qtr_spread(qtr_spread):
        temp_args_dict = copy.deepcopy(method_args_dict)
        temp_args_dict["qtr_spread"] = qtr_spread
        tg_qtr = TabGAN(data_train, **temp_args_dict)
        return tg_qtr

    hp_info["qtr_spread"] = {
        "vec": np.round(np.linspace(0, 1, 21), 2),
        "n_synthetic_datasets": 25,
        "n_epochs": N_EPOCHS,
        "tabGAN_func": create_tabGAN_for_qtr_spread,
        "batch_size": BATCH_SIZE,
        "hyperparams_subname": None
    }

    hp_info["qtr_spread_300_epochs"] = {
        "vec": np.round(np.linspace(0, 1, 21), 2),
        "n_synthetic_datasets": 25,
        "n_epochs": 300,
        "tabGAN_func": create_tabGAN_for_qtr_spread,
        "batch_size": BATCH_SIZE,
        "hyperparams_subname": None
    }

    noise_discrete_unif_max_vec_partial = np.arange(0, 0.21, 0.01).tolist() + [0.001, 0.003, 0.005, 0.007, 0.015, 0.025]
    gumbel_temp_vec_partial = [0.1, 0.2, 0.3, 0.4, 0.5, 0.7, 1]
    noise_and_gumbel_temp_vec = [(noise_discrete_unif_max, gumbel_temp)
                                 for noise_discrete_unif_max in noise_discrete_unif_max_vec_partial
                                 for gumbel_temp in gumbel_temp_vec_partial]

    def create_tabGAN_for_noise_and_gumbel_temperature(noise_discrete_unif_max, gumbel_temperature):
        temp_args_dict = copy.deepcopy(method_args_dict)
        temp_args_dict["noise_discrete_unif_max"] = noise_discrete_unif_max
        temp_args_dict["gumbel_temperature"] = gumbel_temperature
        tg_qtr = TabGAN(data_train, **temp_args_dict)
        return tg_qtr

    hp_info["oh_encoding_choices"] = {
        "vec": noise_and_gumbel_temp_vec,
        "n_synthetic_datasets": 10,
        "n_epochs": N_EPOCHS,
        "tabGAN_func": create_tabGAN_for_noise_and_gumbel_temperature,
        "batch_size": BATCH_SIZE,
        "hyperparams_subname": ["noise_discrete_unif_max", "gumbel_temp"]
    }

    noise_discrete_unif_max_vec_partial = np.round(np.arange(0, 0.21, 0.01), 3).tolist()
    ctgan_binomial_distance_floor_partial = ["0", "0.01", "noise_discrete_unif_max"]
    noise_ctgan_vec = [(noise_discrete_unif_max, ctgan_binomial_distance_floor)
                 for noise_discrete_unif_max in noise_discrete_unif_max_vec_partial
                 for ctgan_binomial_distance_floor in ctgan_binomial_distance_floor_partial]

    def create_tabGAN_for_noise_ctgan(noise_discrete_unif_max, ctgan_binomial_distance_floor):
        temp_args_dict = copy.deepcopy(method_args_dict)
        temp_args_dict["noise_discrete_unif_max"] = noise_discrete_unif_max
        if ctgan_binomial_distance_floor == "noise_discrete_unif_max":
            temp_args_dict["ctgan_binomial_distance_floor"] = ctgan_binomial_distance_floor
        else:
            temp_args_dict["ctgan_binomial_distance_floor"] = float(ctgan_binomial_distance_floor)
        tg_qtr = TabGAN(data_train, **temp_args_dict)
        return tg_qtr

    hp_info["noise_ctgan"] = {
        "vec": noise_ctgan_vec,
        "n_synthetic_datasets": 10,
        "n_epochs": N_EPOCHS,
        "tabGAN_func": create_tabGAN_for_noise_ctgan,
        "batch_size": BATCH_SIZE,
        "hyperparams_subname": ["noise_discrete_unif_max", "ctgan_binomial_distance_floor"]
    }

    gumbel_temp_vec = np.round(np.arange(0.1, 2.01, 0.1), 3).tolist()
    gumbel_temp_vec += np.round(np.arange(0.01, 0.1, 0.01), 3).tolist()

    def create_tabGAN_for_gumbel_temp(gumbel_temperature):
        temp_args_dict = copy.deepcopy(method_args_dict)
        temp_args_dict["gumbel_temperature"] = gumbel_temperature
        tg_qtr = TabGAN(data_train, **temp_args_dict)
        return tg_qtr

    hp_info["gumbel_temperature"] = {
        "vec": gumbel_temp_vec,
        "n_synthetic_datasets": 10,
        "n_epochs": N_EPOCHS,
        "tabGAN_func": create_tabGAN_for_gumbel_temp,
        "batch_size": BATCH_SIZE,
        "hyperparams_subname": None
    }

    add_connection_vec = [(False, False), (True, False), (False, True)]

    def create_tabGAN_for_add_connection(add_connection_discrete_to_num, add_connection_num_to_discrete):
        temp_args_dict = copy.deepcopy(method_args_dict)
        temp_args_dict["add_connection_discrete_to_num"] = add_connection_discrete_to_num
        temp_args_dict["add_connection_num_to_discrete"] = add_connection_num_to_discrete
        tg_qtr = TabGAN(data_train, **temp_args_dict)
        return tg_qtr

    hp_info["add_connection"] = {
        "vec": add_connection_vec,
        "n_synthetic_datasets": 25,
        "n_epochs": N_EPOCHS,
        "tabGAN_func": create_tabGAN_for_add_connection,
        "batch_size": BATCH_SIZE,
        "hyperparams_subname": ["discrete_to_num", "num_to_discrete"]
    }

    add_connection_advanced_vec = [(0, "None")]
    add_connection_advanced_vec += [(dim_hidden_connection, connection)
                                    for connection in ["discrete_to_num", "num_to_discrete"]
                                    for dim_hidden_connection in [0, 1, 5, 10, 25, 50, 100, 200]
                                    ]

    def create_tabGAN_for_add_connection_advanced(dim_hidden_connection, connection):
        temp_args_dict = copy.deepcopy(method_args_dict)
        if connection == "None":
            pass
        elif connection == "discrete_to_num":
            temp_args_dict["add_connection_discrete_to_num"] = True
            temp_args_dict["dim_hidden_layer_discrete_to_num"] = dim_hidden_connection
        elif connection == "num_to_discrete":
            temp_args_dict["add_connection_num_to_discrete"] = True
            temp_args_dict["dim_hidden_layer_num_to_discrete"] = dim_hidden_connection
        else:
            raise ValueError("For the hyperparameter tuning add_connection_advanced then the connection parameter"
                             " only takes as input 'None', 'discrete_to_num' or 'num_to_discrete'.")
        tg_qtr = TabGAN(data_train, **temp_args_dict)
        return tg_qtr

    hp_info["add_connection_advanced"] = {
        "vec": add_connection_advanced_vec,
        "n_synthetic_datasets": 10,
        "n_epochs": N_EPOCHS,
        "tabGAN_func": create_tabGAN_for_add_connection_advanced,
        "batch_size": BATCH_SIZE,
        "hyperparams_subname": ["dim_hidden_connection", "connection"]
    }

    activation_function_vec = [("GELU", False), ("GELU", True)]
    activation_function_vec += [(function, False) for function in ["ReLU", "LeakyReLU", "SquaredReLU", "ELU", "Swish",
                                                                   "SELU", "LeakySquaredReLU"]]

    def create_tabGAN_for_activation_function(activation_function, approximate):
        temp_args_dict = copy.deepcopy(method_args_dict)
        temp_args_dict["activation_function"] = activation_function
        if activation_function.lower() == "gelu" and approximate:
            temp_args_dict["gelu_approximate"] = True
        tg_qtr = TabGAN(data_train, **temp_args_dict)
        return tg_qtr

    hp_info["activation_function"] = {
        "vec": activation_function_vec,
        "n_synthetic_datasets": 25,
        "n_epochs": N_EPOCHS,
        "tabGAN_func": create_tabGAN_for_activation_function,
        "batch_size": BATCH_SIZE,
        "hyperparams_subname": ["function", "approximate"]
    }

    wgan_penalty_query_vec = ["Same_queries_generator_and_critic", "Different_queries_generator_and_critic",
                              "Same_queries_generator_and_critic_but_diverse_penalty",
                              "Same_queries_generator_and_critic_but_no_query_input_to_critic",
                              "Different_queries_generator_and_critic_but_no_query_input_to_critic",
                              "Same_queries_generator_and_critic_but_no_queries_wgan_penalty",
                              "Different_queries_generator_and_critic_but_no_queries_wgan_penalty"]

    def create_tabGAN_for_wgan_penalty_query(argument):
        temp_args_dict = copy.deepcopy(method_args_dict)
        query_input_to_critic = True
        train_step_critic_wgan_penalty_query_diversity = False
        train_step_critic_query_wgan_penalty = True
        if argument == "Same_queries_generator_and_critic":
            train_step_critic_same_queries_for_critic_and_gen = True
        elif argument == "Different_queries_generator_and_critic":
            train_step_critic_same_queries_for_critic_and_gen = False
        elif argument == "Same_queries_generator_and_critic_but_diverse_penalty":
            train_step_critic_same_queries_for_critic_and_gen = True
            train_step_critic_wgan_penalty_query_diversity = True
        elif argument == "Same_queries_generator_and_critic_but_no_query_input_to_critic":
            train_step_critic_same_queries_for_critic_and_gen = True
            query_input_to_critic = False
        elif argument == "Different_queries_generator_and_critic_but_no_query_input_to_critic":
            train_step_critic_same_queries_for_critic_and_gen = False
            query_input_to_critic = False
        elif argument == "Same_queries_generator_and_critic_but_no_queries_wgan_penalty":
            train_step_critic_same_queries_for_critic_and_gen = True
            train_step_critic_query_wgan_penalty = False
        elif argument == "Different_queries_generator_and_critic_but_no_queries_wgan_penalty":
            train_step_critic_same_queries_for_critic_and_gen = False
            train_step_critic_query_wgan_penalty = False
        else:
            raise ValueError("Argument must be one of the following: 'Same_queries_generator_and_critic', "
                             "'Different_queries_generator_and_critic', or" 
                             "'Same_queries_generator_and_critic_but_diverse_penalty'")
        temp_args_dict["train_step_critic_same_queries_for_critic_and_gen"] = train_step_critic_same_queries_for_critic_and_gen
        temp_args_dict["train_step_critic_wgan_penalty_query_diversity"] = train_step_critic_wgan_penalty_query_diversity
        temp_args_dict["critic_use_query_input"] = query_input_to_critic
        temp_args_dict["train_step_critic_query_wgan_penalty"] = train_step_critic_query_wgan_penalty
        tg_qtr = TabGAN(data_train, **temp_args_dict)
        return tg_qtr

    hp_info["wgan_penalty_query"] = {
        "vec": wgan_penalty_query_vec,
        "n_synthetic_datasets": 25,
        "n_epochs": N_EPOCHS,
        "tabGAN_func": create_tabGAN_for_wgan_penalty_query,
        "batch_size": BATCH_SIZE,
        "hyperparams_subname": None
    }
    return hp_info

