import os
import helpers.hp_tuning.hp_gen
import helpers.hp_tuning.hp_gen
from tabGAN import TabGAN
from src import constants as const
import numpy as np
import pandas as pd
import copy

JIT_COMPILE_TRAIN_STEP = True
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
    "pac": 1,
    # WGAN Query penalty and critic's query usage
    "train_step_critic_same_queries_for_critic_and_gen": True,
    "train_step_critic_wgan_penalty_query_diversity": False,
    "train_step_critic_query_wgan_penalty": True,
    "critic_use_query_input": True
}

dataset_train_path = os.path.join(const.dir.data(), "df_adult_edited_train.csv")
data_train = pd.read_csv(dataset_train_path)

def fetch_hp_info(method="ctabGAN-qtr", version=2):
    if method == "tabGAN-qtr":
        method_args_dict = tabgan_args_dict
    elif method == "ctabGAN-qtr":
        method_args_dict = ctabgan_args_dict
    else:
        raise ValueError(f"Entered method name {method} has not yet any hyperparameter tuning info")

    if version >= 3:
        N_EPOCHS = 300
        method_args_dict["activation_function"] = "GELU"
        method_args_dict["gelu_approximate"] = False
        if method == "ctabGAN-qtr":
            method_args_dict["train_step_critic_same_queries_for_critic_and_gen"] = False,
    else:
        N_EPOCHS = 100

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

    add_connection_advanced_vec = [(0, "None", "None")]
    add_connection_advanced_vec += [(dim_hidden_connection, connection, activation_function)
                                    for connection in ["discrete_to_num", "num_to_discrete"]
                                    for dim_hidden_connection in [0, 1, 5, 10, 25, 50, 100, 200]
                                    for activation_function in ["None", "LeakyReLU", "GELU"]]
    if method == "ctabGAN-qtr":
        add_connection_advanced_vec += [(0, "query_to_discrete", "None")]

    def create_tabGAN_for_add_connection_advanced(dim_hidden_connection, connection,
                                                  add_connection_activation_function):
        temp_args_dict = copy.deepcopy(method_args_dict)
        if add_connection_activation_function == "None":
            add_connection_activation_function = None
        temp_args_dict["add_connection_activation_function"] = add_connection_activation_function
        if connection == "None":
            pass
        elif connection == "discrete_to_num":
            temp_args_dict["add_connection_discrete_to_num"] = True
            temp_args_dict["dim_hidden_layer_discrete_to_num"] = dim_hidden_connection
        elif connection == "num_to_discrete":
            temp_args_dict["add_connection_num_to_discrete"] = True
            temp_args_dict["dim_hidden_layer_num_to_discrete"] = dim_hidden_connection
        elif connection == "query_to_discrete":
            temp_args_dict_dict["add_connection_query_to_discrete"] = True
        else:
            raise ValueError("For the hyperparameter tuning add_connection_advanced then the connection parameter"
                             " only takes as input 'None', 'discrete_to_num' or 'num_to_discrete' or" 
                             " 'query_to_discrete'")
        tg_qtr = TabGAN(data_train, **temp_args_dict)
        return tg_qtr

    hp_info["add_connection_advanced"] = {
        "vec": add_connection_advanced_vec,
        "n_synthetic_datasets": 10,
        "n_epochs": N_EPOCHS,
        "tabGAN_func": create_tabGAN_for_add_connection_advanced,
        "batch_size": BATCH_SIZE,
        "hyperparams_subname": ["dim_hidden_connection", "connection", "connection_activation_function"]
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

    # wgan_penalty_query_vec = ["Same_queries_generator_and_critic", "Different_queries_generator_and_critic",
    #                           "Same_queries_generator_and_critic_but_diverse_penalty",
    #                           "Same_queries_generator_and_critic_but_no_query_input_to_critic",
    #                           "Same_queries_generator_and_critic_but_no_queries_wgan_penalty",
    #                           "Different_queries_generator_and_critic_but_no_queries_wgan_penalty"]
    wgan_penalty_query_vec = [(same_queries_generator_and_critic, other_argument)
                              for same_queries_generator_and_critic in [False, True]
                              for other_argument in ["None", "No wgan query penalty"]]
    wgan_penalty_query_vec += [(True, "Diverse query WGAN penalty"), (True, "No query input to critic")]

    def create_tabGAN_for_wgan_penalty_query(same_queries_generator_and_critic, other_argument):
        temp_args_dict = copy.deepcopy(method_args_dict)
        query_input_to_critic = True
        train_step_critic_wgan_penalty_query_diversity = False
        train_step_critic_query_wgan_penalty = True
        if other_argument == "None":
            pass
        elif other_argument == "Diverse query WGAN penalty":
            train_step_critic_wgan_penalty_query_diversity = True
        elif other_argument == "No query input to critic":
            query_input_to_critic = False
        elif other_argument == "No wgan query penalty":
            train_step_critic_query_wgan_penalty = False
        else:
            raise ValueError("other_argument parameter must be one of the following: 'None', 'Diverse query WGAN penalty', "
                             "'No query input to critic', or" 
                             "'No wgan query penalty'")
        temp_args_dict["train_step_critic_same_queries_for_critic_and_gen"] = same_queries_generator_and_critic
        temp_args_dict["train_step_critic_wgan_penalty_query_diversity"] = train_step_critic_wgan_penalty_query_diversity
        temp_args_dict["critic_use_query_input"] = query_input_to_critic
        temp_args_dict["train_step_critic_query_wgan_penalty"] = train_step_critic_query_wgan_penalty
        temp_args_dict["jit_compile"] = False
        tg_qtr = TabGAN(data_train, **temp_args_dict)
        return tg_qtr

    hp_info["wgan_penalty_query"] = {
        "vec": wgan_penalty_query_vec,
        "n_synthetic_datasets": 25,
        "n_epochs": N_EPOCHS,
        "tabGAN_func": create_tabGAN_for_wgan_penalty_query,
        "batch_size": BATCH_SIZE,
        "hyperparams_subname": ["Same_queries_generator_and_critic", "other_argument"]
    }

    qt_distribution_vec = ["normal", "uniform"]

    def create_tabGAN_for_qt_distribution(qt_distribution):
        temp_args_dict = copy.deepcopy(method_args_dict)
        temp_args_dict["qt_distribution"] = qt_distribution
        tg_qtr = TabGAN(data_train, **temp_args_dict)
        return tg_qtr

    hp_info["qt_distribution"] = {
        "vec": qt_distribution_vec,
        "n_synthetic_datasets": 10,
        "n_epochs": N_EPOCHS,
        "tabGAN_func": create_tabGAN_for_qt_distribution,
        "batch_size": BATCH_SIZE,
        "hyperparams_subname": None
    }

    qt_transformation_vec = [(qtr_spread, qt_distribution, latent_distribution)
                             for qtr_spread in [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]
                          for qt_distribution in ["normal", "uniform"]
                          for latent_distribution in ["normal", "uniform"]]

    def create_tabGAN_for_qt_transformation(qtr_spread, qt_distribution, latent_distribution):
        temp_args_dict = copy.deepcopy(method_args_dict)
        temp_args_dict["qtr_spread"] = qtr_spread
        temp_args_dict["qt_distribution"] = qt_distribution
        temp_args_dict["latent_distribution"] = latent_distribution
        tg_qtr = TabGAN(data_train, **temp_args_dict)
        return tg_qtr

    hp_info["qt_transformation"] = {
        "vec": qt_transformation_vec,
        "n_synthetic_datasets": 10,
        "n_epochs": N_EPOCHS,
        "tabGAN_func": create_tabGAN_for_qt_transformation,
        "batch_size": BATCH_SIZE,
        "hyperparams_subname": ["qtr_spread", "qt_distribution", "latent_distribution"]
    }

    oh_encoding_activation_function_vec = ["softmax", "gumbel"]

    def create_tabGAN_for_oh_encoding_activation_function(oh_encoding_activation_function):
        temp_args_dict = copy.deepcopy(method_args_dict)
        temp_args_dict["oh_encoding_activation_function"] = oh_encoding_activation_function
        tg_qtr = TabGAN(data_train, **temp_args_dict)
        return tg_qtr

    hp_info["oh_encoding_activation_function"] = {
        "vec": oh_encoding_activation_function_vec,
        "n_synthetic_datasets": 10,
        "n_epochs": N_EPOCHS,
        "tabGAN_func": create_tabGAN_for_oh_encoding_activation_function,
        "batch_size": BATCH_SIZE,
        "hyperparams_subname": None
    }

    reapply_qtr_continuously_vec = [False, True]
    def create_tabGAN_for_reapply_qtr_continuously(reapply_qtr_continuously):
        temp_args_dict = copy.deepcopy(method_args_dict)
        temp_args_dict["reapply_qtr_continuously"] = reapply_qtr_continuously
        tg_qtr = TabGAN(data_train, **temp_args_dict)
        return tg_qtr

    hp_info["reapply_qtr_continuously"] = {
        "vec": reapply_qtr_continuously_vec,
        "n_synthetic_datasets": 25,
        "n_epochs": N_EPOCHS,
        "tabGAN_func": create_tabGAN_for_reapply_qtr_continuously,
        "batch_size": BATCH_SIZE,
        "hyperparams_subname": None
    }

    return hp_info

