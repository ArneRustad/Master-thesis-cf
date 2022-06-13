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

    if version >= 4:
        method_args_dict["activation_function"] = "Mish"
        method_args_dict["qtr_spread"] = 1
        method_args_dict["adam_beta1"] = 0.9
    if version >= 5:
        method_args_dict["adam_beta1"] = 0.7
    if version >= 6:
        method_args_dict["activation_function"] = "GELU"
        method_args_dict["gelu_approximate"] = False
    if version >= 7:
        method_args_dict["batch_normalization_generator"] = True
        method_args_dict["activation_function"] = "Mish"
    if version >= 8:
        method_args_dict["activation_function"] = "GELU"
        method_args_dict["gelu_approximate"] = False

    hp_info = {}

    def create_tabGAN_for_qtr_spread(qtr_spread):
        temp_args_dict = copy.deepcopy(method_args_dict)
        temp_args_dict["qtr_spread"] = qtr_spread
        tg_qtr = TabGAN(data_train, **temp_args_dict)
        return tg_qtr

    hp_info["qtr_spread"] = {
        "vec": np.round(np.linspace(0, 1, 21), 2),
        "n_synthetic_datasets": 10 if version >= 6 else 25,
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
    add_connection_activation_function_vec = ["None", "LeakyReLU", "GELU"]
    if version >= 4:
        add_connection_activation_function_vec += "Mish"
    add_connection_advanced_vec += [(dim_hidden_connection, connection, activation_function)
                                    for connection in ["discrete_to_num", "num_to_discrete"]
                                    for dim_hidden_connection in [0, 1, 5, 10, 25, 50, 100, 200]
                                    for activation_function in add_connection_activation_function_vec]
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
                                                                   "SELU", "LeakySquaredReLU", "Mish"]]

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

    oh_encoding_activation_function_partial_vec = ["softmax", "gumbel"]
    oh_encoding_temperature_partial_vec = np.round(np.arange(0.1, 1.01, 0.1),3).tolist()
    oh_encoding_vec = [(temperature, activation_function)
                       for temperature in oh_encoding_temperature_partial_vec
                       for activation_function in oh_encoding_activation_function_partial_vec]

    def create_tabGAN_for_oh_encoding(temperature, oh_encoding_activation_function):
        temp_args_dict = copy.deepcopy(method_args_dict)
        temp_args_dict["oh_encoding_activation_function"] = oh_encoding_activation_function
        if oh_encoding_activation_function.lower() == "gumbel":
            temp_args_dict["gumbel_temperature"] = temperature
        elif oh_encoding_activation_function.lower() == "softmax":
            temp_args_dict["softmax_temperature"] = temperature
        else:
            raise ValueError("Only softmax and gumbel are valid oh_encoding_activation_functions")
        tg_qtr = TabGAN(data_train, **temp_args_dict)
        return tg_qtr

    hp_info["oh_encoding"] = {
        "vec": oh_encoding_vec,
        "n_synthetic_datasets": 10,
        "n_epochs": N_EPOCHS,
        "tabGAN_func": create_tabGAN_for_oh_encoding,
        "batch_size": BATCH_SIZE,
        "hyperparams_subname": ["temperature", "activation_function"]
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

    spread_and_activation_vec = [(qtr_spread, activation_function, reapply_qtr_continuously)
                                 for activation_function in ["GELU", "GELU_approx", "ReLU", "LeakyReLU", "SquaredReLU",
                                                             "ELU", "Swish", "SELU", "LeakySquaredReLU", "Mish"]
                                 for qtr_spread in np.round(np.arange(0, 1.01, 0.1), 3).tolist()
                                 for reapply_qtr_continuously in [False]
                                 ]

    def create_tabGAN_for_spread_and_activation(qtr_spread, activation_function, reapply_qtr_continuously):
        temp_args_dict = copy.deepcopy(method_args_dict)
        if activation_function.lower() == "gelu_approx":
            temp_args_dict["gelu_approximate"] = True
            activation_function = "gelu"
        temp_args_dict["activation_function"] = activation_function
        temp_args_dict["qtr_spread"] = qtr_spread
        temp_args_dict["reapply_qtr_continuously"] = reapply_qtr_continuously

        tg_qtr = TabGAN(data_train, **temp_args_dict)
        return tg_qtr

    hp_info["spread_and_activation"] = {
        "vec": spread_and_activation_vec,
        "n_synthetic_datasets": 10,
        "n_epochs": N_EPOCHS,
        "tabGAN_func": create_tabGAN_for_spread_and_activation,
        "batch_size": BATCH_SIZE,
        "hyperparams_subname": ["qtr_spread", "activation_function", "reapply_qtr_continuously"]
    }

    spread_and_activations_vec = [(qtr_spread, (activation_function_generator, activation_function_critic),
                                   reapply_qtr_continuously)
                                  for activation_function_generator in ["GELU", "ReLU", "LeakyReLU", "Mish"]
                                  for activation_function_critic in ["GELU", "ReLU", "LeakyReLU", "Mish"]
                                  for qtr_spread in [0, 0.1, 0.8, 1] #np.round(np.arange(0, 1.01, 0.1), 3).tolist()
                                  for reapply_qtr_continuously in [False, True]
                                  ]

    def create_tabGAN_for_spread_and_activations(qtr_spread, activation_functions, reapply_qtr_continuously):
        temp_args_dict = copy.deepcopy(method_args_dict)
        temp_args_dict["activation_function_generator"] = activation_functions[0]
        temp_args_dict["activation_function_critic"] = activation_functions[1]
        temp_args_dict["qtr_spread"] = qtr_spread
        temp_args_dict["reapply_qtr_continuously"] = reapply_qtr_continuously

        tg_qtr = TabGAN(data_train, **temp_args_dict)
        return tg_qtr

    hp_info["spread_and_activations"] = {
        "vec": spread_and_activations_vec,
        "n_synthetic_datasets": 10,
        "n_epochs": N_EPOCHS,
        "tabGAN_func": create_tabGAN_for_spread_and_activations,
        "batch_size": BATCH_SIZE,
        "hyperparams_subname": ["qtr_spread", "activation_function_generator_and_critic", "reapply_qtr_continuously"]
    }

    qtr_vec = [(qtr_spread, qtr_uniform_on_normal_scale, reapply_qtr_continuously)
               for qtr_uniform_on_normal_scale in [False, True]
               for qtr_spread in np.round(np.arange(0, 1.01, 0.1), 2)
               for reapply_qtr_continuously in [False]]

    def create_tabGAN_for_qtr(qtr_spread, qtr_uniform_on_normal_scale, reapply_qtr_continuously):
        temp_args_dict = copy.deepcopy(method_args_dict)
        temp_args_dict["qtr_spread"] = qtr_spread
        temp_args_dict["qtr_uniform_on_normal_scale"] = qtr_uniform_on_normal_scale
        temp_args_dict["reapply_qtr_continuously"] = reapply_qtr_continuously
        tg_qtr = TabGAN(data_train, **temp_args_dict)
        return tg_qtr

    hp_info["qtr"] = {
        "vec": qtr_vec,
        "n_synthetic_datasets": 10,
        "n_epochs": N_EPOCHS,
        "tabGAN_func": create_tabGAN_for_qtr,
        "batch_size": BATCH_SIZE,
        "hyperparams_subname": ["qtr_spread", "qtr_uniform_on_normal_scale", "reapply_qtr_continuously"]
    }

    qtr_distribution_beta_param_vec = np.round(np.arange(1, 5), 3).tolist()

    def create_tabGAN_for_qtr_distribution_beta_param(beta_param):
        temp_args_dict = copy.deepcopy(method_args_dict)
        temp_args_dict["qtr_spread"] = 1
        temp_args_dict["reapply_qtr_continuously"] = False
        temp_args_dict["qtr_distribution"] = "beta"
        temp_args_dict["qtr_beta_distribution_parameter"] = beta_param
        tg_qtr = TabGAN(data_train, **temp_args_dict)
        return tg_qtr

    hp_info["qtr_distribution_beta_param"] = {
        "vec": qtr_distribution_beta_param_vec,
        "n_synthetic_datasets": 10,
        "n_epochs": N_EPOCHS,
        "tabGAN_func": create_tabGAN_for_qtr_distribution_beta_param,
        "batch_size": BATCH_SIZE,
        "hyperparams_subname": None
    }

    adam_beta1_vec = np.round(np.arange(0.05, 1, 0.05), 3).tolist() + [0.01, 0.02, 0.03, 0.04] + [0.00001]

    def create_tabGAN_for_adam_beta1(adam_beta1):
        temp_args_dict = copy.deepcopy(method_args_dict)
        temp_args_dict["adam_beta1"] = adam_beta1
        tg_qtr = TabGAN(data_train, **temp_args_dict)
        return tg_qtr

    hp_info["adam_beta1"] = {
        "vec": adam_beta1_vec,
        "n_synthetic_datasets": 10,
        "n_epochs": N_EPOCHS,
        "tabGAN_func": create_tabGAN_for_adam_beta1,
        "batch_size": BATCH_SIZE,
        "hyperparams_subname": None
    }

    noise_discrete_unif_max_vec = np.round(np.arange(0, 0.21, 0.01), 3).tolist() + [0.002, 0.004, 0.006, 0.008]

    def create_tabGAN_for_noise_discrete_unif_max(noise_discrete_unif_max):
        temp_args_dict = copy.deepcopy(method_args_dict)
        temp_args_dict["noise_discrete_unif_max"] = noise_discrete_unif_max
        tg_qtr = TabGAN(data_train, **temp_args_dict)
        return tg_qtr

    hp_info["noise_discrete_unif_max"] = {
        "vec": noise_discrete_unif_max_vec,
        "n_synthetic_datasets": 10,
        "n_epochs": N_EPOCHS,
        "tabGAN_func": create_tabGAN_for_noise_discrete_unif_max,
        "batch_size": BATCH_SIZE,
        "hyperparams_subname": None
    }

    best_activation_function_vec = ["Mish", "GELU"]
    def create_tabGAN_for_best_activation_function(activation_function):
        temp_args_dict = copy.deepcopy(method_args_dict)
        temp_args_dict["activation_function"] = activation_function
        temp_args_dict["gelu_approximate"] = False
        tg_qtr = TabGAN(data_train, **temp_args_dict)
        return tg_qtr

    hp_info["best_activation_function"] = {
        "vec": best_activation_function_vec,
        "n_synthetic_datasets": 25,
        "n_epochs": N_EPOCHS,
        "tabGAN_func": create_tabGAN_for_best_activation_function,
        "batch_size": BATCH_SIZE,
        "hyperparams_subname": None
    }

    bn_vec = [(False, False), (True, False), (True, True)]
    def create_tabGAN_for_bn(batch_normalization_generator,
                                                   concatenate_with_previous_layer):
        temp_args_dict = copy.deepcopy(method_args_dict)
        temp_args_dict["batch_normalization_generator"] = batch_normalization_generator
        temp_args_dict["generator_concatenate_hidden_with_previous_layer"] = concatenate_with_previous_layer
        tg_qtr = TabGAN(data_train, **temp_args_dict)
        return tg_qtr

    hp_info["BN"] = {
        "vec": bn_vec,
        "n_synthetic_datasets": 25 if version >= 7 else 10,
        "n_epochs": N_EPOCHS,
        "tabGAN_func": create_tabGAN_for_bn,
        "batch_size": BATCH_SIZE,
        "hyperparams_subname": ["batch_normalization_generator", "concatenate_with_previous_layer"]
    }

    bn_advanced_vec = [(False, False, False), (True, False, False)]
    bn_advanced_vec += [(concatenate_with_previous, True, bn_before_activation)
                       for concatenate_with_previous in [False, True]
                       for bn_before_activation in [False, True]]
    def create_tabGAN_for_bn_advanced(concatenate_with_previous_layer, batch_normalization_generator,
                                      batch_normalization_before_activation):
        temp_args_dict = copy.deepcopy(method_args_dict)
        temp_args_dict["generator_concatenate_hidden_with_previous_layer"] = concatenate_with_previous_layer
        temp_args_dict["batch_normalization_generator"] = batch_normalization_generator
        temp_args_dict["batch_normalization_before_activation"] = batch_normalization_before_activation
        tg_qtr = TabGAN(data_train, **temp_args_dict)
        return tg_qtr

    hp_info["BN_advanced"] = {
        "vec": bn_advanced_vec,
        "n_synthetic_datasets": 10,
        "n_epochs": N_EPOCHS,
        "tabGAN_func": create_tabGAN_for_bn_advanced,
        "batch_size": BATCH_SIZE,
        "hyperparams_subname": ["concatenate_with_previous_layer", "batch_normalization", "BN_before_activation"]
    }

    ln_advanced_vec = [(False, "None")]
    ln_advanced_vec += [(ln_before_activation, ln_type)
                        for ln_type in ["Standard", "Simple"]
                        for ln_before_activation in [False, True]]
    def create_tabGAN_for_ln_advanced(ln_before_activation, ln_type):
        temp_args_dict = copy.deepcopy(method_args_dict)
        temp_args_dict["layer_normalization_before_activation"] = ln_before_activation

        if ln_type.lower() == "none":
            temp_args_dict["layer_normalization_critic"] = False
            print("yay")
        elif ln_type.lower() in ["standard", "simple"]:
            temp_args_dict["layer_normalization_critic"] = True
        else:
            raise ValueError(f"Wrong input to ln_type: {ln_type}")

        if ln_type.lower() == "simple":
            print("Simple layer type! Yey")
            temp_args_dict["layer_normalization_simple_type"] = True
        elif ln_type.lower() in ["standard", "none"]:
            temp_args_dict["layer_normalization_simple_type"] = False
        else:
            raise ValueError(f"Wrong input to ln_type: {ln_type}")

        tg_qtr = TabGAN(data_train, **temp_args_dict)
        return tg_qtr

    hp_info["LN_advanced"] = {
        "vec": ln_advanced_vec,
        "n_synthetic_datasets": 15,
        "n_epochs": N_EPOCHS,
        "tabGAN_func": create_tabGAN_for_ln_advanced,
        "batch_size": BATCH_SIZE,
        "hyperparams_subname": ["LN_before_activation", "LayerNormalization"]
    }

    critic_dropout_vec = [(0, [])]
    critic_dropout_vec += [(rate, layers)
                           for rate in [0.25, 0.5]
                           for layers in [[1], [2], [1, 2]]
                           ]
    def create_tabGAN_for_critic_dropout(dropout_rate, dropout_layers):
        temp_args_dict = copy.deepcopy(method_args_dict)
        temp_args_dict["add_dropout_critic"] = dropout_layers
        temp_args_dict["dropout_rate_critic"] = dropout_rate
        tg_qtr = TabGAN(data_train, **temp_args_dict)
        return tg_qtr

    hp_info["critic_dropout"] = {
        "vec": critic_dropout_vec,
        "n_synthetic_datasets": 10,
        "n_epochs": N_EPOCHS,
        "tabGAN_func": create_tabGAN_for_critic_dropout,
        "batch_size": BATCH_SIZE,
        "hyperparams_subname": ["rate", "layers"]
    }

    return hp_info

