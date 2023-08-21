from enum import Enum

import numpy as np
import pandas as pd


import tensorflow as tf
from tensorflow import one_hot
from tensorflow.python.keras.layers import Concatenate


class Mode(Enum):
    PRL2 = 1
    PRL4 = 2
    PRL2_intractable = 3
    HRL2 = 4
    HRL2_Bayes = 6
    HRL2_StickyBayes = 7
    Bayes = 8
    StickyBayes = 9


LIST_HRL_MODELS = [Mode.HRL2, Mode.HRL2_Bayes, Mode.HRL2_StickyBayes]


def get_features(data, n_agent, n_trial, n_action=2, mode=Mode.PRL2):
    """Extract the features from reward and action sequences.

    Args:
        data: A matrix containing the feature representation computed with CEBRA.
        n_agent: The number of agents in the data.
        n_trial: The number of trials.
        n_action: The number of available actions.
        mode: The simulation task that data is generated from.

    Returns:
        A tensor object with shape (num_agent, n_trial, num_feature).
    """
    if mode in LIST_HRL_MODELS:
        stim_prefix = "allstims"
        side0 = (
            data[f"{stim_prefix}0"]
            .to_numpy()
            .astype(np.float32)
            .reshape((n_agent, n_trial))
        )
        side1 = (
            data[f"{stim_prefix}1"]
            .to_numpy()
            .astype(np.float32)
            .reshape((n_agent, n_trial))
        )
        side2 = (
            data[f"{stim_prefix}2"]
            .to_numpy()
            .astype(np.float32)
            .reshape((n_agent, n_trial))
        )
        action = (
            data["chosenside"].to_numpy().astype(np.int32).reshape((n_agent, n_trial))
        )
        reward = (
            data["rewards"].to_numpy().astype(np.float32).reshape((n_agent, n_trial))
        )
        # turn action into one-hot
        action_onehot = one_hot(action, n_action)
        # concatenate reward with action
        return Concatenate(axis=2)(
            [
                side0[:, :, np.newaxis],
                side1[:, :, np.newaxis],
                side2[:, :, np.newaxis],
                reward[:, :, np.newaxis],
                action_onehot,
            ]
        )

    # Probablistic RL
    action = data["actions"].to_numpy().astype(np.int32).reshape((n_agent, n_trial))
    reward = data["rewards"].to_numpy().astype(np.float32).reshape((n_agent, n_trial))

    # turn action into one-hot
    action_onehot = one_hot(action, n_action)
    # concatenate reward with action
    return Concatenate(axis=2)([reward[:, :, np.newaxis], action_onehot])


def get_label_names_by_mode(mode):
    """Retrieve label names given simulation task.

    Args:
        mode: The simulation task that data is generated from.

    Returns:
        A list of latent variables of the simulation task.
    """
    if mode == Mode.PRL2 or mode == Mode.HRL2:
        return ["alpha", "beta"]
    elif mode == Mode.PRL2_intractable:
        return ["alpha", "beta", "T"]
    elif mode == Mode.Bayes:
        return ["beta", "preward", "pswitch"]
    elif mode == Mode.StickyBayes:
        return ["beta", "preward", "pswitch", "stickiness"]
    elif mode == Mode.PRL4:
        return ["alpha", "beta", "neg_alpha", "stickiness"]
    else:
        raise Exception("The task mode is invalid")


def get_labels(data, mode=Mode.PRL2):
    """Retrieve label from the generated data given simulation task.

    Args:
        mode: The simulation task that data is generated from.

    Returns:
        A dict contains latent parameter names to its values.
    """
    name_to_labels = {}
    for l in get_label_names_by_mode(mode):
        name_to_labels[l] = data.groupby("agentid")[l].agg(["mean"]).to_numpy()

    return name_to_labels


def normalize_train_labels(name_to_labels: dict):
    """Standardize labels by removing the mean and scaling to unit variance

    Args:
        name_to_labels: A dict contains latent parameter names to its values.

    Returns:
        A tuple with the first element being an array of
        normalized labels with shape (num_agent, num_latent_variables).
        The second element is a dict containing latent parameter name to its fitted scaler
    """
    from sklearn.preprocessing import StandardScaler

    names = list(name_to_labels.keys())
    names.sort()

    normalized_labels = []
    name_to_scaler = {}
    for name in names:
        scaler = StandardScaler()
        normalized_labels.append(scaler.fit_transform(name_to_labels[name]))
        name_to_scaler[name] = scaler

    return np.concatenate(normalized_labels, axis=-1), name_to_scaler


def normalize_val_labels(name_to_labels: dict, name_to_scaler: dict):
    """Normalize the validation labels given training labels scaler.

    Args:
        name_to_labels: A dict contains latent parameter names to its values.
        name_to_scaler: A dict containing latent parameter name to its fitted scaler.

    Returns:
        A dict contains latent parameter names to its values.
    """
    names = list(name_to_labels.keys())
    names.sort()

    normalized_labels = []
    for name in names:
        scaler = name_to_scaler[name]
        normalized_labels.append(scaler.transform(name_to_labels[name]))

    return np.concatenate(normalized_labels, axis=-1)


def padding(data, max_num_trial: int, num_trials: int):
    """Padding the features with -1 based on maximum number of trial.

    Args:
        data: A tensor object with shape (num_agent, num_trials, num_feature).
        max_num_trial: The maximum number of trial.
        num_trials: The number of trial in the data.

    Returns:
        A padded tensor object with shape (num_agent, max_num_trial, num_feature)
        with features after num_trials as -1.
    """
    paddings = tf.constant([[0, max_num_trial - num_trials], [0, 0]])
    a_list = tf.unstack(data)
    for j in range(len(data)):
        padded_inputs = tf.pad(data[j], paddings, "CONSTANT", constant_values=-1)
        a_list[j] = padded_inputs

    return tf.stack(a_list)


def get_mixed_trials_features(all_trials_features: list, list_trials: list):
    """Generate features with different trials.

    Args:
        all_trials_features: A tensor object with shape (num_agent, num_trial, num_feature).
        list_trials: A list of number of trials to be generated. e.g. [100, 200, 300, 400, 500]

    Returns:
        A tensor object with shape (num_agent, num_trial, num_feature) with each number of trials generated equally. 
        For example, if there are five items in list_trials and 30k agents. 
        Each trial data will have 6k agents in the output.
    """
    num_agents, max_num_trial = (
        all_trials_features.shape[0],
        all_trials_features.shape[1],
    )
    if len(list_trials) == 1 and list_trials[0] == max_num_trial:
        return all_trials_features

    features = []
    start = int(num_agents / len(list_trials))
    for i, num_trials in zip(range(start, num_agents + 1, start), list_trials):
        partial_features = all_trials_features[i - start : i, :num_trials, :]
        padded_inputs = padding(partial_features, max_num_trial, num_trials)
        features.append(padded_inputs)

    return tf.concat(features, 0)


def recover_parameter(prediction, scaler):
    """Scale back the prediction to the original representation.

    Args:
        prediction: The standardized prediction from neural network
        scaler: A fitted scaler

    Returns:
        A original values based on training data fitted scaler.
    """  
    estimated = prediction.reshape(prediction.shape[0], 1)
    return scaler.inverse_transform(estimated)[:, 0]

def get_recovered_parameters(name_to_scaler, name_to_true_parms, prediction):
    """Scale parameters back to its original range based on training fitted scaler.

    Args:
        name_to_scaler: A dict with key as parameter name and value as its fitted scaler.
        name_to_true_parms: A dict with key as parameter name and value as true parameter values.
        prediction: The standardized prediction from neural network

    Returns:
        A dict contains latent parameter names to its values.
    """
    from collections import defaultdict

    sorted_label_names = list(name_to_true_parms.keys())
    sorted_label_names.sort()
    param_all_test = defaultdict(list)
    idx = 0
    for l in sorted_label_names:
        k = f'true_{l}'
        param_all_test[k] = name_to_true_parms[l][:, 0]

        k = f'dl_{l}'
        param_all_test[k] = recover_parameter(prediction[:, idx], name_to_scaler[l])
        idx += 1

    return pd.DataFrame(param_all_test)