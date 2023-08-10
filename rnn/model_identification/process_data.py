import numpy as np
import tensorflow as tf

from utils.data_utils import (
    padding,
    get_features
)

def concat_by_cols(a, b):
    return np.hstack([a, b]) if a.size else b


def get_model_idx_to_features(
    cognitive_model_to_raw_data: dict, num_agents: int, max_num_trial: int
):
    model_idx_to_features = {}
    idx = 0
    for mode, data in cognitive_model_to_raw_data.items():
        model_idx_to_features[idx] = get_features(
            data, num_agents, max_num_trial, mode=mode
        )
        idx += 1

    return model_idx_to_features


def process_data(
    cognitive_model_to_raw_data: dict, num_agents: int, max_num_trial: int, targeted_trials: int
):
    model_idx_to_features = get_model_idx_to_features(cognitive_model_to_raw_data, num_agents, max_num_trial)

    features = []
    labels = np.array([])
    for idx, all_trials_features in model_idx_to_features.items():
        padded_inputs = padding(
            all_trials_features[:, :targeted_trials], max_num_trial, targeted_trials
        )
        features.append(padded_inputs)
        labels = concat_by_cols(labels, np.repeat(idx, num_agents))

    return tf.concat(features, 0), tf.keras.utils.to_categorical(
        labels, num_classes=len(model_idx_to_features)
    )

def get_mixed_trials_features(
    cognitive_model_to_raw_data: dict, num_agents: int, max_num_trial: int, list_trials: list
):
    model_idx_to_features = get_model_idx_to_features(cognitive_model_to_raw_data, num_agents, max_num_trial)

    features = []
    labels = np.array([])
    for idx, all_trials_features in model_idx_to_features.items():
        labels = concat_by_cols(labels, np.repeat(idx, num_agents))
        start = int(num_agents / len(list_trials))
        for i, num_trials in zip(range(start, num_agents + 1, start), list_trials):
            partial_features = all_trials_features[i - start : i, :num_trials, :]
            padded_inputs = padding(partial_features, max_num_trial, num_trials)
            features.append(padded_inputs)

    return tf.concat(features, 0), tf.keras.utils.to_categorical(
        labels, num_classes=len(model_idx_to_features)
    )