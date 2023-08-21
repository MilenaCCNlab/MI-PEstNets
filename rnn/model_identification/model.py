from tensorflow.keras.layers import (
    Dense,
    Dropout,
    GRU,
)

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.optimizers import Adam


def create_gru_model(
    input_x: int,
    input_y: int,
    output_dim: int,
    units: int = 70,
    dropout: float = 0.2,
    dropout1: float = 0.2,
    dropout2: float = 0.1,
    learning_rate: float = 1e-3,
):
    activation_func = "relu"

    model = keras.Sequential(
        [
            tf.keras.layers.Masking(mask_value=-1.0, input_shape=(input_x, input_y)),
            GRU(units, return_sequences=False),
        ]
    )
    model.add(Dropout(dropout))
    model.add(Dense(int(units / 2), activation=activation_func))
    model.add(Dropout(dropout1))
    model.add(Dense(int(units / 4), activation=activation_func))
    model.add(Dropout(dropout2))
    model.add(Dense(output_dim, activation="softmax"))

    optimizer = Adam(learning_rate=learning_rate)
    model.compile(loss="categorical_crossentropy", optimizer=optimizer, metrics=["acc"])
    return model


def get_gru_model(parms):
    return create_gru_model(
        parms["input_x"],
        parms["input_y"],
        parms["output_dim"],
        parms["units"],
        parms["dropout"],
        parms["dropout1"],
        parms["dropout2"],
        parms["learning_rate"],
    )
