from kerashypetune import KerasBayesianSearch
from hyperopt import hp, Trials
from keras.callbacks import EarlyStopping


def bayesian_search(
    model_func, param_grid, train_features, train_labels, val_features, val_labels
):
    """Bayesian hyperparamater searching and optimization on a fixed validation set.
    The function uses this open source wrapper https://github.com/cerlymarco/keras-hypetune

    Args:
        model_func: A callable that takes parameters in dict format and returns a TF Model instance.

        param_grid: Hyperparameters to try, 1-to-1 mapped with the parameters dict keys
            present in the hypermodel function.

        train_features:  Input data.
        train_labels:  Targeted label.
        val_features, val_labels: Data on which evaluate the loss and any model metrics at the end of
            each epoch. All the validation_data formats supported by Keras model are accepted.

    Returns:
        A tuple that contains the best results from search: (best_model, best_params)

        best_model : TF Model. The best model (in term of score).
        best_params : dict. The dict containing the best combination (in term of score) of hyperparameters.
    """
    kbs = KerasBayesianSearch(
        model_func,
        param_grid,
        monitor="val_loss",
        greater_is_better=False,
        n_iter=10,
        sampling_seed=33,
        store_model=True,
    )
    kbs.search(
        train_features,
        train_labels,
        trials=Trials(),
        validation_data=(val_features, val_labels),
        callbacks=[EarlyStopping(monitor="val_loss", patience=20)],
    )
    print(kbs.best_params)
    print(kbs.scores)

    return kbs.best_model, kbs.best_params
