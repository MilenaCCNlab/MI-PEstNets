from kerashypetune import KerasBayesianSearch
from hyperopt import hp, Trials
from keras.callbacks import EarlyStopping


def bayesian_search(
    model_func, param_grid, train_features, train_labels, val_features, val_labels
):
    kbs = KerasBayesianSearch(
        model_func,
        param_grid,
        monitor="val_loss",
        greater_is_better=False,
        n_iter=10,
        sampling_seed=33,
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
