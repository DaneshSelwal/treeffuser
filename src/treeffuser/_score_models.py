"""
Contains different score models to be used to approximate the score of a given SDE.
This version has been modified to support multiple tree-based models as a backend.
"""

import abc
from typing import List, Optional, Any

import numpy as np
from jaxtyping import Float, Int
from sklearn.model_selection import train_test_split

# Local (within library) imports
from treeffuser.sde import DiffusionSDE

###################################################
# Helper functions
###################################################


def _fit_one_tree_model(
    model_name: str,
    X: Float[np.ndarray, "batch x_dim"],
    y: Float[np.ndarray, "batch y_dim"],
    X_val: Optional[Float[np.ndarray, "batch x_dim"]],
    y_val: Optional[Float[np.ndarray, "batch y_dim"]],
    seed: int,
    verbose: int,
    cat_idx: Optional[List[int]] = None,
    n_jobs: int = -1,
    **model_args,
) -> Any:
    """
    A universal wrapper for fitting different tree-based models.

    This function initializes and fits a specified model (e.g., LightGBM, XGBoost,
    RandomForest) with its specific parameters and early stopping logic.
    """
    eval_set = None if X_val is None else [(X_val, y_val)]

    # --- LightGBM ---
    if model_name == 'lightgbm':
        import lightgbm as lgb
        callbacks = None
        early_stopping_rounds = model_args.pop('early_stopping_rounds', None)
        if early_stopping_rounds is not None and eval_set is not None:
            callbacks = [lgb.early_stopping(early_stopping_rounds, verbose=verbose > 0)]

        model = lgb.LGBMRegressor(
            random_state=seed,
            n_jobs=n_jobs,
            verbose=verbose,
            **model_args,
        )
        model.fit(
            X=X,
            y=y,
            eval_set=eval_set,
            callbacks=callbacks,
            categorical_feature=cat_idx if cat_idx is not None else "auto",
        )
        return model

    # --- XGBoost ---
    elif model_name == 'xgboost':
        import xgboost as xgb
        early_stopping_rounds = model_args.pop('early_stopping_rounds', None)
        fit_params = {}
        if early_stopping_rounds is not None and eval_set is not None:
            fit_params['early_stopping_rounds'] = early_stopping_rounds
            fit_params['verbose'] = verbose > 0

        model = xgb.XGBRegressor(
            random_state=seed,
            n_jobs=n_jobs,
            **model_args,
        )
        model.fit(X, y, eval_set=eval_set, **fit_params)
        return model

    # --- Scikit-learn RandomForest ---
    elif model_name == 'random_forest':
        from sklearn.ensemble import RandomForestRegressor
        # RandomForest doesn't use a validation set for early stopping
        model = RandomForestRegressor(
            random_state=seed,
            n_jobs=n_jobs,
            verbose=verbose,
            **model_args,
        )
        model.fit(X, y)
        return model

    else:
        raise ValueError(f"Model '{model_name}' is not supported.")


def _make_training_data(
    X: Float[np.ndarray, "batch x_dim"],
    y: Float[np.ndarray, "batch y_dim"],
    sde: DiffusionSDE,
    n_repeats: int,
    eval_percent: Optional[float],
    cat_idx: Optional[List[int]] = None,
    seed: Optional[int] = None,
):
    """
    Creates the training data for the score model. This functions assumes that
    1.  Score is parametrized as score(y, x, t) = GBT(y, x, t) / std(t)
    2.  The loss that we want to use is
        || std(t) * score(y_perturbed, x, t) - (mean(y, t) - y_perturbed)/std(t) ||^2
        Which corresponds to the standard denoising objective with weights std(t)**2
        This ends up meaning that we optimize
        || GBT(y_perturbed, x, t) - (-z)||^2
        where z is the noise added to y_perturbed.
    """
    EPS = 1e-5  # smallest step we can sample from
    T = sde.T
    if seed is not None:
        np.random.seed(seed)

    X_train, X_test, y_train, y_test = X, None, y, None
    predictors_train, predictors_val = None, None
    predicted_train, predicted_val = None, None

    if eval_percent is not None and eval_percent > 0:
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=eval_percent, random_state=seed
        )
    else: # Use all data for training if eval_percent is 0 or None
        X_train, y_train = X, y


    # TRAINING DATA
    n_train_samples = X_train.shape[0]
    X_train_repeated = np.tile(X_train, (n_repeats, 1))
    y_train_repeated = np.tile(y_train, (n_repeats, 1))
    t_train = np.random.uniform(0, 1, size=(y_train_repeated.shape[0], 1)) * (T - EPS) + EPS
    z_train = np.random.normal(size=y_train_repeated.shape)

    train_mean, train_std = sde.get_mean_std_pt_given_y0(y_train_repeated, t_train)
    perturbed_y_train = train_mean + train_std * z_train
    predictors_train = np.concatenate([perturbed_y_train, X_train_repeated, t_train], axis=1)
    predicted_train = -1.0 * z_train

    # VALIDATION DATA
    if X_test is not None and y_test is not None:
        t_val = np.random.uniform(0, 1, size=(y_test.shape[0], 1)) * (T - EPS) + EPS
        z_val = np.random.normal(size=y_test.shape)

        val_mean, val_std = sde.get_mean_std_pt_given_y0(y_test, t_val)
        perturbed_y_val = val_mean + val_std * z_val
        predictors_val = np.concatenate([perturbed_y_val, X_test, t_val], axis=1)
        predicted_val = -1.0 * z_val
    else:
        predictors_val, predicted_val = None, None

    cat_idx = [c + y.shape[1] for c in cat_idx] if cat_idx is not None else None

    return predictors_train, predictors_val, predicted_train, predicted_val, cat_idx


###################################################
# Score models
###################################################


class ScoreModel(abc.ABC):
    @abc.abstractmethod
    def score(self, y: Float[np.ndarray, "batch y_dim"], X: Float[np.ndarray, "batch x_dim"], t: Int[np.ndarray, "batch 1"]):
        pass

    @abc.abstractmethod
    def fit(self, X: Float[np.ndarray, "batch x_dim"], y: Float[np.ndarray, "batch y_dim"], sde: DiffusionSDE, cat_idx: Optional[List[int]] = None):
        pass


class TreeBasedScoreModel(ScoreModel):
    """
    A generic score model that uses a specified tree-based model (e.g., LightGBM, XGBoost)
    to approximate the score of a given SDE.
    """
    def __init__(
        self,
        model_name: str,
        model_params: dict,
        n_repeats: Optional[int] = 10,
        eval_percent: float = 0.1,
        n_jobs: Optional[int] = -1,
        seed: Optional[int] = None,
        verbose: int = 0,
    ) -> None:
        self.model_name = model_name
        self._model_params = model_params
        self.n_repeats = n_repeats
        self.eval_percent = eval_percent
        self.n_jobs = n_jobs
        self.seed = seed
        self.verbose = verbose

        self.sde = None
        self.models = None
        self.n_estimators_true = None

    def score(
        self,
        y: Float[np.ndarray, "batch y_dim"],
        X: Float[np.ndarray, "batch x_dim"],
        t: Int[np.ndarray, "batch 1"],
    ) -> Float[np.ndarray, "batch y_dim"]:
        if self.sde is None:
            raise ValueError("The model has not been fitted yet.")

        scores = []
        predictors = np.concatenate([y, X, t], axis=1)
        _, std = self.sde.get_mean_std_pt_given_y0(y, t)
        
        for i in range(y.shape[-1]):
            # Use a generic predict call. n_jobs is handled at model initialization.
            score_p = self.models[i].predict(predictors)
            score = score_p / std[:, i]
            scores.append(score)
        return np.array(scores).T

    def fit(
        self,
        X: Float[np.ndarray, "batch x_dim"],
        y: Float[np.ndarray, "batch y_dim"],
        sde: DiffusionSDE,
        cat_idx: Optional[List[int]] = None,
    ):
        y_dim = y.shape[1]
        self.sde = sde

        train_predictors, val_predictors, train_targets, val_targets, processed_cat_idx = _make_training_data(
            X=X, y=y, sde=self.sde,
            n_repeats=self.n_repeats,
            eval_percent=self.eval_percent,
            cat_idx=cat_idx,
            seed=self.seed,
        )

        models = []
        for i in range(y_dim):
            val_targets_i = val_targets[:, i] if val_targets is not None else None
            
            score_model_i = _fit_one_tree_model(
                model_name=self.model_name,
                X=train_predictors,
                y=train_targets[:, i],
                X_val=val_predictors,
                y_val=val_targets_i,
                cat_idx=processed_cat_idx,
                seed=self.seed,
                verbose=self.verbose,
                n_jobs=self.n_jobs,
                **self._model_params,
            )
            models.append(score_model_i)
        self.models = models

        # Collect the true number of trees learned by each model if early stopping was used
        self.n_estimators_true = []
        for model in self.models:
            if hasattr(model, 'best_iteration_'):  # LightGBM with older versions
                self.n_estimators_true.append(model.best_iteration_)
            elif hasattr(model, 'best_iteration'): # LightGBM newer versions & XGBoost
                self.n_estimators_true.append(model.best_iteration)
            elif hasattr(model, 'n_estimators_'): # Scikit-learn models
                self.n_estimators_true.append(model.n_estimators_)
            else:
                self.n_estimators_true.append(self._model_params.get('n_estimators'))


class LightGBMScoreModel(TreeBasedScoreModel):
    """
    Backward-compatible wrapper for the original LightGBMScoreModel.
    This is now a thin wrapper around the more generic TreeBasedScoreModel.
    """
    def __init__(
        self,
        n_repeats: Optional[int] = 10,
        eval_percent: float = 0.1,
        n_jobs: Optional[int] = -1,
        seed: Optional[int] = None,
        verbose: int = 0,
        **lgbm_args,
    ) -> None:
        super().__init__(
            model_name='lightgbm',
            model_params=lgbm_args,
            n_repeats=n_repeats,
            eval_percent=eval_percent,
            n_jobs=n_jobs,
            seed=seed,
            verbose=verbose,
        )
