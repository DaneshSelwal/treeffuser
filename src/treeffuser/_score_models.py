"""
Contains different score models to be used to approximate the score of a given SDE.
This version has been modified to support multiple tree-based models as a backend
and handles complex hyperparameter dictionaries to prevent TypeErrors.
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
    **model_args,
) -> Any:
    """
    A universal wrapper for fitting different tree-based models.
    This function is now more robust and handles API differences between libraries.
    """
    eval_set = None if X_val is None else [(X_val, y_val)]
    verbose = model_args.get("verbose", 0)

    # --- LightGBM ---
    if model_name == 'lightgbm':
        import lightgbm as lgb
        callbacks = None
        early_stopping_rounds = model_args.pop('early_stopping_rounds', None)
        if early_stopping_rounds is not None and eval_set is not None:
            callbacks = [lgb.early_stopping(early_stopping_rounds, verbose=verbose)]

        fit_kwargs = {'eval_set': eval_set, 'callbacks': callbacks}
        if 'cat_idx' in model_args and model_args['cat_idx'] is not None:
            fit_kwargs['categorical_feature'] = model_args.pop('cat_idx')

        model = lgb.LGBMRegressor(**model_args)
        model.fit(X=X, y=y, **fit_kwargs)
        return model

    # --- XGBoost ---
    elif model_name == 'xgboost':
        import xgboost as xgb
        early_stopping_rounds = model_args.pop('early_stopping_rounds', None)
        fit_params = {}
        if early_stopping_rounds is not None and eval_set is not None:
            fit_params['early_stopping_rounds'] = early_stopping_rounds
            fit_params['verbose'] = verbose > 0
        model_args.pop('cat_idx', None)
        model = xgb.XGBRegressor(**model_args)
        model.fit(X, y, eval_set=eval_set, **fit_params)
        return model

    # --- CatBoost ---
    elif model_name == 'catboost':
        import catboost
        if 'n_jobs' in model_args:
            model_args['thread_count'] = model_args.pop('n_jobs')

        # FINAL FIX: CatBoost takes 'cat_features', not 'cat_idx'.
        # We handle this translation and remove the original key.
        if 'cat_idx' in model_args and model_args['cat_idx'] is not None:
            model_args['cat_features'] = model_args.pop('cat_idx')
        else:
            model_args.pop('cat_idx', None)

        model = catboost.CatBoostRegressor(**model_args)
        model.fit(X, y, eval_set=eval_set, early_stopping_rounds=model_args.pop('early_stopping_rounds', None))
        return model

    # --- Scikit-learn Models ---
    elif model_name in ['random_forest', 'gradient_boosting', 'hist_gradient_boosting']:
        model_args.pop('cat_idx', None)
        model_args.pop('early_stopping_rounds', None)
        
        if model_name == 'random_forest':
            from sklearn.ensemble import RandomForestRegressor
            model = RandomForestRegressor(**model_args)
        elif model_name == 'gradient_boosting':
            from sklearn.ensemble import GradientBoostingRegressor
            model_args.pop('n_jobs', None)
            model = GradientBoostingRegressor(**model_args)
        else:
            from sklearn.ensemble import HistGradientBoostingRegressor
            model_args.pop('n_jobs', None)
            model = HistGradientBoostingRegressor(**model_args)
        model.fit(X, y)
        return model

    # --- GPBoost ---
    elif model_name == 'gpboost':
        import gpboost as gpb
        callbacks = None
        early_stopping_rounds = model_args.pop('early_stopping_rounds', None)
        if early_stopping_rounds is not None and eval_set is not None:
            callbacks = [gpb.early_stopping(early_stopping_rounds, verbose=verbose)]
        model_args.pop('cat_idx', None)
        model = gpb.GPBoostRegressor(**model_args)
        model.fit(X=X, y=y, eval_set=eval_set, callbacks=callbacks)
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
    EPS = 1e-5
    T = sde.T
    rng = np.random.default_rng(seed)

    X_train, X_test, y_train, y_test = X, None, y, None
    if eval_percent is not None and eval_percent > 0:
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=eval_percent, random_state=seed
        )
    else:
        X_train, y_train = X, y

    X_train_repeated = np.tile(X_train, (n_repeats, 1))
    y_train_repeated = np.tile(y_train, (n_repeats, 1))
    t_train = rng.uniform(0, 1, size=(y_train_repeated.shape[0], 1)) * (T - EPS) + EPS
    z_train = rng.normal(size=y_train_repeated.shape)
    train_mean, train_std = sde.get_mean_std_pt_given_y0(y_train_repeated, t_train)
    perturbed_y_train = train_mean + train_std * z_train
    predictors_train = np.concatenate([perturbed_y_train, X_train_repeated, t_train], axis=1)
    predicted_train = -1.0 * z_train

    if X_test is not None and y_test is not None:
        t_val = rng.uniform(0, 1, size=(y_test.shape[0], 1)) * (T - EPS) + EPS
        z_val = rng.normal(size=y_test.shape)
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

            params_for_fit = self._model_params.copy()
            params_for_fit['n_jobs'] = self.n_jobs
            params_for_fit['verbose'] = self.verbose
            
            if self.model_name == 'catboost':
                 params_for_fit['random_seed'] = self.seed
            else:
                 params_for_fit['random_state'] = self.seed

            score_model_i = _fit_one_tree_model(
                model_name=self.model_name,
                X=train_predictors,
                y=train_targets[:, i],
                X_val=val_predictors,
                y_val=val_targets_i,
                cat_idx=processed_cat_idx,
                **params_for_fit,
            )
            models.append(score_model_i)
        self.models = models
        self.n_estimators_true = []
        for model in self.models:
            if hasattr(model, 'best_iteration_') and model.best_iteration_ is not None:
                self.n_estimators_true.append(model.best_iteration_)
            elif hasattr(model, 'best_iteration') and model.best_iteration is not None:
                self.n_estimators_true.append(model.best_iteration)
            elif hasattr(model, 'n_estimators_'):
                self.n_estimators_true.append(model.n_estimators_)
            else:
                self.n_estimators_true.append(self._model_params.get('n_estimators'))

class LightGBMScoreModel(TreeBasedScoreModel):
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
