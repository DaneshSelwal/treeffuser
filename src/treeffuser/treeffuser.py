from __future__ import annotations

from typing import Literal, Optional

from treeffuser._base_tabular_diffusion import BaseTabularDiffusion
# The generic TreeBasedScoreModel should be in _score_models, so we can import it.
from treeffuser._score_models import ScoreModel, TreeBasedScoreModel
from treeffuser.sde import DiffusionSDE
from treeffuser.sde import get_diffusion_sde


class Treeffuser(BaseTabularDiffusion):
    def __init__(
        self,
        # --- NEW PARAMETER to allow injecting a custom score model ---
        score_model: ScoreModel | None = None,
        # -----------------------------------------------------------
        n_repeats: int = 30,
        sde_name: str = "vesde",
        sde_initialize_from_data: bool = False,
        sde_hyperparam_min: float | Literal["default"] | None = None,
        sde_hyperparam_max: float | Literal["default"] | None = None,
        seed: int | None = None,
        verbose: int = 0,
        **score_model_kwargs,
    ):
        """
        score_model : ScoreModel | None
            A pre-initialized score model object (e.g., an instance of TreeBasedScoreModel).
            If None (default), a default LightGBM score model will be created using
            the provided score_model_kwargs.
        n_repeats : int
            How many times to repeat the training dataset when fitting the score. That is, how many
            noisy versions of a point to generate for training.
        sde_name : str
            SDE: Name of the SDE to use. See `treeffuser.sde.get_diffusion_sde` for available SDEs.
        sde_initialize_from_data : bool
            SDE: Whether to initialize the SDE from the data.
        sde_hyperparam_min : float or "default"
            SDE: The scale of the SDE at t=0.
        sde_hyperparam_max : float or "default"
            SDE: The scale of the SDE at t=T.
        seed : int
            Random seed for generating the training data and fitting the model.
        verbose : int
            Verbosity of the score model.
        **score_model_kwargs :
            Keyword arguments passed to the default LightGBM score model if a custom `score_model`
            is NOT provided. Includes arguments like `n_estimators`, `learning_rate`, etc.
        """
        super().__init__(
            sde_initialize_from_data=sde_initialize_from_data,
        )
        self.sde_name = sde_name
        self.seed = seed
        self.verbose = verbose
        self.sde_initialize_from_data = sde_initialize_from_data
        self.sde_hyperparam_min = sde_hyperparam_min
        self.sde_hyperparam_max = sde_hyperparam_max

        # --- MODIFIED LOGIC ---
        # Store the custom score model if provided.
        self._score_model_custom = score_model
        # Store the keyword arguments for the default model.
        self._score_model_kwargs = score_model_kwargs
        self._score_model_kwargs.setdefault("n_repeats", n_repeats)
        # --- END MODIFIED LOGIC ---

    def get_new_sde(self) -> DiffusionSDE:
        sde_cls = get_diffusion_sde(self.sde_name)
        sde_kwargs = {}
        if self.sde_hyperparam_min is not None:
            sde_kwargs["hyperparam_min"] = self.sde_hyperparam_min
        if self.sde_hyperparam_max is not None:
            sde_kwargs["hyperparam_max"] = self.sde_hyperparam_max
        sde = sde_cls(**sde_kwargs)
        return sde

    def get_new_score_model(self) -> ScoreModel:
        # --- MODIFIED LOGIC ---
        # If a custom score model was passed during __init__, return it.
        if self._score_model_custom is not None:
            return self._score_model_custom

        # Otherwise, create the default LightGBM model as before.
        # We assume TreeBasedScoreModel is available and its default is LightGBM,
        # or we create a LightGBMScoreModel directly.
        from treeffuser._score_models import LightGBMScoreModel
        score_model = LightGBMScoreModel(
            seed=self.seed,
            verbose=self.verbose,
            **self._score_model_kwargs,
        )
        return score_model
        # --- END MODIFIED LOGIC ---

    @property
    def n_estimators_true(self) -> list[int]:
        """
        The number of estimators that are actually used in the models (after early stopping).
        """
        if hasattr(self, "score_model") and hasattr(self.score_model, "n_estimators_true"):
             return self.score_model.n_estimators_true
        return []

