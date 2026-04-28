"""HMM training with multi-restart Baum-Welch and AIC/BIC model selection."""

from rde.models.change_point import (
    BOCPDConfig,
    BOCPDResult,
    align_bocpd_to_regimes,
    bocpd,
)
from rde.models.ensemble import (
    EnsembleConfig,
    EnsembleResult,
    align_states,
    train_ensemble_hmm,
)
from rde.models.garch_hmm import (
    GARCHFit,
    GARCHParams,
    fit_regime_garch,
    forecast_garch_vol,
    garch_conditional_vol,
)
from rde.models.hmm import FittedModel, train_hmm
from rde.models.persistence import load_model, save_model
from rde.models.selection import select_n_states
from rde.models.semi_markov import (
    HSMMConfig,
    NegBinDuration,
    apply_hsmm_correction,
    fit_duration_params,
)
from rde.models.spectral_init import (
    SpectralInitResult,
    apply_spectral_init_to_hmm,
    spectral_init_hmm_params,
    spectral_kmeans_init,
    tensor_power_init,
)
from rde.models.student_t_hmm import StudentTHMM

__all__ = [
    # change point detection
    "BOCPDConfig",
    "BOCPDResult",
    "bocpd",
    "align_bocpd_to_regimes",
    # core
    "FittedModel",
    "train_hmm",
    "select_n_states",
    "save_model",
    "load_model",
    # student-t
    "StudentTHMM",
    # garch-hmm
    "GARCHParams",
    "GARCHFit",
    "fit_regime_garch",
    "garch_conditional_vol",
    "forecast_garch_vol",
    # semi-markov / hsmm
    "NegBinDuration",
    "HSMMConfig",
    "fit_duration_params",
    "apply_hsmm_correction",
    # spectral init
    "SpectralInitResult",
    "spectral_kmeans_init",
    "tensor_power_init",
    "spectral_init_hmm_params",
    "apply_spectral_init_to_hmm",
    # ensemble
    "EnsembleConfig",
    "EnsembleResult",
    "train_ensemble_hmm",
    "align_states",
]
