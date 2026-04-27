"""HMM training with multi-restart Baum-Welch and AIC/BIC model selection."""

from rde.models.hmm import FittedModel, train_hmm
from rde.models.selection import select_n_states
from rde.models.student_t_hmm import StudentTHMM

__all__ = ["FittedModel", "train_hmm", "select_n_states", "StudentTHMM"]
