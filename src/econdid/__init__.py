"""Public API for EconDID Studio."""

from .cleaning import CleaningReport, clean_dataframe, load_dataset
from .mapping import RoleSuggestion, suggest_roles
from .models import AnalysisResult, ModelConfig, PRESETS, fit_model
from .synthetic import generate_ddd_data, generate_did_data

__all__ = [
    "AnalysisResult",
    "CleaningReport",
    "ModelConfig",
    "PRESETS",
    "RoleSuggestion",
    "clean_dataframe",
    "fit_model",
    "generate_ddd_data",
    "generate_did_data",
    "load_dataset",
    "suggest_roles",
]

__version__ = "0.1.0"

