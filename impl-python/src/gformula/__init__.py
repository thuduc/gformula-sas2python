"""
GFORMULA: Python implementation of the parametric g-formula for causal inference.

This package implements the parametric g-formula (g-computation algorithm formula)
for estimating the effects of time-varying treatments in the presence of time-varying
confounding.
"""

__version__ = "0.1.0"

from .core.gformula import GFormula
from .core.data_structures import (
    GFormulaParams,
    CovariateSpec,
    InterventionSpec,
    GFormulaResults,
)

__all__ = [
    "GFormula",
    "GFormulaParams",
    "CovariateSpec",
    "InterventionSpec",
    "GFormulaResults",
]