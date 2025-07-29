"""Base model class for gformula."""

from abc import ABC, abstractmethod
from typing import Optional, Dict, Any, List, Union
import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf


class BaseModel(ABC):
    """Abstract base class for statistical models in gformula."""
    
    def __init__(self, formula: Optional[str] = None):
        """Initialize base model.
        
        Args:
            formula: Model formula (if None, will be constructed from predictors)
        """
        self.formula = formula
        self.model = None
        self.results = None
        self.fitted_values = None
        self.residuals = None
        self.coefficients = None
        self.diagnostics = {}
        
    @abstractmethod
    def fit(self, data: pd.DataFrame, outcome: str, 
            predictors: List[str], **kwargs) -> 'BaseModel':
        """Fit the model to data.
        
        Args:
            data: Input data
            outcome: Name of outcome variable
            predictors: List of predictor variable names
            **kwargs: Additional model-specific arguments
            
        Returns:
            Self for method chaining
        """
        pass
    
    @abstractmethod
    def predict(self, data: pd.DataFrame, **kwargs) -> np.ndarray:
        """Generate predictions for new data.
        
        Args:
            data: Data for prediction
            **kwargs: Additional prediction arguments
            
        Returns:
            Array of predictions
        """
        pass
    
    def get_formula(self, outcome: str, predictors: List[str],
                   interactions: Optional[List[str]] = None) -> str:
        """Construct model formula from outcome and predictors.
        
        Args:
            outcome: Name of outcome variable
            predictors: List of predictor names
            interactions: Optional list of interaction terms
            
        Returns:
            Formula string
        """
        if self.formula is not None:
            return self.formula
            
        # Basic formula
        formula_parts = [outcome, "~"]
        
        # Add predictors
        if predictors:
            formula_parts.append(" + ".join(predictors))
        else:
            formula_parts.append("1")  # Intercept only
            
        # Add interactions if specified
        if interactions:
            formula_parts.append(" + " + " + ".join(interactions))
            
        return " ".join(formula_parts)
    
    def get_diagnostics(self) -> Dict[str, Any]:
        """Get model diagnostics.
        
        Returns:
            Dictionary of diagnostic measures
        """
        if self.results is None:
            return {}
            
        diagnostics = {
            'n_obs': self.results.nobs if hasattr(self.results, 'nobs') else None,
            'converged': self.results.converged if hasattr(self.results, 'converged') else True,
        }
        
        # Add model-specific diagnostics
        if hasattr(self.results, 'aic'):
            diagnostics['aic'] = self.results.aic
        if hasattr(self.results, 'bic'):
            diagnostics['bic'] = self.results.bic
        if hasattr(self.results, 'llf'):
            diagnostics['log_likelihood'] = self.results.llf
        if hasattr(self.results, 'rsquared'):
            diagnostics['r_squared'] = self.results.rsquared
        if hasattr(self.results, 'prsquared'):
            diagnostics['pseudo_r_squared'] = self.results.prsquared
            
        return diagnostics
    
    def get_coefficients(self) -> pd.DataFrame:
        """Get model coefficients with statistics.
        
        Returns:
            DataFrame with coefficients and statistics
        """
        if self.results is None:
            return pd.DataFrame()
            
        coef_data = {
            'coefficient': self.results.params,
            'std_error': self.results.bse,
            'z_value': self.results.tvalues if hasattr(self.results, 'tvalues') else None,
            'p_value': self.results.pvalues,
        }
        
        # Add confidence intervals
        if hasattr(self.results, 'conf_int'):
            ci = self.results.conf_int()
            coef_data['ci_lower'] = ci[0]
            coef_data['ci_upper'] = ci[1]
            
        return pd.DataFrame(coef_data)
    
    def validate_data(self, data: pd.DataFrame, variables: List[str]) -> None:
        """Validate that required variables exist in data.
        
        Args:
            data: Input data
            variables: List of required variable names
            
        Raises:
            ValueError: If variables are missing
        """
        missing = [var for var in variables if var not in data.columns]
        if missing:
            raise ValueError(f"Variables not found in data: {missing}")
    
    def handle_missing_data(self, data: pd.DataFrame, 
                          outcome: str,
                          predictors: List[str]) -> pd.DataFrame:
        """Handle missing data by dropping rows with missing values.
        
        Args:
            data: Input data
            outcome: Outcome variable name
            predictors: List of predictor names
            
        Returns:
            Data with missing values handled
        """
        # For now, use complete case analysis
        # Future: implement multiple imputation
        variables = [outcome] + predictors
        return data[variables].dropna()