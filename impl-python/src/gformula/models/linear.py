"""Linear regression models for continuous outcomes."""

from typing import Optional, List, Dict, Any, Tuple
import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf
from scipy import stats

from .base import BaseModel
from .logistic import LogisticModel


class LinearModel(BaseModel):
    """Linear regression model for continuous outcomes."""
    
    def fit(self, data: pd.DataFrame, outcome: str,
            predictors: List[str], **kwargs) -> 'LinearModel':
        """Fit linear regression model.
        
        Args:
            data: Input data
            outcome: Name of continuous outcome variable
            predictors: List of predictor variable names
            **kwargs: Additional arguments for statsmodels
            
        Returns:
            Self for method chaining
        """
        # Validate data
        self.validate_data(data, [outcome] + predictors)
        
        # Handle missing data
        clean_data = self.handle_missing_data(data, outcome, predictors)
        
        # Construct formula
        formula = self.get_formula(outcome, predictors)
        
        # Fit model
        try:
            self.model = smf.ols(formula, data=clean_data)
            self.results = self.model.fit(**kwargs)
            
            # Store results
            self.fitted_values = self.results.fittedvalues
            self.residuals = self.results.resid
            self.coefficients = self.get_coefficients()
            self.diagnostics = self.get_diagnostics()
            
        except Exception as e:
            print(f"Error fitting linear model for {outcome}: {e}")
            raise
            
        return self
    
    def predict(self, data: pd.DataFrame, 
                prediction_interval: bool = False,
                alpha: float = 0.05,
                **kwargs) -> np.ndarray:
        """Generate predictions for new data.
        
        Args:
            data: Data for prediction
            prediction_interval: Whether to return prediction intervals
            alpha: Significance level for intervals
            **kwargs: Additional arguments
            
        Returns:
            Array of predictions (or tuple with intervals if requested)
        """
        if self.results is None:
            raise ValueError("Model must be fitted before prediction")
            
        # Get point predictions
        predictions = self.results.predict(data)
        
        if not prediction_interval:
            return predictions.values
            
        # Calculate prediction intervals
        predict_mean_se = self.results.get_prediction(data).se_mean
        margin = stats.t.ppf(1 - alpha/2, self.results.df_resid) * predict_mean_se
        
        lower = predictions - margin
        upper = predictions + margin
        
        return predictions.values, lower.values, upper.values
    
    def simulate_continuous(self, data: pd.DataFrame,
                          random_state: Optional[int] = None) -> np.ndarray:
        """Simulate continuous outcomes with noise.
        
        Args:
            data: Data for prediction
            random_state: Random state for reproducibility
            
        Returns:
            Array of simulated continuous values
        """
        if random_state is not None:
            np.random.seed(random_state)
            
        # Get predictions
        mean_preds = self.predict(data)
        
        # Add Gaussian noise based on model residual standard deviation
        if hasattr(self.results, 'mse_resid'):
            sigma = np.sqrt(self.results.mse_resid)
        else:
            sigma = np.std(self.residuals)
            
        noise = np.random.normal(0, sigma, size=len(mean_preds))
        
        return mean_preds + noise


class TruncatedLinearModel(LinearModel):
    """Linear regression with truncated normal distribution.
    
    Used for continuous outcomes that are bounded (e.g., cannot be negative).
    """
    
    def __init__(self, formula: Optional[str] = None,
                 lower_bound: Optional[float] = None,
                 upper_bound: Optional[float] = None):
        """Initialize truncated linear model.
        
        Args:
            formula: Model formula
            lower_bound: Lower truncation bound
            upper_bound: Upper truncation bound
        """
        super().__init__(formula)
        self.lower_bound = lower_bound
        self.upper_bound = upper_bound
        
    def simulate_continuous(self, data: pd.DataFrame,
                          random_state: Optional[int] = None) -> np.ndarray:
        """Simulate truncated continuous outcomes.
        
        Args:
            data: Data for prediction
            random_state: Random state for reproducibility
            
        Returns:
            Array of simulated truncated values
        """
        if random_state is not None:
            np.random.seed(random_state)
            
        # Get predictions
        mean_preds = self.predict(data)
        
        # Get residual standard deviation
        if hasattr(self.results, 'mse_resid'):
            sigma = np.sqrt(self.results.mse_resid)
        else:
            sigma = np.std(self.residuals)
            
        # Generate truncated normal samples
        simulated = np.zeros(len(mean_preds))
        
        for i in range(len(mean_preds)):
            # Use scipy's truncated normal
            a = (self.lower_bound - mean_preds[i]) / sigma if self.lower_bound is not None else -np.inf
            b = (self.upper_bound - mean_preds[i]) / sigma if self.upper_bound is not None else np.inf
            
            simulated[i] = stats.truncnorm.rvs(a, b, loc=mean_preds[i], scale=sigma)
            
        return simulated


class ZeroInflatedLinearModel(BaseModel):
    """Zero-inflated linear model for continuous outcomes with excess zeros.
    
    This is a two-part model:
    1. Logistic regression for zero vs non-zero
    2. Linear regression for non-zero values
    """
    
    def __init__(self, formula: Optional[str] = None):
        """Initialize zero-inflated linear model."""
        super().__init__(formula)
        self.zero_model = LogisticModel()
        self.continuous_model = LinearModel()
        
    def fit(self, data: pd.DataFrame, outcome: str,
            predictors: List[str], **kwargs) -> 'ZeroInflatedLinearModel':
        """Fit zero-inflated linear model.
        
        Args:
            data: Input data
            outcome: Name of outcome variable
            predictors: List of predictor variable names
            **kwargs: Additional arguments
            
        Returns:
            Self for method chaining
        """
        # Create binary indicator for zero vs non-zero
        data['is_zero'] = (data[outcome] == 0).astype(int)
        
        # Fit logistic model for zero probability
        self.zero_model.fit(data, 'is_zero', predictors, **kwargs)
        
        # Fit linear model on non-zero values
        nonzero_data = data[data[outcome] > 0].copy()
        if len(nonzero_data) > 0:
            self.continuous_model.fit(nonzero_data, outcome, predictors, **kwargs)
        else:
            print("Warning: No non-zero values to fit continuous model")
            
        # Clean up
        data.drop('is_zero', axis=1, inplace=True)
        
        return self
    
    def predict(self, data: pd.DataFrame, **kwargs) -> np.ndarray:
        """Generate predictions for zero-inflated model.
        
        Args:
            data: Data for prediction
            **kwargs: Additional arguments
            
        Returns:
            Array of expected values
        """
        # Probability of zero
        prob_zero = self.zero_model.predict(data)
        
        # Expected value given non-zero
        nonzero_mean = self.continuous_model.predict(data)
        
        # Combined expectation
        expected = (1 - prob_zero) * nonzero_mean
        
        return expected
    
    def simulate_continuous(self, data: pd.DataFrame,
                          random_state: Optional[int] = None) -> np.ndarray:
        """Simulate zero-inflated continuous outcomes.
        
        Args:
            data: Data for prediction
            random_state: Random state for reproducibility
            
        Returns:
            Array of simulated values
        """
        if random_state is not None:
            np.random.seed(random_state)
            
        n = len(data)
        simulated = np.zeros(n)
        
        # Determine which observations are zero
        prob_zero = self.zero_model.predict(data)
        is_zero = np.random.binomial(1, prob_zero)
        
        # For non-zero observations, simulate from continuous model
        nonzero_idx = np.where(is_zero == 0)[0]
        if len(nonzero_idx) > 0:
            nonzero_data = data.iloc[nonzero_idx]
            nonzero_values = self.continuous_model.simulate_continuous(
                nonzero_data, random_state
            )
            simulated[nonzero_idx] = nonzero_values
            
        return simulated