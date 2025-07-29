"""Logistic regression models for binary outcomes."""

from typing import Optional, List, Dict, Any
import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf
from scipy import stats

from .base import BaseModel


class LogisticModel(BaseModel):
    """Logistic regression model for binary outcomes."""
    
    def fit(self, data: pd.DataFrame, outcome: str,
            predictors: List[str], **kwargs) -> 'LogisticModel':
        """Fit logistic regression model.
        
        Args:
            data: Input data
            outcome: Name of binary outcome variable
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
            self.model = smf.logit(formula, data=clean_data)
            
            # Try to fit with default method
            try:
                self.results = self.model.fit(disp=False, **kwargs)
            except np.linalg.LinAlgError:
                # If singular matrix, try with regularization
                print(f"Warning: Singular matrix for {outcome}, trying with method='bfgs'")
                try:
                    self.results = self.model.fit(method='bfgs', disp=False, **kwargs)
                except:
                    # If still fails, try with different starting values
                    print(f"Warning: Still singular, trying with method='nm'")
                    self.results = self.model.fit(method='nm', disp=False, maxiter=1000, **kwargs)
            
            # Store diagnostics
            self.fitted_values = self.results.fittedvalues
            self.coefficients = self.get_coefficients()
            self.diagnostics = self.get_diagnostics()
            
            # Check for convergence
            if not self.results.converged:
                print(f"Warning: Logistic model did not converge for {outcome}")
                
        except Exception as e:
            print(f"Error fitting logistic model for {outcome}: {e}")
            raise
            
        return self
    
    def predict(self, data: pd.DataFrame, type: str = 'response',
                **kwargs) -> np.ndarray:
        """Generate predictions for new data.
        
        Args:
            data: Data for prediction
            type: Type of prediction ('response' for probabilities, 'linear' for log-odds)
            **kwargs: Additional arguments
            
        Returns:
            Array of predictions
        """
        if self.results is None:
            raise ValueError("Model must be fitted before prediction")
            
        # Get predictions
        if type == 'response':
            predictions = self.results.predict(data)
        elif type == 'linear':
            # Get linear predictor (log-odds)
            predictions = self.results.predict(data, linear=True)
        else:
            raise ValueError(f"Unknown prediction type: {type}")
            
        return predictions.values
    
    def predict_binary(self, data: pd.DataFrame, threshold: float = 0.5,
                      random_state: Optional[int] = None) -> np.ndarray:
        """Generate binary predictions.
        
        Args:
            data: Data for prediction
            threshold: Probability threshold for classification
            random_state: Random state for reproducibility
            
        Returns:
            Array of binary predictions
        """
        # Get probabilities
        probs = self.predict(data, type='response')
        
        # Generate binary outcomes
        if random_state is not None:
            np.random.seed(random_state)
            
        # Random draw based on predicted probabilities
        binary_preds = np.random.binomial(1, probs)
        
        return binary_preds


class PooledLogisticModel(LogisticModel):
    """Pooled logistic regression for survival outcomes.
    
    This model is used for discrete-time survival analysis where
    the outcome is modeled as repeated binary observations.
    """
    
    def __init__(self, formula: Optional[str] = None,
                 time_var: str = 'time',
                 time_method: str = 'concat'):
        """Initialize pooled logistic model.
        
        Args:
            formula: Model formula
            time_var: Name of time variable
            time_method: Method for handling time ('concat' or 'cat')
        """
        super().__init__(formula)
        self.time_var = time_var
        self.time_method = time_method
        
    def fit(self, data: pd.DataFrame, outcome: str,
            predictors: List[str], time_knots: Optional[List[int]] = None,
            **kwargs) -> 'PooledLogisticModel':
        """Fit pooled logistic regression model.
        
        Args:
            data: Input data (person-time format)
            outcome: Name of binary outcome variable
            predictors: List of predictor variable names
            time_knots: Knot points for time splines (if time_method='concat')
            **kwargs: Additional arguments for statsmodels
            
        Returns:
            Self for method chaining
        """
        # Add time to predictors based on method
        time_predictors = self._create_time_predictors(data, time_knots)
        all_predictors = predictors + time_predictors
        
        # Fit using parent class method
        super().fit(data, outcome, all_predictors, **kwargs)
        
        return self
    
    def _create_time_predictors(self, data: pd.DataFrame,
                               time_knots: Optional[List[int]] = None) -> List[str]:
        """Create time predictor variables.
        
        Args:
            data: Input data
            time_knots: Knot points for splines
            
        Returns:
            List of time predictor names
        """
        time_predictors = []
        
        if self.time_method == 'cat':
            # Categorical time - create dummy variables
            time_dummies = pd.get_dummies(data[self.time_var], prefix='time')
            for col in time_dummies.columns:
                data[col] = time_dummies[col]
                time_predictors.append(col)
                
        elif self.time_method == 'concat':
            # Continuous time with splines
            if time_knots is None:
                # Use linear time
                time_predictors.append(self.time_var)
            else:
                # Create restricted cubic splines
                from sklearn.preprocessing import SplineTransformer
                
                spline = SplineTransformer(
                    n_knots=len(time_knots),
                    degree=3,
                    knots=np.array(time_knots).reshape(-1, 1),
                    include_bias=False
                )
                
                time_array = data[self.time_var].values.reshape(-1, 1)
                spline_features = spline.fit_transform(time_array)
                
                # Add spline features to data
                for i in range(spline_features.shape[1]):
                    col_name = f'time_spline_{i+1}'
                    data[col_name] = spline_features[:, i]
                    time_predictors.append(col_name)
                    
        return time_predictors
    
    def get_hazard(self, data: pd.DataFrame) -> np.ndarray:
        """Get hazard probabilities from the model.
        
        Args:
            data: Data for prediction
            
        Returns:
            Array of hazard probabilities
        """
        return self.predict(data, type='response')
    
    def get_survival(self, data: pd.DataFrame, id_var: str) -> pd.DataFrame:
        """Calculate survival probabilities.
        
        Args:
            data: Data in person-time format
            id_var: Name of ID variable
            
        Returns:
            DataFrame with survival probabilities
        """
        # Get hazard probabilities
        data['hazard'] = self.get_hazard(data)
        
        # Calculate survival as cumulative product of (1 - hazard)
        data['survival_contrib'] = 1 - data['hazard']
        
        # Group by ID and calculate cumulative survival
        survival_df = data.groupby(id_var).apply(
            lambda x: x.sort_values(self.time_var)['survival_contrib'].cumprod()
        ).reset_index(name='survival')
        
        # Merge back to original data
        result = data.merge(survival_df, left_index=True, right_on='level_1')
        
        return result[['id_var', self.time_var, 'survival']]