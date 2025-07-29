"""Survival analysis models."""

from typing import Optional, List, Dict, Any, Tuple
import numpy as np
import pandas as pd

from .base import BaseModel
from .logistic import PooledLogisticModel


class SurvivalModel(BaseModel):
    """Survival model for time-to-event outcomes.
    
    This wraps the pooled logistic model for discrete-time survival analysis.
    """
    
    def __init__(self, formula: Optional[str] = None,
                 time_var: str = 'time',
                 time_method: str = 'concat'):
        """Initialize survival model.
        
        Args:
            formula: Model formula
            time_var: Name of time variable
            time_method: Method for handling time
        """
        super().__init__(formula)
        self.time_var = time_var
        self.time_method = time_method
        self.pooled_logistic = PooledLogisticModel(
            formula=formula,
            time_var=time_var,
            time_method=time_method
        )
        
    def fit(self, data: pd.DataFrame, outcome: str,
            predictors: List[str], 
            competing_event: Optional[str] = None,
            censor: Optional[str] = None,
            time_knots: Optional[List[int]] = None,
            **kwargs) -> 'SurvivalModel':
        """Fit survival model.
        
        Args:
            data: Input data in person-time format
            outcome: Name of event indicator
            predictors: List of predictor names
            competing_event: Name of competing event variable
            censor: Name of censoring variable
            time_knots: Knot points for time splines
            **kwargs: Additional arguments
            
        Returns:
            Self for method chaining
        """
        # Create composite outcome for pooled logistic regression
        # Event occurs if outcome=1 and not censored/competing
        work_data = data.copy()
        
        # Handle competing events and censoring
        if competing_event is not None and competing_event in data.columns:
            # Censor at competing event
            work_data['_composite_outcome'] = (
                (work_data[outcome] == 1) & 
                (work_data[competing_event] == 0)
            ).astype(int)
        else:
            work_data['_composite_outcome'] = work_data[outcome]
            
        if censor is not None and censor in data.columns:
            # Set outcome to 0 when censored
            work_data.loc[work_data[censor] == 1, '_composite_outcome'] = 0
            
        # Fit pooled logistic model
        self.pooled_logistic.fit(
            work_data, 
            '_composite_outcome',
            predictors,
            time_knots=time_knots,
            **kwargs
        )
        
        # Copy results
        self.results = self.pooled_logistic.results
        self.coefficients = self.pooled_logistic.coefficients
        self.diagnostics = self.pooled_logistic.diagnostics
        
        return self
    
    def predict(self, data: pd.DataFrame, type: str = 'hazard',
                **kwargs) -> np.ndarray:
        """Generate predictions.
        
        Args:
            data: Data for prediction
            type: Type of prediction ('hazard' or 'survival')
            **kwargs: Additional arguments
            
        Returns:
            Array of predictions
        """
        if type == 'hazard':
            return self.pooled_logistic.get_hazard(data)
        elif type == 'survival':
            # Need to calculate cumulative survival
            # This requires the full person-time data
            raise NotImplementedError(
                "Survival prediction requires full person-time data. "
                "Use get_survival_curve() instead."
            )
        else:
            raise ValueError(f"Unknown prediction type: {type}")
            
    def get_survival_curve(self, data: pd.DataFrame, id_var: str) -> pd.DataFrame:
        """Calculate survival curves.
        
        Args:
            data: Person-time format data
            id_var: Name of ID variable
            
        Returns:
            DataFrame with survival probabilities by time
        """
        # Get hazard for each person-time
        data = data.copy()
        data['_hazard'] = self.predict(data, type='hazard')
        
        # Calculate survival as cumulative product of (1 - hazard)
        data['_surv_contrib'] = 1 - data['_hazard']
        
        # Sort by ID and time
        data = data.sort_values([id_var, self.time_var])
        
        # Calculate cumulative survival
        data['survival'] = data.groupby(id_var)['_surv_contrib'].cumprod()
        
        # Also calculate cumulative incidence (1 - survival)
        data['cumulative_incidence'] = 1 - data['survival']
        
        # Clean up temporary columns
        data = data.drop(['_hazard', '_surv_contrib'], axis=1)
        
        return data
    
    def simulate_event_times(self, data: pd.DataFrame,
                           id_var: str,
                           max_time: int,
                           competing_event_model: Optional['SurvivalModel'] = None,
                           censor_model: Optional['SurvivalModel'] = None,
                           random_state: Optional[int] = None) -> pd.DataFrame:
        """Simulate event times for counterfactual scenarios.
        
        Args:
            data: Baseline covariate data (one row per subject)
            id_var: Name of ID variable
            max_time: Maximum follow-up time
            competing_event_model: Optional model for competing events
            censor_model: Optional model for censoring
            random_state: Random state for reproducibility
            
        Returns:
            DataFrame with simulated event times and indicators
        """
        if random_state is not None:
            np.random.seed(random_state)
            
        # Expand to person-time format
        person_time_data = []
        
        for _, subject in data.iterrows():
            subject_id = subject[id_var]
            
            for t in range(max_time):
                # Create person-time record
                pt_record = subject.to_dict()
                pt_record[self.time_var] = t
                
                # Get hazard probability
                hazard = self.predict(pd.DataFrame([pt_record]), type='hazard')[0]
                
                # Simulate event
                event_occurs = np.random.binomial(1, hazard)
                
                # Check competing event if model provided
                competing_occurs = 0
                if competing_event_model is not None:
                    comp_hazard = competing_event_model.predict(
                        pd.DataFrame([pt_record]), type='hazard'
                    )[0]
                    competing_occurs = np.random.binomial(1, comp_hazard)
                    
                # Check censoring if model provided
                censor_occurs = 0
                if censor_model is not None:
                    cens_hazard = censor_model.predict(
                        pd.DataFrame([pt_record]), type='hazard'
                    )[0]
                    censor_occurs = np.random.binomial(1, cens_hazard)
                    
                # Add outcomes
                pt_record['event'] = event_occurs
                pt_record['competing_event'] = competing_occurs
                pt_record['censored'] = censor_occurs
                
                person_time_data.append(pt_record)
                
                # Stop if any event occurs
                if event_occurs or competing_occurs or censor_occurs:
                    break
                    
        return pd.DataFrame(person_time_data)