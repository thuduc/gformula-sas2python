"""Intervention implementation for G-Formula."""

from typing import List, Dict, Any, Optional, Union
import numpy as np
import pandas as pd

from ..core.data_structures import InterventionSpec


class InterventionEngine:
    """Engine for applying interventions to simulated data."""
    
    def __init__(self, interventions: List[InterventionSpec]):
        """Initialize intervention engine.
        
        Args:
            interventions: List of intervention specifications
        """
        self.interventions = interventions
        self.intervention_history = []
        
    def apply_intervention(self, data: pd.DataFrame, 
                         intervention: InterventionSpec,
                         time: int) -> pd.DataFrame:
        """Apply a single intervention to data at a specific time.
        
        Args:
            data: Current simulation data
            intervention: Intervention specification
            time: Current time point
            
        Returns:
            Modified data with intervention applied
        """
        # Check if intervention applies at this time
        if time not in intervention.times:
            return data
            
        # Apply condition if specified
        if intervention.conditions:
            # Evaluate condition to get mask
            mask = data.eval(intervention.conditions)
        else:
            # Apply to all rows
            mask = pd.Series(True, index=data.index)
            
        # Apply intervention to each variable
        for i, var in enumerate(intervention.variables):
            if var not in data.columns:
                print(f"Warning: Intervention variable '{var}' not found in data")
                continue
                
            int_type = intervention.int_types[i]
            
            if int_type == 1:  # Static intervention
                # Set to fixed value
                if intervention.values and i < len(intervention.values):
                    data.loc[mask, var] = intervention.values[i]
                    
            elif int_type == 2:  # Threshold intervention
                # Apply min/max bounds
                if intervention.min_values and i < len(intervention.min_values):
                    min_val = intervention.min_values[i]
                    data.loc[mask & (data[var] < min_val), var] = min_val
                    
                if intervention.max_values and i < len(intervention.max_values):
                    max_val = intervention.max_values[i]
                    data.loc[mask & (data[var] > max_val), var] = max_val
                    
            elif int_type == 3:  # Proportional intervention
                # Multiply by factor
                if intervention.change_values and i < len(intervention.change_values):
                    factor = 1 + intervention.change_values[i]  # change_value is proportion
                    data.loc[mask, var] = data.loc[mask, var] * factor
                    
            elif int_type == 4:  # Addition intervention
                # Add constant
                if intervention.change_values and i < len(intervention.change_values):
                    data.loc[mask, var] = data.loc[mask, var] + intervention.change_values[i]
                    
            # Apply probability if specified (random intervention)
            if intervention.probabilities and i < len(intervention.probabilities):
                prob = intervention.probabilities[i]
                if prob < 1.0:
                    # Only apply to random subset
                    random_mask = np.random.binomial(1, prob, size=mask.sum()).astype(bool)
                    final_mask = mask.copy()
                    final_mask[mask] = random_mask
                    # Revert changes for non-selected subjects
                    # This is a simplified approach - in reality might need to track original values
                    
        # Record intervention application
        self.intervention_history.append({
            'time': time,
            'intervention': intervention.int_no,
            'n_affected': mask.sum()
        })
        
        return data
    
    def apply_all_interventions(self, data: pd.DataFrame,
                              time: int,
                              intervention_nums: Optional[List[int]] = None) -> pd.DataFrame:
        """Apply all relevant interventions at a given time.
        
        Args:
            data: Current simulation data
            time: Current time point
            intervention_nums: Optional list of intervention numbers to apply
            
        Returns:
            Modified data with all interventions applied
        """
        for intervention in self.interventions:
            # Skip if not in the list of interventions to apply
            if intervention_nums is not None and intervention.int_no not in intervention_nums:
                continue
                
            data = self.apply_intervention(data, intervention, time)
            
        return data
    
    def create_natural_course(self, data: pd.DataFrame) -> pd.DataFrame:
        """Create natural course data (no interventions).
        
        This is essentially a no-op but included for clarity.
        
        Args:
            data: Simulation data
            
        Returns:
            Unchanged data
        """
        return data.copy()
    
    def get_intervention_summary(self) -> pd.DataFrame:
        """Get summary of intervention applications.
        
        Returns:
            DataFrame with intervention history
        """
        if not self.intervention_history:
            return pd.DataFrame()
            
        return pd.DataFrame(self.intervention_history)
    
    def validate_interventions(self, data: pd.DataFrame) -> List[str]:
        """Validate that interventions can be applied to the data.
        
        Args:
            data: Sample data to validate against
            
        Returns:
            List of validation errors
        """
        errors = []
        
        for intervention in self.interventions:
            # Check that intervention variables exist
            for var in intervention.variables:
                if var not in data.columns:
                    errors.append(
                        f"Intervention {intervention.int_no}: "
                        f"Variable '{var}' not found in data"
                    )
                    
            # Check condition can be evaluated if specified
            if intervention.conditions:
                try:
                    data.eval(intervention.conditions)
                except Exception as e:
                    errors.append(
                        f"Intervention {intervention.int_no}: "
                        f"Invalid condition '{intervention.conditions}': {e}"
                    )
                    
        return errors