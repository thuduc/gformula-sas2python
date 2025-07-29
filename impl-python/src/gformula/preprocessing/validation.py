"""Data validation functionality."""

import numpy as np
import pandas as pd
from typing import List, Optional, Tuple, Dict, Any


class DataValidator:
    """Validate data for G-Formula analysis."""
    
    def __init__(self, data: pd.DataFrame, id_var: str, time_var: str):
        """Initialize the data validator.
        
        Args:
            data: Input data frame
            id_var: Name of ID variable
            time_var: Name of time variable
        """
        self.data = data
        self.id_var = id_var
        self.time_var = time_var
        self.validation_results = {}
        
    def validate_structure(self) -> Tuple[bool, List[str]]:
        """Validate basic data structure.
        
        Returns:
            Tuple of (is_valid, error_messages)
        """
        errors = []
        
        # Check if required columns exist
        if self.id_var not in self.data.columns:
            errors.append(f"ID variable '{self.id_var}' not found in data")
        if self.time_var not in self.data.columns:
            errors.append(f"Time variable '{self.time_var}' not found in data")
            
        # Check for duplicates
        if self.id_var in self.data.columns and self.time_var in self.data.columns:
            duplicates = self.data.duplicated(subset=[self.id_var, self.time_var])
            if duplicates.any():
                n_dups = duplicates.sum()
                errors.append(f"Found {n_dups} duplicate ID-time combinations")
                
        # Check time consistency
        if self.time_var in self.data.columns and self.id_var in self.data.columns:
            time_gaps = self._check_time_gaps()
            if time_gaps:
                errors.extend(time_gaps)
                
        self.validation_results['structure'] = (len(errors) == 0, errors)
        return len(errors) == 0, errors
    
    def _check_time_gaps(self) -> List[str]:
        """Check for gaps in time series for each subject."""
        errors = []
        
        for subject_id, group in self.data.groupby(self.id_var):
            times = group[self.time_var].sort_values()
            expected_times = np.arange(times.min(), times.max() + 1)
            
            if len(times) != len(expected_times):
                missing_times = set(expected_times) - set(times)
                if missing_times:
                    errors.append(
                        f"Subject {subject_id} has missing time points: {sorted(missing_times)}"
                    )
                    
        return errors
    
    def validate_outcome(self, outcome: str, outcome_type: str,
                        competing_event: Optional[str] = None,
                        censor: Optional[str] = None) -> Tuple[bool, List[str]]:
        """Validate outcome variable.
        
        Args:
            outcome: Name of outcome variable
            outcome_type: Type of outcome (binsurv, bineofu, cateofu, conteofu)
            competing_event: Name of competing event variable (if applicable)
            censor: Name of censoring variable (if applicable)
            
        Returns:
            Tuple of (is_valid, error_messages)
        """
        errors = []
        
        # Check if outcome exists
        if outcome not in self.data.columns:
            errors.append(f"Outcome variable '{outcome}' not found in data")
            return False, errors
            
        # Validate based on outcome type
        if outcome_type == 'binsurv':
            errors.extend(self._validate_binary_survival(outcome, competing_event, censor))
        elif outcome_type == 'bineofu':
            errors.extend(self._validate_binary_eofu(outcome))
        elif outcome_type == 'cateofu':
            errors.extend(self._validate_categorical_eofu(outcome))
        elif outcome_type == 'conteofu':
            errors.extend(self._validate_continuous_eofu(outcome))
            
        self.validation_results['outcome'] = (len(errors) == 0, errors)
        return len(errors) == 0, errors
    
    def _validate_binary_survival(self, outcome: str, competing_event: Optional[str],
                                 censor: Optional[str]) -> List[str]:
        """Validate binary survival outcome."""
        errors = []
        
        # Check binary values
        unique_vals = self.data[outcome].dropna().unique()
        if not set(unique_vals).issubset({0, 1}):
            errors.append(
                f"Binary survival outcome '{outcome}' has non-binary values: {unique_vals}"
            )
            
        # Check competing event if specified
        if competing_event and competing_event in self.data.columns:
            comp_vals = self.data[competing_event].dropna().unique()
            if not set(comp_vals).issubset({0, 1}):
                errors.append(
                    f"Competing event '{competing_event}' has non-binary values: {comp_vals}"
                )
                
            # Check for conflicts
            both_events = (self.data[outcome] == 1) & (self.data[competing_event] == 1)
            if both_events.any():
                errors.append(
                    f"Found {both_events.sum()} records with both outcome and competing event"
                )
                
        # Check censoring if specified
        if censor and censor in self.data.columns:
            cens_vals = self.data[censor].dropna().unique()
            if not set(cens_vals).issubset({0, 1}):
                errors.append(
                    f"Censoring variable '{censor}' has non-binary values: {cens_vals}"
                )
                
        return errors
    
    def _validate_binary_eofu(self, outcome: str) -> List[str]:
        """Validate binary end-of-follow-up outcome."""
        errors = []
        
        # Check that outcome only appears at last time point
        max_time = self.data[self.time_var].max()
        
        # Non-missing outcomes should only be at max time
        non_missing = self.data[outcome].notna()
        non_max_time = self.data[self.time_var] != max_time
        
        if (non_missing & non_max_time).any():
            errors.append(
                f"Binary EOFU outcome '{outcome}' has values before end of follow-up"
            )
            
        # Check binary values where not missing
        unique_vals = self.data[outcome].dropna().unique()
        if not set(unique_vals).issubset({0, 1}):
            errors.append(
                f"Binary EOFU outcome '{outcome}' has non-binary values: {unique_vals}"
            )
            
        return errors
    
    def _validate_categorical_eofu(self, outcome: str) -> List[str]:
        """Validate categorical end-of-follow-up outcome."""
        errors = []
        
        # Similar to binary EOFU but allow multiple categories
        max_time = self.data[self.time_var].max()
        
        non_missing = self.data[outcome].notna()
        non_max_time = self.data[self.time_var] != max_time
        
        if (non_missing & non_max_time).any():
            errors.append(
                f"Categorical EOFU outcome '{outcome}' has values before end of follow-up"
            )
            
        # Check that values are integers
        if self.data[outcome].dropna().dtype not in ['int64', 'int32', 'int16', 'int8']:
            errors.append(
                f"Categorical EOFU outcome '{outcome}' should have integer values"
            )
            
        return errors
    
    def _validate_continuous_eofu(self, outcome: str) -> List[str]:
        """Validate continuous end-of-follow-up outcome."""
        errors = []
        
        max_time = self.data[self.time_var].max()
        
        non_missing = self.data[outcome].notna()
        non_max_time = self.data[self.time_var] != max_time
        
        if (non_missing & non_max_time).any():
            errors.append(
                f"Continuous EOFU outcome '{outcome}' has values before end of follow-up"
            )
            
        # Check that values are numeric
        try:
            pd.to_numeric(self.data[outcome].dropna())
        except:
            errors.append(
                f"Continuous EOFU outcome '{outcome}' has non-numeric values"
            )
            
        return errors
    
    def validate_covariates(self, covariates: List[str],
                          cov_types: List[int]) -> Tuple[bool, List[str]]:
        """Validate covariate variables.
        
        Args:
            covariates: List of covariate names
            cov_types: List of covariate types
            
        Returns:
            Tuple of (is_valid, error_messages)
        """
        errors = []
        
        for cov, cov_type in zip(covariates, cov_types):
            if cov not in self.data.columns:
                errors.append(f"Covariate '{cov}' not found in data")
                continue
                
            # Validate based on covariate type
            if cov_type in [1, 2]:  # Binary
                unique_vals = self.data[cov].dropna().unique()
                if not set(unique_vals).issubset({0, 1}):
                    errors.append(
                        f"Binary covariate '{cov}' has non-binary values: {unique_vals}"
                    )
            elif cov_type in [3, 4, 5]:  # Continuous variants
                try:
                    pd.to_numeric(self.data[cov].dropna())
                except:
                    errors.append(f"Continuous covariate '{cov}' has non-numeric values")
                    
        self.validation_results['covariates'] = (len(errors) == 0, errors)
        return len(errors) == 0, errors
    
    def get_summary(self) -> Dict[str, Any]:
        """Get summary of validation results.
        
        Returns:
            Dictionary with validation summary
        """
        summary = {
            'n_subjects': self.data[self.id_var].nunique() if self.id_var in self.data.columns else 0,
            'n_records': len(self.data),
            'time_range': (
                self.data[self.time_var].min() if self.time_var in self.data.columns else None,
                self.data[self.time_var].max() if self.time_var in self.data.columns else None
            ),
            'validation_results': self.validation_results,
            'all_valid': all(
                result[0] for result in self.validation_results.values()
            ) if self.validation_results else None
        }
        
        return summary