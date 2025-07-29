"""Covariate preprocessing functionality."""

import numpy as np
import pandas as pd
from typing import List, Dict, Optional, Tuple
from scipy.interpolate import BSpline
from sklearn.preprocessing import SplineTransformer

from ..core.data_structures import CovariateSpec


class CovariateProcessor:
    """Process covariates for G-Formula analysis."""
    
    def __init__(self, data: pd.DataFrame, id_var: str, time_var: str):
        """Initialize the covariate processor.
        
        Args:
            data: Input data frame
            id_var: Name of ID variable
            time_var: Name of time variable
        """
        self.data = data.copy()
        self.id_var = id_var
        self.time_var = time_var
        self.processed_data = None
        
    def create_lagged_variables(self, covariate: str, lag: int) -> pd.DataFrame:
        """Create lagged variables for a covariate.
        
        Args:
            covariate: Name of covariate
            lag: Number of lags
            
        Returns:
            DataFrame with lagged variables added
        """
        df = self.data.copy()
        
        # Sort by ID and time to ensure correct lagging
        df = df.sort_values([self.id_var, self.time_var])
        
        # Create lagged variable
        lag_name = f"{covariate}_l{lag}"
        df[lag_name] = df.groupby(self.id_var)[covariate].shift(lag)
        
        return df
    
    def create_cumulative_average(self, covariate: str, lag: int = 0) -> pd.DataFrame:
        """Create cumulative average of a covariate.
        
        Args:
            covariate: Name of covariate
            lag: Lag for cumulative average (0 for current, 1 for lag1)
            
        Returns:
            DataFrame with cumulative average added
        """
        df = self.data.copy()
        df = df.sort_values([self.id_var, self.time_var])
        
        # Calculate cumulative average
        if lag == 0:
            cumavg_name = f"{covariate}_cumavg"
            df[cumavg_name] = df.groupby(self.id_var)[covariate].expanding().mean().values
        else:
            # For lagged cumulative average, shift after calculating
            cumavg_name = f"{covariate}_cumavglag{lag}"
            cumavg = df.groupby(self.id_var)[covariate].expanding().mean()
            df[cumavg_name] = cumavg.groupby(self.id_var).shift(lag).values
            
        return df
    
    def create_time_switch_variable(self, covariate: str, switch_type: int) -> pd.DataFrame:
        """Create time switch variable.
        
        Time switch variables track when a binary covariate switches from 0 to 1.
        
        Args:
            covariate: Name of binary covariate
            switch_type: 0 for tsswitch0, 1 for tsswitch1
            
        Returns:
            DataFrame with time switch variable added
        """
        df = self.data.copy()
        df = df.sort_values([self.id_var, self.time_var])
        
        switch_name = f"{covariate}_tsswitch{switch_type}"
        
        # Create empty column
        df[switch_name] = 0
        
        # Process each subject
        for subject_id in df[self.id_var].unique():
            mask = df[self.id_var] == subject_id
            subject_data = df[mask].sort_values(self.time_var)
            
            # Find first time covariate equals 1
            switch_times = subject_data[subject_data[covariate] == 1][self.time_var]
            
            if len(switch_times) == 0:
                # Never switches to 1
                if switch_type == 0:
                    df.loc[mask, switch_name] = 0  # Always 0
                else:
                    df.loc[mask, switch_name] = subject_data[self.time_var] + 1  # Time + 1
            else:
                switch_time = switch_times.min()
                
                if switch_type == 0:
                    # 0 before switch, 1 after
                    df.loc[mask, switch_name] = (subject_data[self.time_var] >= switch_time).astype(int).values
                else:
                    # Time since switch (0 before switch)
                    times = subject_data[self.time_var].values
                    result = np.maximum(0, times - switch_time + 1)
                    result[times < switch_time] = 0
                    df.loc[mask, switch_name] = result
        
        return df
    
    def create_polynomial_terms(self, covariate: str, degree: int, lag: int = 0) -> pd.DataFrame:
        """Create polynomial terms for a covariate.
        
        Args:
            covariate: Name of covariate
            degree: Polynomial degree (2 for quadratic, 3 for cubic)
            lag: Lag to apply before creating polynomial
            
        Returns:
            DataFrame with polynomial terms added
        """
        df = self.data.copy()
        
        # Get base variable (with lag if specified)
        if lag > 0:
            base_var = f"{covariate}_l{lag}"
            if base_var not in df.columns:
                df = self.create_lagged_variables(covariate, lag)
        else:
            base_var = covariate
            
        # Create polynomial terms
        for d in range(2, degree + 1):
            poly_name = f"{base_var}_pow{d}"
            df[poly_name] = df[base_var] ** d
            
        return df
    
    def create_spline_terms(self, covariate: str, n_knots: int, lag: int = 0) -> pd.DataFrame:
        """Create penalized spline terms for a covariate.
        
        Args:
            covariate: Name of covariate
            n_knots: Number of knots (3, 4, or 5)
            lag: Lag to apply before creating splines
            
        Returns:
            DataFrame with spline terms added
        """
        df = self.data.copy()
        
        # Get base variable
        if lag > 0:
            base_var = f"{covariate}_l{lag}"
            if base_var not in df.columns:
                df = self.create_lagged_variables(covariate, lag)
        else:
            base_var = covariate
            
        # Create spline transformer
        spline = SplineTransformer(n_knots=n_knots, degree=3, include_bias=False)
        
        # Fit and transform
        X = df[base_var].values.reshape(-1, 1)
        spline_features = spline.fit_transform(X)
        
        # Add spline features to dataframe
        for i in range(spline_features.shape[1]):
            spline_name = f"{base_var}_spline{n_knots}_{i+1}"
            df[spline_name] = spline_features[:, i]
            
        return df
    
    def process_covariate(self, cov_spec: CovariateSpec) -> pd.DataFrame:
        """Process a single covariate according to its specification.
        
        Args:
            cov_spec: Covariate specification
            
        Returns:
            DataFrame with processed covariate
        """
        df = self.data.copy()
        
        # Handle different predictor types
        if cov_spec.predictor_type == 'lag1bin':
            df = self.create_lagged_variables(cov_spec.name, 1)
        elif cov_spec.predictor_type == 'lag2bin':
            df = self.create_lagged_variables(cov_spec.name, 1)
            df = self.create_lagged_variables(cov_spec.name, 2)
        elif cov_spec.predictor_type == 'lag1':
            df = self.create_lagged_variables(cov_spec.name, 1)
        elif cov_spec.predictor_type == 'lag2':
            df = self.create_lagged_variables(cov_spec.name, 1)
            df = self.create_lagged_variables(cov_spec.name, 2)
        elif cov_spec.predictor_type == 'lag3':
            df = self.create_lagged_variables(cov_spec.name, 1)
            df = self.create_lagged_variables(cov_spec.name, 2)
            df = self.create_lagged_variables(cov_spec.name, 3)
        elif cov_spec.predictor_type == 'cumavg':
            df = self.create_cumulative_average(cov_spec.name, 0)
        elif cov_spec.predictor_type == 'cumavglag1':
            df = self.create_cumulative_average(cov_spec.name, 1)
        elif cov_spec.predictor_type == 'tsswitch0':
            df = self.create_time_switch_variable(cov_spec.name, 0)
        elif cov_spec.predictor_type == 'tsswitch1':
            df = self.create_time_switch_variable(cov_spec.name, 1)
        elif cov_spec.predictor_type == 'lag1quad':
            df = self.create_lagged_variables(cov_spec.name, 1)
            df = self.create_polynomial_terms(cov_spec.name, 2, lag=1)
        elif cov_spec.predictor_type == 'lag2quad':
            df = self.create_lagged_variables(cov_spec.name, 1)
            df = self.create_lagged_variables(cov_spec.name, 2)
            df = self.create_polynomial_terms(cov_spec.name, 2, lag=2)
        elif cov_spec.predictor_type == 'lag1cub':
            df = self.create_lagged_variables(cov_spec.name, 1)
            df = self.create_polynomial_terms(cov_spec.name, 3, lag=1)
        elif cov_spec.predictor_type == 'lag2cub':
            df = self.create_lagged_variables(cov_spec.name, 1)
            df = self.create_lagged_variables(cov_spec.name, 2)
            df = self.create_polynomial_terms(cov_spec.name, 3, lag=2)
        elif cov_spec.predictor_type.startswith('pspline'):
            n_knots = int(cov_spec.predictor_type[-1])
            df = self.create_spline_terms(cov_spec.name, n_knots)
            
        self.data = df
        return df
    
    def process_all_covariates(self, covariates: List[CovariateSpec]) -> pd.DataFrame:
        """Process all covariates.
        
        Args:
            covariates: List of covariate specifications
            
        Returns:
            DataFrame with all processed covariates
        """
        for cov_spec in covariates:
            self.data = self.process_covariate(cov_spec)
            
        self.processed_data = self.data
        return self.data
    
    def create_baseline_variables(self, covariates: List[str]) -> pd.DataFrame:
        """Create baseline (time 0) variables for specified covariates.
        
        Args:
            covariates: List of covariate names
            
        Returns:
            DataFrame with baseline variables added
        """
        df = self.data.copy()
        df = df.sort_values([self.id_var, self.time_var])
        
        # Get baseline values (first time point for each subject)
        baseline = df.groupby(self.id_var).first()
        
        for cov in covariates:
            if cov in baseline.columns:
                baseline_name = f"{cov}_b"
                # Merge baseline values back to main dataframe
                df = df.merge(
                    baseline[[cov]].rename(columns={cov: baseline_name}),
                    left_on=self.id_var,
                    right_index=True,
                    how='left'
                )
                
        self.data = df
        return df