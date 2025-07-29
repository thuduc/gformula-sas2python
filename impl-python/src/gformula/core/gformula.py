"""Main G-Formula implementation."""

from typing import Dict, List, Optional, Any, Union
import numpy as np
import pandas as pd
import warnings
from datetime import datetime

from .data_structures import GFormulaParams, CovariateSpec, InterventionSpec, GFormulaResults
from ..preprocessing import CovariateProcessor, DataValidator
from ..models import (
    LogisticModel, LinearModel, TruncatedLinearModel, 
    PooledLogisticModel, SurvivalModel
)
from ..simulation import MonteCarloSimulator
from ..utils.bootstrap import Bootstrapper
from ..utils.results import ResultsProcessor


class GFormula:
    """Main class for parametric g-formula analysis."""
    
    def __init__(self, **kwargs):
        """Initialize G-Formula analysis.
        
        Args:
            **kwargs: Parameters matching GFormulaParams dataclass
        """
        # Create parameters object
        self.params = GFormulaParams(**kwargs)
        
        # Initialize components
        self.validator = DataValidator(
            self.params.data,
            self.params.id_var,
            self.params.time_var
        )
        self.processor = CovariateProcessor(
            self.params.data,
            self.params.id_var, 
            self.params.time_var
        )
        
        # Storage for models and results
        self.covariate_models = {}
        self.outcome_model = None
        self.simulation_results = {}
        self.results = None
        
        # Set random seed if specified
        if self.params.seed is not None:
            np.random.seed(self.params.seed)
            
    def fit(self, verbose: bool = True) -> GFormulaResults:
        """Fit the G-Formula model.
        
        Args:
            verbose: Whether to print progress messages
            
        Returns:
            GFormulaResults object
        """
        if verbose:
            print("Starting G-Formula analysis...")
            print(f"Outcome: {self.params.outcome} (type: {self.params.outcome_type})")
            print(f"Number of covariates: {len(self.params.covariates)}")
            print(f"Number of interventions: {len(self.params.interventions)}")
            
        # Step 1: Validate data
        if verbose:
            print("\n1. Validating data...")
        self._validate_data()
        
        # Step 2: Process covariates
        if verbose:
            print("\n2. Processing covariates...")
        self._process_covariates()
        
        # Step 3: Fit covariate models
        if verbose:
            print("\n3. Fitting covariate models...")
        self._fit_covariate_models(verbose)
        
        # Step 4: Fit outcome model
        if verbose:
            print("\n4. Fitting outcome model...")
        self._fit_outcome_model(verbose)
        
        # Step 5: Run simulations
        if verbose:
            print("\n5. Running Monte Carlo simulations...")
        self._run_simulations(verbose)
        
        # Step 6: Calculate results
        if verbose:
            print("\n6. Calculating results...")
        self._calculate_results(verbose)
        
        # Step 7: Bootstrap if requested
        if self.params.n_samples > 0:
            if verbose:
                print("\n7. Running bootstrap...")
            self._run_bootstrap(verbose)
            
        if verbose:
            print("\nG-Formula analysis completed!")
            
        return self.results
    
    def _validate_data(self):
        """Validate input data."""
        # Basic structure validation
        is_valid, errors = self.validator.validate_structure()
        if not is_valid:
            raise ValueError(f"Data validation failed:\n" + "\n".join(errors))
            
        # Outcome validation
        is_valid, errors = self.validator.validate_outcome(
            self.params.outcome,
            self.params.outcome_type,
            self.params.competing_event,
            self.params.censor
        )
        if not is_valid:
            raise ValueError(f"Outcome validation failed:\n" + "\n".join(errors))
            
        # Covariate validation
        cov_names = [c.name for c in self.params.covariates]
        cov_types = [c.cov_type for c in self.params.covariates]
        is_valid, errors = self.validator.validate_covariates(cov_names, cov_types)
        if not is_valid:
            raise ValueError(f"Covariate validation failed:\n" + "\n".join(errors))
            
    def _process_covariates(self):
        """Process covariates to create derived variables."""
        # Process each covariate according to its specification
        self.processor.process_all_covariates(self.params.covariates)
        
        # Create baseline variables for fixed covariates
        if self.params.fixed_covariates:
            self.processor.create_baseline_variables(self.params.fixed_covariates)
            
        # Create time variables if needed
        if self.params.time_method == 'concat' and self.params.time_knots:
            self._create_time_splines()
        elif self.params.time_method == 'cat':
            self._create_time_dummies()
            
        # Update data in params
        self.params.data = self.processor.processed_data
        
    def _create_time_splines(self):
        """Create time spline variables."""
        from sklearn.preprocessing import SplineTransformer
        
        data = self.processor.processed_data
        time_array = data[self.params.time_var].values.reshape(-1, 1)
        
        # Create spline transformer with specified knots
        spline = SplineTransformer(
            n_knots=len(self.params.time_knots),
            degree=3,
            knots=np.array(self.params.time_knots).reshape(-1, 1),
            include_bias=False
        )
        
        spline_features = spline.fit_transform(time_array)
        
        # Add spline features to data
        for i in range(spline_features.shape[1]):
            col_name = f'time_spline_{i+1}'
            data[col_name] = spline_features[:, i]
            
        self.processor.processed_data = data
        
    def _create_time_dummies(self):
        """Create time dummy variables."""
        data = self.processor.processed_data
        time_dummies = pd.get_dummies(data[self.params.time_var], prefix='time')
        
        # Add dummy columns to data
        for col in time_dummies.columns:
            data[col] = time_dummies[col]
            
        self.processor.processed_data = data
        
    def _fit_covariate_models(self, verbose: bool):
        """Fit models for time-varying covariates."""
        for i, cov_spec in enumerate(self.params.covariates):
            if verbose:
                print(f"  Fitting model for {cov_spec.name} "
                      f"(type: {cov_spec.cov_type}, predictor: {cov_spec.predictor_type})")
                      
            # Get predictors for this covariate
            predictors = self._get_covariate_predictors(cov_spec, i)
            
            # Select appropriate model based on covariate type
            if cov_spec.cov_type in [1, 2]:  # Binary
                model = LogisticModel()
            elif cov_spec.cov_type == 3:  # Zero-inflated continuous
                # Would need ZeroInflatedLinearModel
                model = LinearModel()  # Simplified for now
            elif cov_spec.cov_type == 4:  # Continuous non-zero
                model = LinearModel()
            elif cov_spec.cov_type == 5:  # Truncated normal
                model = TruncatedLinearModel(lower_bound=0)
            else:
                raise ValueError(f"Unknown covariate type: {cov_spec.cov_type}")
                
            # Fit the model
            try:
                # Filter to non-missing outcome times
                fit_data = self.params.data[self.params.data[cov_spec.name].notna()]
                model.fit(fit_data, cov_spec.name, predictors)
                self.covariate_models[cov_spec.name] = model
                
                if self.params.check_cov_models and verbose:
                    diagnostics = model.get_diagnostics()
                    print(f"    Converged: {diagnostics.get('converged', True)}, "
                          f"N: {diagnostics.get('n_obs', 'NA')}")
                          
            except Exception as e:
                warnings.warn(f"Failed to fit model for {cov_spec.name}: {e}")
                
    def _get_covariate_predictors(self, cov_spec: CovariateSpec, 
                                 cov_index: int) -> List[str]:
        """Get predictor variables for a covariate model.
        
        Args:
            cov_spec: Covariate specification
            cov_index: Index of covariate in list
            
        Returns:
            List of predictor variable names
        """
        predictors = []
        
        # Add time variable or time splines
        if self.params.time_method == 'concat':
            if self.params.time_knots:
                # Add time spline variables
                for i in range(len(self.params.time_knots)):
                    predictors.append(f'time_spline_{i+1}')
            else:
                predictors.append(self.params.time_var)
        else:
            # Categorical time - add dummies
            predictors.extend([f'time_{t}' for t in range(1, self.params.time_points)])
            
        # Add fixed covariates
        if self.params.fixed_covariates:
            predictors.extend(self.params.fixed_covariates)
            
        # Add previous covariates and their histories
        for j in range(cov_index):
            prev_cov = self.params.covariates[j]
            predictors.append(prev_cov.name)
            
            # Add lagged versions based on predictor type
            if 'lag1' in prev_cov.predictor_type:
                predictors.append(f"{prev_cov.name}_l1")
            if 'lag2' in prev_cov.predictor_type:
                predictors.append(f"{prev_cov.name}_l2")
                
        # Add history of current covariate
        if 'lag1' in cov_spec.predictor_type:
            predictors.append(f"{cov_spec.name}_l1")
        if 'lag2' in cov_spec.predictor_type:
            predictors.append(f"{cov_spec.name}_l2")
            
        return predictors
    
    def _fit_outcome_model(self, verbose: bool):
        """Fit the outcome model."""
        if verbose:
            print(f"  Fitting {self.params.outcome_type} model for {self.params.outcome}")
            
        # Get predictors for outcome
        predictors = self._get_outcome_predictors()
        
        # Select model based on outcome type
        if self.params.outcome_type == 'binsurv':
            model = SurvivalModel(
                time_var=self.params.time_var,
                time_method=self.params.time_method
            )
            # Fit survival model
            fit_data = self.params.data.copy()
            model.fit(
                fit_data,
                self.params.outcome,
                predictors,
                competing_event=self.params.competing_event,
                censor=self.params.censor,
                time_knots=self.params.time_knots
            )
            
        elif self.params.outcome_type == 'bineofu':
            model = LogisticModel()
            # Only use data at final time point
            max_time = self.params.data[self.params.time_var].max()
            fit_data = self.params.data[
                self.params.data[self.params.time_var] == max_time
            ]
            model.fit(fit_data, self.params.outcome, predictors)
            
        elif self.params.outcome_type == 'conteofu':
            model = LinearModel()
            # Only use data at final time point
            max_time = self.params.data[self.params.time_var].max()
            fit_data = self.params.data[
                self.params.data[self.params.time_var] == max_time
            ]
            model.fit(fit_data, self.params.outcome, predictors)
            
        else:
            raise ValueError(f"Outcome type {self.params.outcome_type} not implemented")
            
        self.outcome_model = model
        
        if verbose and self.params.check_cov_models:
            diagnostics = model.get_diagnostics()
            print(f"    Converged: {diagnostics.get('converged', True)}, "
                  f"N: {diagnostics.get('n_obs', 'NA')}")
                  
    def _get_outcome_predictors(self) -> List[str]:
        """Get predictor variables for outcome model."""
        predictors = []
        
        # Add all covariates
        for cov_spec in self.params.covariates:
            predictors.append(cov_spec.name)
            
            # Add derived variables based on predictor type
            if 'lag1' in cov_spec.predictor_type:
                predictors.append(f"{cov_spec.name}_l1")
            if 'lag2' in cov_spec.predictor_type:
                predictors.append(f"{cov_spec.name}_l2")
            if 'cumavg' in cov_spec.predictor_type:
                if 'lag1' in cov_spec.predictor_type:
                    predictors.append(f"{cov_spec.name}_cumavglag1")
                else:
                    predictors.append(f"{cov_spec.name}_cumavg")
                    
        # Add fixed covariates
        if self.params.fixed_covariates:
            predictors.extend(self.params.fixed_covariates)
            
        # Add time variables (for survival outcomes)
        if self.params.outcome_type == 'binsurv':
            if self.params.time_method == 'concat' and self.params.time_knots:
                for i in range(len(self.params.time_knots)):
                    predictors.append(f'time_spline_{i+1}')
            else:
                predictors.append(self.params.time_var)
                
        return predictors
    
    def _run_simulations(self, verbose: bool):
        """Run Monte Carlo simulations."""
        # Prepare models dictionary
        all_models = self.covariate_models.copy()
        all_models[self.params.outcome] = self.outcome_model
        
        # Initialize simulator
        simulator = MonteCarloSimulator(self.params, all_models)
        
        # Get baseline data (time 0)
        baseline_data = self.params.data[
            self.params.data[self.params.time_var] == 0
        ].copy()
        
        # Run natural course
        if verbose:
            print("  Simulating natural course...")
        self.simulation_results[0] = simulator.simulate_natural_course(
            baseline_data, verbose=verbose
        )
        
        # Run each intervention
        for intervention in self.params.interventions:
            if verbose:
                print(f"  Simulating intervention {intervention.int_no}: {intervention.int_label}")
            self.simulation_results[intervention.int_no] = simulator.simulate_intervention(
                baseline_data, intervention.int_no, verbose=verbose
            )
            
    def _calculate_results(self, verbose: bool):
        """Calculate results from simulations."""
        processor = ResultsProcessor(
            self.params,
            self.simulation_results,
            self.params.data
        )
        
        self.results = processor.calculate_results()
        
        if verbose and self.params.print_log_stats:
            print("\nResults Summary:")
            print(self.results.summary())
            
    def _run_bootstrap(self, verbose: bool):
        """Run bootstrap for confidence intervals."""
        bootstrapper = Bootstrapper(
            self.params,
            self.covariate_models,
            self.outcome_model
        )
        
        bootstrap_results = bootstrapper.run_bootstrap(
            n_samples=self.params.n_samples,
            sample_start=self.params.sample_start,
            sample_end=self.params.sample_end,
            verbose=verbose
        )
        
        # Update results with bootstrap CIs
        self.results.bootstrap_samples = bootstrap_results['samples']
        self.results.confidence_intervals = bootstrap_results['confidence_intervals']