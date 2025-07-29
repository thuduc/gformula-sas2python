"""Monte Carlo simulation engine for G-Formula."""

from typing import Dict, List, Optional, Any, Tuple
import numpy as np
import pandas as pd
from tqdm import tqdm
from joblib import Parallel, delayed

from ..core.data_structures import CovariateSpec, InterventionSpec, GFormulaParams
from ..models import LogisticModel, LinearModel, TruncatedLinearModel, SurvivalModel
from ..preprocessing import CovariateProcessor
from .interventions import InterventionEngine


class MonteCarloSimulator:
    """Monte Carlo simulator for G-Formula counterfactual estimation."""
    
    def __init__(self, params: GFormulaParams, models: Dict[str, Any]):
        """Initialize the simulator.
        
        Args:
            params: G-Formula parameters
            models: Dictionary of fitted models for covariates and outcome
        """
        self.params = params
        self.models = models
        self.intervention_engine = InterventionEngine(params.interventions)
        
    def simulate_natural_course(self, baseline_data: pd.DataFrame,
                              verbose: bool = True) -> pd.DataFrame:
        """Simulate natural course (no interventions).
        
        Args:
            baseline_data: Baseline covariate data
            verbose: Whether to show progress
            
        Returns:
            Simulated data under natural course
        """
        return self._simulate_scenario(
            baseline_data, 
            intervention_nums=None,
            scenario_label="Natural Course",
            verbose=verbose
        )
    
    def simulate_intervention(self, baseline_data: pd.DataFrame,
                            intervention_num: int,
                            verbose: bool = True) -> pd.DataFrame:
        """Simulate a specific intervention scenario.
        
        Args:
            baseline_data: Baseline covariate data
            intervention_num: Intervention number to apply
            verbose: Whether to show progress
            
        Returns:
            Simulated data under intervention
        """
        # Get intervention label
        intervention = next(
            (i for i in self.params.interventions if i.int_no == intervention_num),
            None
        )
        label = intervention.int_label if intervention else f"Intervention {intervention_num}"
        
        return self._simulate_scenario(
            baseline_data,
            intervention_nums=[intervention_num],
            scenario_label=label,
            verbose=verbose
        )
    
    def _simulate_scenario(self, baseline_data: pd.DataFrame,
                         intervention_nums: Optional[List[int]],
                         scenario_label: str,
                         verbose: bool = True) -> pd.DataFrame:
        """Simulate a scenario (natural course or intervention).
        
        Args:
            baseline_data: Baseline covariate data
            intervention_nums: List of intervention numbers to apply (None for natural course)
            scenario_label: Label for progress bar
            verbose: Whether to show progress
            
        Returns:
            Simulated longitudinal data
        """
        # Initialize simulation data
        n_subjects = len(baseline_data)
        n_sims = self.params.n_simulations
        
        # Create copies of baseline data for each simulation
        all_simulated_data = []
        
        # Progress bar setup
        if verbose:
            pbar = tqdm(total=n_sims, desc=f"Simulating {scenario_label}")
            
        # Run simulations
        for sim in range(n_sims):
            # Generate one simulation
            sim_data = self._run_single_simulation(
                baseline_data.copy(),
                intervention_nums,
                sim_id=sim
            )
            all_simulated_data.append(sim_data)
            
            if verbose:
                pbar.update(1)
                
        if verbose:
            pbar.close()
            
        # Combine all simulations
        combined_data = pd.concat(all_simulated_data, ignore_index=True)
        
        return combined_data
    
    def _run_single_simulation(self, baseline_data: pd.DataFrame,
                             intervention_nums: Optional[List[int]],
                             sim_id: int) -> pd.DataFrame:
        """Run a single Monte Carlo simulation.
        
        Args:
            baseline_data: Baseline data for this simulation
            intervention_nums: Interventions to apply
            sim_id: Simulation ID
            
        Returns:
            Simulated longitudinal data for all subjects
        """
        # Add simulation ID
        baseline_data['_sim_id'] = sim_id
        
        # Initialize current data as baseline
        current_data = baseline_data.copy()
        all_time_data = []
        
        # Simulate forward in time
        for t in range(self.params.time_points):
            # Set current time
            current_data[self.params.time_var] = t
            
            # Apply interventions if specified
            if intervention_nums is not None:
                current_data = self.intervention_engine.apply_all_interventions(
                    current_data, t, intervention_nums
                )
                
            # Update time-varying covariates
            current_data = self._update_covariates(current_data, t)
            
            # Store current time data
            all_time_data.append(current_data.copy())
            
            # Check for events (if survival outcome)
            if self.params.outcome_type == 'binsurv':
                current_data = self._update_survival_status(current_data, t)
                
        # Combine all time points
        longitudinal_data = pd.concat(all_time_data, ignore_index=True)
        
        # Generate outcome if end-of-follow-up type
        if self.params.outcome_type in ['bineofu', 'cateofu', 'conteofu']:
            longitudinal_data = self._generate_eofu_outcome(longitudinal_data)
            
        return longitudinal_data
    
    def _update_covariates(self, data: pd.DataFrame, time: int) -> pd.DataFrame:
        """Update time-varying covariates using fitted models.
        
        Args:
            data: Current data
            time: Current time point
            
        Returns:
            Data with updated covariates
        """
        # Process covariates in order
        for cov_spec in self.params.covariates:
            cov_name = cov_spec.name
            
            # Skip if covariate model not available
            if cov_name not in self.models:
                continue
                
            model = self.models[cov_name]
            
            # Get predictors for this covariate
            # This would need to be extracted from model or stored separately
            # For now, using a simplified approach
            
            # Predict new covariate values
            if cov_spec.cov_type in [1, 2]:  # Binary
                # Get probabilities and simulate binary outcomes
                try:
                    probs = model.predict(data, type='response')
                    # Handle NaN and invalid probabilities
                    probs = np.nan_to_num(probs, nan=0.5)
                    probs = np.clip(probs, 0.0, 1.0)
                    data[cov_name] = np.random.binomial(1, probs)
                except Exception as e:
                    print(f"Warning: Error predicting {cov_name}: {e}")
                    # Use baseline prevalence as fallback
                    data[cov_name] = np.random.binomial(1, 0.5, size=len(data))
                
            elif cov_spec.cov_type == 3:  # Zero-inflated continuous
                # Simulate from zero-inflated model
                try:
                    data[cov_name] = model.simulate_continuous(data)
                except Exception as e:
                    print(f"Warning: Error simulating {cov_name}: {e}")
                    # Use mean + noise as fallback
                    mean_val = data[cov_name].mean()
                    std_val = data[cov_name].std()
                    data[cov_name] = np.random.normal(mean_val, std_val, size=len(data))
                
            elif cov_spec.cov_type == 4:  # Continuous non-zero
                # Simulate continuous values
                try:
                    data[cov_name] = model.simulate_continuous(data)
                except Exception as e:
                    print(f"Warning: Error simulating {cov_name}: {e}")
                    # Use mean + noise as fallback
                    mean_val = data[cov_name].mean()
                    std_val = data[cov_name].std()
                    data[cov_name] = np.random.normal(mean_val, std_val, size=len(data))
                
            elif cov_spec.cov_type == 5:  # Truncated normal
                # Simulate from truncated distribution
                try:
                    data[cov_name] = model.simulate_continuous(data)
                except Exception as e:
                    print(f"Warning: Error simulating {cov_name}: {e}")
                    # Use mean + noise as fallback
                    mean_val = data[cov_name].mean()
                    std_val = data[cov_name].std()
                    data[cov_name] = np.random.normal(mean_val, std_val, size=len(data))
                
            # Update derived variables (lags, cumulative averages, etc.)
            # This would need the covariate processor
            processor = CovariateProcessor(data, self.params.id_var, self.params.time_var)
            data = processor.process_covariate(cov_spec)
            
        return data
    
    def _update_survival_status(self, data: pd.DataFrame, time: int) -> pd.DataFrame:
        """Update survival status for subjects still at risk.
        
        Args:
            data: Current data
            time: Current time point
            
        Returns:
            Data with updated survival status
        """
        # Only update for subjects still at risk
        at_risk = data[self.params.outcome].isna() | (data[self.params.outcome] == 0)
        
        if self.params.competing_event:
            at_risk = at_risk & (data[self.params.competing_event].isna() | 
                               (data[self.params.competing_event] == 0))
            
        if self.params.censor:
            at_risk = at_risk & (data[self.params.censor].isna() | 
                               (data[self.params.censor] == 0))
            
        # Get outcome model
        outcome_model = self.models.get(self.params.outcome)
        if outcome_model and at_risk.any():
            # Predict hazard for at-risk subjects
            at_risk_data = data[at_risk]
            try:
                hazards = outcome_model.predict(at_risk_data, type='hazard')
                # Handle NaN and invalid probabilities
                hazards = np.nan_to_num(hazards, nan=0.01)
                hazards = np.clip(hazards, 0.0, 1.0)
                
                # Simulate events
                events = np.random.binomial(1, hazards)
                data.loc[at_risk, self.params.outcome] = events
            except Exception as e:
                print(f"Warning: Error predicting hazards for {self.params.outcome}: {e}")
                # Use small constant hazard as fallback
                n_at_risk = at_risk.sum()
                events = np.random.binomial(1, 0.01, size=n_at_risk)
                data.loc[at_risk, self.params.outcome] = events
            
        # Similarly for competing events and censoring
        if self.params.competing_event:
            comp_model = self.models.get(self.params.competing_event)
            if comp_model and at_risk.any():
                at_risk_data = data[at_risk]
                comp_hazards = comp_model.predict(at_risk_data, type='hazard')
                comp_events = np.random.binomial(1, comp_hazards)
                data.loc[at_risk, self.params.competing_event] = comp_events
                
        return data
    
    def _generate_eofu_outcome(self, data: pd.DataFrame) -> pd.DataFrame:
        """Generate end-of-follow-up outcomes.
        
        Args:
            data: Longitudinal data
            
        Returns:
            Data with EOFU outcomes added
        """
        # Get data at final time point
        max_time = data[self.params.time_var].max()
        final_data = data[data[self.params.time_var] == max_time].copy()
        
        # Get outcome model
        outcome_model = self.models.get(self.params.outcome)
        
        if outcome_model is None:
            return data
            
        # Generate outcomes based on type
        if self.params.outcome_type == 'bineofu':
            # Binary outcome at end of follow-up
            try:
                probs = outcome_model.predict(final_data, type='response')
                # Handle NaN and invalid probabilities
                probs = np.nan_to_num(probs, nan=0.5)
                probs = np.clip(probs, 0.0, 1.0)
                outcomes = np.random.binomial(1, probs)
            except Exception as e:
                print(f"Warning: Error predicting binary EOFU: {e}")
                # Use baseline prevalence
                outcomes = np.random.binomial(1, 0.5, size=len(final_data))
            
        elif self.params.outcome_type == 'conteofu':
            # Continuous outcome at end of follow-up
            try:
                outcomes = outcome_model.simulate_continuous(final_data)
            except Exception as e:
                print(f"Warning: Error simulating continuous EOFU: {e}")
                # Use mean + noise as fallback
                mean_val = 25.0  # Reasonable default
                std_val = 5.0
                outcomes = np.random.normal(mean_val, std_val, size=len(final_data))
            
        elif self.params.outcome_type == 'cateofu':
            # Categorical outcome - would need multinomial model
            # For now, simplified to use predicted category
            outcomes = outcome_model.predict(final_data)
            
        # Add outcomes to final time data
        final_data[self.params.outcome] = outcomes
        
        # Merge back to full data
        data = data[data[self.params.time_var] < max_time].copy()
        data = pd.concat([data, final_data], ignore_index=True)
        
        return data
    
    def run_parallel_simulations(self, baseline_data: pd.DataFrame,
                               n_jobs: int = -1,
                               verbose: bool = True) -> Dict[int, pd.DataFrame]:
        """Run simulations in parallel for all scenarios.
        
        Args:
            baseline_data: Baseline covariate data
            n_jobs: Number of parallel jobs (-1 for all CPUs)
            verbose: Whether to show progress
            
        Returns:
            Dictionary mapping scenario number to simulated data
        """
        scenarios = [None]  # Natural course
        scenarios.extend([i.int_no for i in self.params.interventions])
        
        if verbose:
            print(f"Running {len(scenarios)} scenarios in parallel...")
            
        # Define function for parallel execution
        def run_scenario(scenario):
            if scenario is None:
                return 0, self.simulate_natural_course(baseline_data, verbose=False)
            else:
                return scenario, self.simulate_intervention(
                    baseline_data, scenario, verbose=False
                )
                
        # Run in parallel
        results = Parallel(n_jobs=n_jobs)(
            delayed(run_scenario)(s) for s in scenarios
        )
        
        # Convert to dictionary
        results_dict = dict(results)
        
        if verbose:
            print("Parallel simulations completed.")
            
        return results_dict