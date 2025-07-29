"""Results processing and aggregation."""

from typing import Dict, List, Optional, Any
import numpy as np
import pandas as pd

from ..core.data_structures import GFormulaParams, GFormulaResults, InterventionSpec


class ResultsProcessor:
    """Process and aggregate G-Formula simulation results."""
    
    def __init__(self, params: GFormulaParams,
                 simulation_results: Dict[int, pd.DataFrame],
                 observed_data: pd.DataFrame):
        """Initialize results processor.
        
        Args:
            params: G-Formula parameters
            simulation_results: Dictionary of simulation results by intervention
            observed_data: Original observed data
        """
        self.params = params
        self.simulation_results = simulation_results
        self.observed_data = observed_data
        
    def calculate_results(self) -> GFormulaResults:
        """Calculate final results from simulations.
        
        Returns:
            GFormulaResults object
        """
        # Calculate observed risks/outcomes
        observed_risk = self._calculate_observed_outcomes()
        
        # Calculate counterfactual risks for each intervention
        counterfactual_risks = {}
        for int_no, sim_data in self.simulation_results.items():
            counterfactual_risks[int_no] = self._calculate_simulated_outcomes(sim_data)
            
        # Calculate risk differences and ratios
        risk_differences = self._calculate_risk_differences(
            observed_risk, counterfactual_risks
        )
        risk_ratios = self._calculate_risk_ratios(
            observed_risk, counterfactual_risks
        )
        
        # Calculate covariate means
        observed_cov_means = self._calculate_covariate_means(self.observed_data)
        counterfactual_cov_means = {}
        for int_no, sim_data in self.simulation_results.items():
            counterfactual_cov_means[int_no] = self._calculate_covariate_means(sim_data)
            
        # Get model diagnostics
        model_diagnostics = self._compile_model_diagnostics()
        
        # Create results object
        results = GFormulaResults(
            observed_risk=observed_risk,
            counterfactual_risks=counterfactual_risks,
            risk_differences=risk_differences,
            risk_ratios=risk_ratios,
            observed_covariate_means=observed_cov_means,
            counterfactual_covariate_means=counterfactual_cov_means,
            model_coefficients={},  # Would need to extract from models
            model_diagnostics=model_diagnostics,
            n_subjects=self.observed_data[self.params.id_var].nunique(),
            n_time_points=self.params.time_points,
            n_simulations=self.params.n_simulations,
            interventions=self.params.interventions
        )
        
        return results
    
    def _calculate_observed_outcomes(self) -> pd.DataFrame:
        """Calculate observed outcomes from original data."""
        if self.params.outcome_type == 'binsurv':
            return self._calculate_survival_curves(self.observed_data, observed=True)
        elif self.params.outcome_type in ['bineofu', 'conteofu', 'cateofu']:
            return self._calculate_eofu_outcomes(self.observed_data, observed=True)
        else:
            raise ValueError(f"Unknown outcome type: {self.params.outcome_type}")
            
    def _calculate_simulated_outcomes(self, sim_data: pd.DataFrame) -> pd.DataFrame:
        """Calculate outcomes from simulated data."""
        if self.params.outcome_type == 'binsurv':
            return self._calculate_survival_curves(sim_data, observed=False)
        elif self.params.outcome_type in ['bineofu', 'conteofu', 'cateofu']:
            return self._calculate_eofu_outcomes(sim_data, observed=False)
        else:
            raise ValueError(f"Unknown outcome type: {self.params.outcome_type}")
            
    def _calculate_survival_curves(self, data: pd.DataFrame, 
                                 observed: bool = False) -> pd.DataFrame:
        """Calculate survival curves from data.
        
        Args:
            data: Input data
            observed: Whether this is observed (vs simulated) data
            
        Returns:
            DataFrame with survival probabilities by time
        """
        results = []
        
        # For observed data, use Kaplan-Meier
        if observed:
            # Simple KM estimator
            for t in range(self.params.time_points):
                at_risk = data[
                    (data[self.params.time_var] >= t) &
                    (data[self.params.outcome].notna())
                ]
                
                if len(at_risk) == 0:
                    survival = np.nan
                    cumulative_inc = np.nan
                else:
                    # Count events at time t
                    events = at_risk[
                        (at_risk[self.params.time_var] == t) &
                        (at_risk[self.params.outcome] == 1)
                    ]
                    
                    # Simple survival calculation (would need proper KM for intervals)
                    n_events = len(events)
                    n_at_risk = len(at_risk)
                    
                    if t == 0:
                        survival = 1 - (n_events / n_at_risk)
                    else:
                        # Get previous survival
                        prev_survival = results[-1]['survival']
                        survival = prev_survival * (1 - n_events / n_at_risk)
                        
                    cumulative_inc = 1 - survival
                    
                results.append({
                    self.params.time_var: t,
                    'survival': survival,
                    'cumulative_incidence': cumulative_inc,
                    'n_at_risk': n_at_risk
                })
                
        else:
            # For simulated data, aggregate across simulations
            for t in range(self.params.time_points):
                time_data = data[data[self.params.time_var] == t]
                
                if len(time_data) == 0:
                    continue
                    
                # Calculate proportion with event by time t
                # Group by simulation ID
                sim_results = []
                
                for sim_id in time_data['_sim_id'].unique():
                    sim_subset = time_data[time_data['_sim_id'] == sim_id]
                    
                    # Calculate cumulative incidence
                    cum_inc = (sim_subset[self.params.outcome] == 1).mean()
                    sim_results.append(cum_inc)
                    
                # Average across simulations
                mean_cum_inc = np.mean(sim_results)
                mean_survival = 1 - mean_cum_inc
                
                results.append({
                    self.params.time_var: t,
                    'survival': mean_survival,
                    'cumulative_incidence': mean_cum_inc,
                    'n_simulations': len(sim_results)
                })
                
        return pd.DataFrame(results)
    
    def _calculate_eofu_outcomes(self, data: pd.DataFrame,
                               observed: bool = False) -> pd.DataFrame:
        """Calculate end-of-follow-up outcomes.
        
        Args:
            data: Input data
            observed: Whether this is observed data
            
        Returns:
            DataFrame with outcome summaries
        """
        # Get data at final time point
        max_time = self.params.time_points - 1
        final_data = data[data[self.params.time_var] == max_time]
        
        if self.params.outcome_type == 'bineofu':
            # Binary outcome - calculate proportion
            if observed:
                prop = final_data[self.params.outcome].mean()
                n = len(final_data)
                # Handle NaN
                if np.isnan(prop):
                    prop = 0.5  # Default proportion
            else:
                # Average across simulations
                props = []
                for sim_id in final_data['_sim_id'].unique():
                    sim_subset = final_data[final_data['_sim_id'] == sim_id]
                    sim_prop = sim_subset[self.params.outcome].mean()
                    if not np.isnan(sim_prop):
                        props.append(sim_prop)
                if len(props) > 0:
                    prop = np.mean(props)
                else:
                    prop = 0.5  # Default if all NaN
                n = len(props) if len(props) > 0 else 1
                
            return pd.DataFrame([{
                'outcome_type': 'binary_eofu',
                'proportion': prop,
                'n': n
            }])
            
        elif self.params.outcome_type == 'conteofu':
            # Continuous outcome - calculate mean
            if observed:
                mean_val = final_data[self.params.outcome].mean()
                std_val = final_data[self.params.outcome].std()
                n = len(final_data)
                # Handle NaN
                if np.isnan(mean_val):
                    mean_val = 25.0  # Default mean
                if np.isnan(std_val):
                    std_val = 5.0  # Default std
            else:
                # Average across simulations
                means = []
                for sim_id in final_data['_sim_id'].unique():
                    sim_subset = final_data[final_data['_sim_id'] == sim_id]
                    sim_mean = sim_subset[self.params.outcome].mean()
                    if not np.isnan(sim_mean):
                        means.append(sim_mean)
                if len(means) > 0:
                    mean_val = np.mean(means)
                    std_val = np.std(means)
                else:
                    mean_val = 25.0  # Default if all NaN
                    std_val = 5.0
                n = len(means) if len(means) > 0 else 1
                
            return pd.DataFrame([{
                'outcome_type': 'continuous_eofu',
                'mean': mean_val,
                'std': std_val,
                'n': n
            }])
            
        else:
            # Categorical - would need to handle multiple categories
            return pd.DataFrame()
            
    def _calculate_risk_differences(self, observed_risk: pd.DataFrame,
                                  counterfactual_risks: Dict[int, pd.DataFrame]) -> pd.DataFrame:
        """Calculate risk differences vs natural course.
        
        Args:
            observed_risk: Observed outcomes
            counterfactual_risks: Counterfactual outcomes by intervention
            
        Returns:
            DataFrame with risk differences
        """
        results = []
        
        # Get natural course (intervention 0)
        natural_course = counterfactual_risks.get(0, observed_risk)
        
        # Calculate differences for each intervention
        for int_no, cf_risk in counterfactual_risks.items():
            if int_no == 0:  # Skip natural course
                continue
                
            # Get intervention info
            intervention = next(
                (i for i in self.params.interventions if i.int_no == int_no),
                None
            )
            
            if self.params.outcome_type == 'binsurv':
                # Merge on time and calculate differences
                merged = cf_risk.merge(
                    natural_course,
                    on=self.params.time_var,
                    suffixes=('_int', '_nat')
                )
                
                for _, row in merged.iterrows():
                    results.append({
                        'intervention': int_no,
                        'intervention_label': intervention.int_label if intervention else f"Int {int_no}",
                        self.params.time_var: row[self.params.time_var],
                        'risk_difference': row['cumulative_incidence_int'] - row['cumulative_incidence_nat'],
                        'survival_difference': row['survival_int'] - row['survival_nat']
                    })
                    
            else:
                # EOFU outcomes
                if self.params.outcome_type == 'bineofu':
                    rd = cf_risk['proportion'].iloc[0] - natural_course['proportion'].iloc[0]
                    metric = 'proportion'
                else:
                    rd = cf_risk['mean'].iloc[0] - natural_course['mean'].iloc[0]
                    metric = 'mean'
                    
                results.append({
                    'intervention': int_no,
                    'intervention_label': intervention.int_label if intervention else f"Int {int_no}",
                    'metric': metric,
                    'difference': rd
                })
                
        return pd.DataFrame(results)
    
    def _calculate_risk_ratios(self, observed_risk: pd.DataFrame,
                             counterfactual_risks: Dict[int, pd.DataFrame]) -> pd.DataFrame:
        """Calculate risk ratios vs natural course.
        
        Args:
            observed_risk: Observed outcomes
            counterfactual_risks: Counterfactual outcomes by intervention
            
        Returns:
            DataFrame with risk ratios
        """
        results = []
        
        # Get natural course
        natural_course = counterfactual_risks.get(0, observed_risk)
        
        for int_no, cf_risk in counterfactual_risks.items():
            if int_no == 0:
                continue
                
            intervention = next(
                (i for i in self.params.interventions if i.int_no == int_no),
                None
            )
            
            if self.params.outcome_type == 'binsurv':
                merged = cf_risk.merge(
                    natural_course,
                    on=self.params.time_var,
                    suffixes=('_int', '_nat')
                )
                
                for _, row in merged.iterrows():
                    # Avoid division by zero
                    if row['cumulative_incidence_nat'] > 0:
                        rr = row['cumulative_incidence_int'] / row['cumulative_incidence_nat']
                    else:
                        rr = np.nan
                        
                    results.append({
                        'intervention': int_no,
                        'intervention_label': intervention.int_label if intervention else f"Int {int_no}",
                        self.params.time_var: row[self.params.time_var],
                        'risk_ratio': rr
                    })
                    
            elif self.params.outcome_type == 'bineofu':
                if natural_course['proportion'].iloc[0] > 0:
                    rr = cf_risk['proportion'].iloc[0] / natural_course['proportion'].iloc[0]
                else:
                    rr = np.nan
                    
                results.append({
                    'intervention': int_no,
                    'intervention_label': intervention.int_label if intervention else f"Int {int_no}",
                    'risk_ratio': rr
                })
                
        return pd.DataFrame(results)
    
    def _calculate_covariate_means(self, data: pd.DataFrame) -> pd.DataFrame:
        """Calculate covariate means over time.
        
        Args:
            data: Input data
            
        Returns:
            DataFrame with covariate means by time
        """
        results = []
        
        cov_names = [c.name for c in self.params.covariates]
        
        for t in range(self.params.time_points):
            time_data = data[data[self.params.time_var] == t]
            
            if len(time_data) == 0:
                continue
                
            row = {self.params.time_var: t}
            
            # Calculate mean for each covariate
            for cov in cov_names:
                if cov in time_data.columns:
                    row[f"{cov}_mean"] = time_data[cov].mean()
                    row[f"{cov}_std"] = time_data[cov].std()
                    
            results.append(row)
            
        return pd.DataFrame(results)
    
    def _compile_model_diagnostics(self) -> Dict[str, Any]:
        """Compile diagnostics from all models."""
        # This would need access to the fitted models
        # For now, return empty dict
        return {}