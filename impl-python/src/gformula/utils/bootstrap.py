"""Bootstrap functionality for confidence intervals."""

from typing import Dict, List, Optional, Any, Tuple
import numpy as np
import pandas as pd
from tqdm import tqdm
from joblib import Parallel, delayed

from ..core.data_structures import GFormulaParams


class Bootstrapper:
    """Bootstrap confidence interval estimation for G-Formula."""
    
    def __init__(self, params: GFormulaParams, 
                 covariate_models: Dict[str, Any],
                 outcome_model: Any):
        """Initialize bootstrapper.
        
        Args:
            params: G-Formula parameters
            covariate_models: Fitted covariate models
            outcome_model: Fitted outcome model
        """
        self.params = params
        self.covariate_models = covariate_models
        self.outcome_model = outcome_model
        
    def run_bootstrap(self, n_samples: int,
                     sample_start: int = 0,
                     sample_end: int = -1,
                     n_jobs: int = -1,
                     verbose: bool = True) -> Dict[str, Any]:
        """Run bootstrap analysis.
        
        Args:
            n_samples: Number of bootstrap samples
            sample_start: Starting sample number
            sample_end: Ending sample number (-1 for n_samples)
            n_jobs: Number of parallel jobs
            verbose: Whether to show progress
            
        Returns:
            Dictionary with bootstrap results
        """
        if sample_end == -1:
            sample_end = n_samples
            
        actual_samples = sample_end - sample_start
        
        if verbose:
            print(f"Running bootstrap: samples {sample_start} to {sample_end-1}")
            
        # Run bootstrap samples
        if n_jobs == 1:
            results = []
            for i in tqdm(range(sample_start, sample_end), disable=not verbose):
                results.append(self._run_single_bootstrap(i))
        else:
            results = Parallel(n_jobs=n_jobs)(
                delayed(self._run_single_bootstrap)(i)
                for i in tqdm(range(sample_start, sample_end), disable=not verbose)
            )
            
        # Combine results
        combined_results = self._combine_bootstrap_results(results)
        
        # Calculate confidence intervals
        confidence_intervals = self._calculate_confidence_intervals(
            combined_results,
            alpha=0.05
        )
        
        return {
            'samples': combined_results,
            'confidence_intervals': confidence_intervals,
            'n_samples': actual_samples
        }
    
    def _run_single_bootstrap(self, sample_id: int) -> Dict[str, pd.DataFrame]:
        """Run a single bootstrap sample.
        
        Args:
            sample_id: Bootstrap sample ID
            
        Returns:
            Dictionary of results for this sample
        """
        # Set random seed for reproducibility
        np.random.seed(self.params.seed + sample_id if self.params.seed else sample_id)
        
        # Resample data with replacement
        n_subjects = self.params.data[self.params.id_var].nunique()
        subject_ids = self.params.data[self.params.id_var].unique()
        
        # Sample subjects with replacement
        sampled_ids = np.random.choice(subject_ids, size=n_subjects, replace=True)
        
        # Create bootstrap dataset
        bootstrap_data = []
        for new_id, old_id in enumerate(sampled_ids):
            subject_data = self.params.data[
                self.params.data[self.params.id_var] == old_id
            ].copy()
            # Assign new ID to avoid conflicts
            subject_data[self.params.id_var] = f"boot_{sample_id}_{new_id}"
            bootstrap_data.append(subject_data)
            
        bootstrap_df = pd.concat(bootstrap_data, ignore_index=True)
        
        # Create new params with bootstrap data
        bootstrap_params = GFormulaParams(
            data=bootstrap_df,
            **{k: v for k, v in self.params.__dict__.items() if k != 'data'}
        )
        
        # Import here to avoid circular import
        from ..core.gformula import GFormula
        
        # Run analysis on bootstrap sample
        gf = GFormula(**bootstrap_params.__dict__)
        
        # Suppress output for bootstrap samples
        results = gf.fit(verbose=False)
        
        # Extract key results
        sample_results = {
            'risk_differences': results.risk_differences,
            'risk_ratios': results.risk_ratios,
            'observed_risk': results.observed_risk,
            'sample_id': sample_id
        }
        
        return sample_results
    
    def _combine_bootstrap_results(self, results: List[Dict]) -> pd.DataFrame:
        """Combine results from multiple bootstrap samples.
        
        Args:
            results: List of bootstrap results
            
        Returns:
            Combined results DataFrame
        """
        # Extract risk differences for each intervention
        all_risk_diffs = []
        
        for result in results:
            sample_id = result['sample_id']
            risk_diff = result['risk_differences'].copy()
            risk_diff['sample_id'] = sample_id
            all_risk_diffs.append(risk_diff)
            
        combined = pd.concat(all_risk_diffs, ignore_index=True)
        
        return combined
    
    def _calculate_confidence_intervals(self, bootstrap_results: pd.DataFrame,
                                      alpha: float = 0.05) -> pd.DataFrame:
        """Calculate confidence intervals from bootstrap samples.
        
        Args:
            bootstrap_results: Combined bootstrap results
            alpha: Significance level
            
        Returns:
            DataFrame with confidence intervals
        """
        # Calculate percentile-based confidence intervals
        lower_pct = (alpha / 2) * 100
        upper_pct = (1 - alpha / 2) * 100
        
        # Group by intervention and time
        ci_results = []
        
        # Get unique interventions and times
        interventions = bootstrap_results['intervention'].unique()
        times = bootstrap_results[self.params.time_var].unique()
        
        for intervention in interventions:
            for time in times:
                mask = (
                    (bootstrap_results['intervention'] == intervention) &
                    (bootstrap_results[self.params.time_var] == time)
                )
                
                subset = bootstrap_results[mask]
                
                if len(subset) > 0:
                    # Calculate percentiles for risk difference
                    risk_diff_values = subset['risk_difference'].values
                    
                    ci_row = {
                        'intervention': intervention,
                        self.params.time_var: time,
                        'risk_diff_mean': np.mean(risk_diff_values),
                        'risk_diff_lower': np.percentile(risk_diff_values, lower_pct),
                        'risk_diff_upper': np.percentile(risk_diff_values, upper_pct),
                        'n_samples': len(risk_diff_values)
                    }
                    
                    # Add risk ratio CIs if available
                    if 'risk_ratio' in subset.columns:
                        rr_values = subset['risk_ratio'].values
                        ci_row.update({
                            'risk_ratio_mean': np.mean(rr_values),
                            'risk_ratio_lower': np.percentile(rr_values, lower_pct),
                            'risk_ratio_upper': np.percentile(rr_values, upper_pct)
                        })
                        
                    ci_results.append(ci_row)
                    
        return pd.DataFrame(ci_results)
    
    def combine_partial_results(self, partial_results: List[str],
                              output_path: str) -> pd.DataFrame:
        """Combine results from partial bootstrap runs.
        
        This is used when bootstrap is run in parts (e.g., samples 0-10, 11-20).
        
        Args:
            partial_results: List of paths to partial result files
            output_path: Path to save combined results
            
        Returns:
            Combined confidence intervals
        """
        all_results = []
        
        for path in partial_results:
            partial_df = pd.read_csv(path)
            all_results.append(partial_df)
            
        combined = pd.concat(all_results, ignore_index=True)
        
        # Recalculate confidence intervals from combined data
        final_cis = self._calculate_confidence_intervals(combined)
        
        # Save combined results
        final_cis.to_csv(output_path, index=False)
        
        return final_cis