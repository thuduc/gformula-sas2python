"""Test migration of example4.sas - Continuous end-of-follow-up outcome."""

import pytest
import numpy as np
import pandas as pd
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'examples'))

from example4 import create_sample_data
from gformula import GFormula, CovariateSpec, InterventionSpec


def test_example4_data_creation():
    """Test data creation for continuous EOFU outcome."""
    data = create_sample_data(seed=123)
    
    # Check data structure
    assert 'cont_e' in data.columns
    assert 'id' in data.columns
    assert 'time' in data.columns
    
    # Continuous outcome should only be at time 9
    assert data[data['time'] < 9]['cont_e'].isna().all()
    
    # Check some subjects have the outcome at time 9
    final_data = data[data['time'] == 9]
    assert final_data['cont_e'].notna().any()
    
    # Check that it's continuous (has decimal values)
    cont_e_values = final_data['cont_e'].dropna()
    assert len(cont_e_values.unique()) > 10  # Many unique values
    assert cont_e_values.dtype == float


def test_example4_continuous_eofu_analysis():
    """Test continuous EOFU outcome analysis."""
    data = create_sample_data(seed=456)
    
    covariates = [
        CovariateSpec(name='hbp', cov_type=2, predictor_type='lag1bin'),
        CovariateSpec(name='bmi', cov_type=3, predictor_type='lag2cub')
    ]
    
    interventions = [
        InterventionSpec(
            int_no=1,
            int_label='BMI<25 & No HBP',
            variables=['bmi', 'hbp'],
            int_types=[2, 1],
            times=list(range(10)),
            max_values=[25, None],
            values=[None, 0]
        )
    ]
    
    # Use smaller dataset
    data_subset = data[data['id'] < 100].copy()
    
    gf = GFormula(
        data=data_subset,
        id_var='id',
        time_var='time',
        outcome='cont_e',
        outcome_type='conteofu',
        time_points=10,
        time_method='cat',
        covariates=covariates,
        fixed_covariates=['baseage'],
        censor='censlost',
        interventions=interventions,
        n_simulations=10,
        n_samples=0,
        seed=123
    )
    
    results = gf.fit(verbose=False)
    
    # Check results
    assert results is not None
    assert 0 in results.counterfactual_risks
    assert 1 in results.counterfactual_risks
    
    # Check that we have mean and std
    for int_no, outcome_data in results.counterfactual_risks.items():
        assert 'mean' in outcome_data.columns
        assert 'std' in outcome_data.columns
        # Mean should be reasonable (around BMI values)
        mean_val = outcome_data['mean'].values[0]
        assert 10 < mean_val < 50


def test_example4_mean_differences():
    """Test mean difference calculation for continuous EOFU."""
    data = create_sample_data(seed=789)
    
    covariates = [
        CovariateSpec(name='hbp', cov_type=2, predictor_type='lag1bin'),
        CovariateSpec(name='bmi', cov_type=3, predictor_type='lag2cub')
    ]
    
    interventions = [
        InterventionSpec(
            int_no=1,
            int_label='BMI reduction',
            variables=['bmi'],
            int_types=[2],
            times=list(range(10)),
            max_values=[23]  # Lower threshold
        ),
        InterventionSpec(
            int_no=2,
            int_label='Proportional reduction',
            variables=['bmi'],
            int_types=[3],
            times=list(range(10)),
            change_values=[-0.1],  # 10% reduction
            probabilities=[1.0]
        )
    ]
    
    # Use smaller dataset
    data_subset = data[data['id'] < 100].copy()
    
    gf = GFormula(
        data=data_subset,
        id_var='id',
        time_var='time',
        outcome='cont_e',
        outcome_type='conteofu',
        time_points=10,
        time_method='cat',
        covariates=covariates,
        fixed_covariates=['baseage'],
        censor='censlost',
        interventions=interventions,
        n_simulations=10,
        n_samples=0,
        seed=123
    )
    
    results = gf.fit(verbose=False)
    
    # Check mean differences
    assert len(results.risk_differences) == 2  # Two interventions
    
    # BMI reduction interventions should reduce the mean outcome
    for _, row in results.risk_differences.iterrows():
        # Difference could be negative (reduction)
        assert -10 < row['difference'] < 10  # Reasonable range


def test_example4_no_competing_events():
    """Test that continuous EOFU has no competing events."""
    data = create_sample_data(seed=999)
    
    # Check that dead column is all NaN for continuous outcome
    assert data['dead'].isna().all()
    
    # Only censoring should affect the outcome
    censored = data[data['censlost'] == 1]
    assert (censored['cont_e'].isna()).all()
    
    # Uncensored subjects at time 9 should have outcome
    final_uncensored = data[(data['time'] == 9) & (data['censlost'] == 0)]
    assert final_uncensored['cont_e'].notna().all()


def test_example4_bootstrap_consistency():
    """Test that bootstrap parts can be run consistently."""
    data = create_sample_data(seed=111)
    
    covariates = [
        CovariateSpec(name='hbp', cov_type=2, predictor_type='lag1bin'),
        CovariateSpec(name='bmi', cov_type=3, predictor_type='lag2cub')
    ]
    
    # Use very small dataset
    data_subset = data[data['id'] < 50].copy()
    
    # Common parameters
    common_params = dict(
        data=data_subset,
        id_var='id',
        time_var='time',
        outcome='cont_e',
        outcome_type='conteofu',
        time_points=10,
        time_method='cat',
        covariates=covariates,
        fixed_covariates=['baseage'],
        censor='censlost',
        interventions=[],
        n_simulations=5,
        seed=123
    )
    
    # Run without bootstrap
    gf0 = GFormula(**common_params, n_samples=0)
    results0 = gf0.fit(verbose=False)
    
    # Run with bootstrap
    gf1 = GFormula(**common_params, n_samples=3, sample_start=0, sample_end=3)
    results1 = gf1.fit(verbose=False)
    
    # Both should produce results
    assert results0 is not None
    assert results1 is not None
    
    # Bootstrap should have confidence intervals
    assert results1.confidence_intervals is not None


if __name__ == "__main__":
    pytest.main([__file__, '-v'])