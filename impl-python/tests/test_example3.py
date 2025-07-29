"""Test migration of example3.sas - Binary end-of-follow-up outcome."""

import pytest
import numpy as np
import pandas as pd
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'examples'))

from example3 import create_sample_data
from gformula import GFormula, CovariateSpec, InterventionSpec


def test_example3_data_creation():
    """Test data creation for binary EOFU outcome."""
    data = create_sample_data(seed=123)
    
    # Check data structure
    assert 'bin_e' in data.columns
    assert 'id' in data.columns
    assert 'time' in data.columns
    
    # Binary outcome should only be at time 9
    assert data[data['time'] < 9]['bin_e'].isna().all()
    
    # Check some subjects have the outcome at time 9
    final_data = data[data['time'] == 9]
    assert final_data['bin_e'].notna().any()
    
    # Check binary values
    bin_e_values = final_data['bin_e'].dropna().unique()
    assert set(bin_e_values).issubset({0.0, 1.0})


def test_example3_binary_eofu_analysis():
    """Test binary EOFU outcome analysis."""
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
        outcome='bin_e',
        outcome_type='bineofu',
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
    
    # Check that we have proportions
    for int_no, outcome_data in results.counterfactual_risks.items():
        assert 'proportion' in outcome_data.columns
        prop = outcome_data['proportion'].values[0]
        assert 0 <= prop <= 1


def test_example3_risk_differences():
    """Test risk difference calculation for binary EOFU."""
    data = create_sample_data(seed=789)
    
    covariates = [
        CovariateSpec(name='hbp', cov_type=2, predictor_type='lag1bin'),
        CovariateSpec(name='bmi', cov_type=3, predictor_type='lag2cub')
    ]
    
    interventions = [
        InterventionSpec(
            int_no=1,
            int_label='Intervention 1',
            variables=['bmi'],
            int_types=[2],
            times=list(range(10)),
            max_values=[25]
        ),
        InterventionSpec(
            int_no=2,
            int_label='Intervention 2',
            variables=['bmi'],
            int_types=[3],
            times=list(range(10)),
            change_values=[-0.1],
            probabilities=[0.5],
            conditions='(hbp == 1)'
        )
    ]
    
    # Use smaller dataset
    data_subset = data[data['id'] < 100].copy()
    
    gf = GFormula(
        data=data_subset,
        id_var='id',
        time_var='time',
        outcome='bin_e',
        outcome_type='bineofu',
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
    
    # Check risk differences
    assert len(results.risk_differences) == 2  # Two interventions
    
    # Check that differences are reasonable
    for _, row in results.risk_differences.iterrows():
        assert -1 <= row['difference'] <= 1  # Probability differences


def test_example3_censoring():
    """Test that censoring is handled correctly."""
    data = create_sample_data(seed=999)
    
    # Check censoring logic
    censored = data[data['censlost'] == 1]
    assert (censored['bin_e'].isna()).all()  # Binary outcome should be missing if censored
    
    # Check that dead subjects also have missing outcome
    dead = data[data['dead'] == 1]
    assert (dead['bin_e'].isna()).all()


if __name__ == "__main__":
    pytest.main([__file__, '-v'])