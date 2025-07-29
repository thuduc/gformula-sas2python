"""Test migration of example2.sas - Multiple analyses and bootstrap."""

import pytest
import numpy as np
import pandas as pd
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'examples'))

from example2 import create_sample_data
from gformula import GFormula, CovariateSpec, InterventionSpec


def test_example2_data_creation():
    """Test data creation for different event types."""
    # Test diabetes data
    data_dia = create_sample_data(event='dia', seed=123)
    assert 'dia' in data_dia.columns
    assert 'dead' in data_dia.columns
    assert 'censlost' in data_dia.columns
    
    # Test continuous outcome data
    data_cont = create_sample_data(event='cont_e', seed=123)
    assert 'cont_e' in data_cont.columns
    # Continuous outcome should only be at time 9
    assert data_cont[data_cont['time'] < 9]['cont_e'].isna().all()
    assert data_cont[data_cont['time'] == 9]['cont_e'].notna().any()
    
    # Test binary EOFU data
    data_bin = create_sample_data(event='bin_e', seed=123)
    assert 'bin_e' in data_bin.columns
    assert data_bin[data_bin['time'] < 9]['bin_e'].isna().all()


def test_example2_binary_survival():
    """Test binary survival outcome analysis."""
    data = create_sample_data(event='dia', seed=456)
    
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
    
    # Use smaller dataset for testing
    data_subset = data[data['id'] < 100].copy()
    
    gf = GFormula(
        data=data_subset,
        id_var='id',
        time_var='time',
        outcome='dia',
        outcome_type='binsurv',
        time_points=10,
        time_method='cat',  # Use categorical time
        covariates=covariates,
        fixed_covariates=['baseage'],
        competing_event='dead',
        censor='censlost',
        interventions=interventions,
        n_simulations=10,
        n_samples=0,
        seed=123
    )
    
    results = gf.fit(verbose=False)
    
    # Check results structure
    assert results is not None
    assert 0 in results.counterfactual_risks
    assert 1 in results.counterfactual_risks
    assert len(results.risk_differences) > 0


def test_example2_continuous_outcome():
    """Test continuous EOFU outcome analysis."""
    data = create_sample_data(event='cont_e', seed=789)
    
    covariates = [
        CovariateSpec(name='hbp', cov_type=2, predictor_type='lag1bin'),
        CovariateSpec(name='bmi', cov_type=3, predictor_type='lag2cub')
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
        interventions=[],  # No interventions
        n_simulations=10,
        n_samples=0,
        seed=123
    )
    
    results = gf.fit(verbose=False)
    
    # Check results
    assert results is not None
    assert 0 in results.counterfactual_risks
    assert 'mean' in results.counterfactual_risks[0].columns
    assert 'std' in results.counterfactual_risks[0].columns


def test_example2_bootstrap_parts():
    """Test bootstrap analysis in parts."""
    data = create_sample_data(event='dia', seed=999)
    
    covariates = [
        CovariateSpec(name='hbp', cov_type=2, predictor_type='lag1bin'),
        CovariateSpec(name='bmi', cov_type=3, predictor_type='lag2cub')
    ]
    
    # Use very small dataset and few bootstrap samples
    data_subset = data[data['id'] < 50].copy()
    
    # Part 1
    gf1 = GFormula(
        data=data_subset,
        id_var='id',
        time_var='time',
        outcome='dia',
        outcome_type='binsurv',
        time_points=10,
        time_method='cat',
        covariates=covariates,
        fixed_covariates=['baseage'],
        competing_event='dead',
        censor='censlost',
        interventions=[],
        n_simulations=5,
        n_samples=5,
        sample_start=0,
        sample_end=3,
        seed=123
    )
    
    results1 = gf1.fit(verbose=False)
    
    # Part 2
    gf2 = GFormula(
        data=data_subset,
        id_var='id',
        time_var='time',
        outcome='dia',
        outcome_type='binsurv',
        time_points=10,
        time_method='cat',
        covariates=covariates,
        fixed_covariates=['baseage'],
        competing_event='dead',
        censor='censlost',
        interventions=[],
        n_simulations=5,
        n_samples=5,
        sample_start=3,
        sample_end=5,
        seed=123
    )
    
    results2 = gf2.fit(verbose=False)
    
    # Both should have results
    assert results1 is not None
    assert results2 is not None
    # Bootstrap samples should be different ranges
    assert results1.bootstrap_samples is not None
    assert results2.bootstrap_samples is not None


if __name__ == "__main__":
    pytest.main([__file__, '-v'])