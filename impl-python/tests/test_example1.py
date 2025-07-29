"""Test migration of example1.sas - Binary survival outcome."""

import pytest
import numpy as np
import pandas as pd
from gformula import GFormula, CovariateSpec, InterventionSpec


def create_example1_data(n_subjects=100):
    """Create sample data matching example1.sas."""
    np.random.seed(5027)
    
    data_list = []
    
    for i in range(n_subjects):
        baseage = int(35 + 25 * np.random.uniform())
        
        # Generate arrays for hbp and act
        ahbp = np.zeros(8)
        aact = np.zeros(8)
        
        for j in range(8):
            ahbp[j] = int(0.2 > np.random.uniform())
            if j > 0 and ahbp[j-1] == 1:
                ahbp[j] = 1
                
            if 0.7 > np.random.uniform():
                aact[j] = int(np.exp(3.5 + 0.4 * np.random.normal()))
            else:
                aact[j] = 0
                
        # Generate person-time data
        for j in range(2, 8):
            time = j - 2
            
            # Current and lagged values
            hbp = ahbp[j]
            hbp_l1 = ahbp[j-1]
            hbp_l2 = ahbp[j-2]
            
            act = aact[j]
            act_l1 = aact[j-1]
            act_l2 = aact[j-2]
            
            # Generate outcomes
            dia = int((j / 500) > np.random.uniform())
            censlost = int(0.05 > np.random.uniform())
            dead = int(0.05 > np.random.uniform())
            
            # Store record
            record = {
                'id': i,
                'time': time,
                'baseage': baseage,
                'hbp': hbp,
                'hbp_l1': hbp_l1,
                'hbp_l2': hbp_l2,
                'act': act,
                'act_l1': act_l1, 
                'act_l2': act_l2,
                'dia': dia,
                'censlost': censlost,
                'dead': dead
            }
            
            data_list.append(record)
            
            # Stop if event occurs
            if dia or censlost or dead:
                break
                
    # Create dataframe
    df = pd.DataFrame(data_list)
    
    # Apply censoring logic
    # If censored, set dia and dead to missing
    df.loc[df['censlost'] == 1, ['dia', 'dead']] = np.nan
    
    # If dead, set dia to missing
    df.loc[df['dead'] == 1, 'dia'] = np.nan
    
    return df


def test_example1_basic():
    """Test basic example1 setup and execution."""
    # Create data
    data = create_example1_data()
    
    # Verify data structure
    assert 'id' in data.columns
    assert 'time' in data.columns
    assert 'dia' in data.columns
    assert data['time'].min() == 0
    assert data['time'].max() <= 5
    
    # Define covariates
    covariates = [
        CovariateSpec(name='hbp', cov_type=2, predictor_type='tsswitch1'),
        CovariateSpec(name='act', cov_type=4, predictor_type='lag2cub')
    ]
    
    # Define intervention - all subjects exercise at least 30 min/day
    intervention = InterventionSpec(
        int_no=1,
        int_label='All subjects exercise at least 30 minutes per day in all intervals',
        variables=['act'],
        int_types=[2],  # threshold
        times=list(range(6)),
        min_values=[30]
    )
    
    # Create and run G-Formula
    gf = GFormula(
        data=data,
        id_var='id',
        time_var='time',
        outcome='dia',
        outcome_type='binsurv',
        time_points=6,
        time_method='cat',  # Use categorical time to avoid spline issues
        covariates=covariates,
        fixed_covariates=['baseage'],
        competing_event='dead',
        competing_event_cens=0,
        censor='censlost',
        interventions=[intervention],
        n_simulations=10,  # Reduced for testing
        seed=9458
    )
    
    # Fit the model
    results = gf.fit(verbose=False)
    
    # Basic checks
    assert results is not None
    assert results.n_subjects == data['id'].nunique()
    assert results.n_time_points == 6
    assert len(results.interventions) == 1
    
    # Check that we have results for natural course and intervention
    assert 0 in results.counterfactual_risks  # Natural course
    assert 1 in results.counterfactual_risks  # Intervention
    
    # Check risk differences
    assert len(results.risk_differences) > 0
    assert 'intervention' in results.risk_differences.columns
    assert 'risk_difference' in results.risk_differences.columns


def test_example1_hazard_ratio():
    """Test hazard ratio calculation as in example1.sas."""
    # Create data
    data = create_example1_data()
    
    # Define covariates
    covariates = [
        CovariateSpec(name='hbp', cov_type=2, predictor_type='tsswitch1'),
        CovariateSpec(name='act', cov_type=4, predictor_type='lag2cub')
    ]
    
    # Define intervention
    intervention = InterventionSpec(
        int_no=1,
        int_label='All subjects exercise at least 30 minutes per day in all intervals',
        variables=['act'],
        int_types=[2],
        times=list(range(6)),
        min_values=[30]
    )
    
    # Run with hazard ratio comparison (intervention 0 vs 1)
    gf = GFormula(
        data=data,
        id_var='id',
        time_var='time',
        outcome='dia',
        outcome_type='binsurv',
        time_points=6,
        time_method='cat',  # Use categorical time to avoid spline issues
        covariates=covariates,
        fixed_covariates=['baseage'],
        competing_event='dead',
        competing_event_cens=0,
        censor='censlost',
        interventions=[intervention],
        n_simulations=10,
        seed=9458
    )
    
    results = gf.fit(verbose=False)
    
    # Check risk ratios
    assert len(results.risk_ratios) > 0
    
    # Risk ratios should be calculated for each time point
    rr_data = results.risk_ratios[results.risk_ratios['intervention'] == 1]
    assert len(rr_data) > 0
    
    # Check that risk ratios are reasonable (between 0 and 5)
    assert all((rr_data['risk_ratio'] >= 0) & (rr_data['risk_ratio'] <= 5))


def test_example1_no_samples():
    """Test example1 without bootstrap samples."""
    data = create_example1_data()
    
    covariates = [
        CovariateSpec(name='hbp', cov_type=2, predictor_type='tsswitch1'),
        CovariateSpec(name='act', cov_type=4, predictor_type='lag2cub')
    ]
    
    intervention = InterventionSpec(
        int_no=1,
        int_label='All subjects exercise at least 30 minutes per day in all intervals',
        variables=['act'],
        int_types=[2],
        times=list(range(6)),
        min_values=[30]
    )
    
    # Run without bootstrap
    gf = GFormula(
        data=data,
        id_var='id',
        time_var='time',
        outcome='dia',
        outcome_type='binsurv',
        time_points=6,
        time_method='cat',  # Use categorical time to avoid spline issues
        covariates=covariates,
        fixed_covariates=['baseage'],
        competing_event='dead',
        competing_event_cens=0,
        censor='censlost',
        interventions=[intervention],
        n_simulations=10,
        n_samples=0,  # No bootstrap
        seed=9458
    )
    
    results = gf.fit(verbose=False)
    
    # Should not have confidence intervals
    assert results.confidence_intervals is None
    assert results.bootstrap_samples is None


@pytest.mark.slow
def test_example1_with_bootstrap():
    """Test example1 with bootstrap confidence intervals."""
    data = create_example1_data()
    
    covariates = [
        CovariateSpec(name='hbp', cov_type=2, predictor_type='tsswitch1'),
        CovariateSpec(name='act', cov_type=4, predictor_type='lag2cub')
    ]
    
    intervention = InterventionSpec(
        int_no=1,
        int_label='All subjects exercise at least 30 minutes per day in all intervals',
        variables=['act'],
        int_types=[2],
        times=list(range(6)),
        min_values=[30]
    )
    
    # Run with small number of bootstrap samples
    gf = GFormula(
        data=data,
        id_var='id',
        time_var='time',
        outcome='dia',
        outcome_type='binsurv',
        time_points=6,
        time_method='cat',  # Use categorical time to avoid spline issues
        covariates=covariates,
        fixed_covariates=['baseage'],
        competing_event='dead',
        competing_event_cens=0,
        censor='censlost',
        interventions=[intervention],
        n_simulations=5,  # Reduced for testing
        n_samples=5,  # Small bootstrap
        seed=9458
    )
    
    results = gf.fit(verbose=False)
    
    # Should have confidence intervals
    assert results.confidence_intervals is not None
    assert len(results.confidence_intervals) > 0
    assert 'risk_diff_lower' in results.confidence_intervals.columns
    assert 'risk_diff_upper' in results.confidence_intervals.columns
    
    # CIs should contain point estimates
    for _, row in results.confidence_intervals.iterrows():
        assert row['risk_diff_lower'] <= row['risk_diff_mean']
        assert row['risk_diff_mean'] <= row['risk_diff_upper']


if __name__ == "__main__":
    # Run basic test
    test_example1_basic()
    print("Example 1 basic test passed!")
    
    # Run hazard ratio test
    test_example1_hazard_ratio()
    print("Example 1 hazard ratio test passed!")
    
    # Run no samples test
    test_example1_no_samples()
    print("Example 1 no samples test passed!")
    
    print("\nAll tests passed!")