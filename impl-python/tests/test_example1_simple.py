"""Simplified test for example1 to debug issues."""

import numpy as np
import pandas as pd
import sys
sys.path.append('../src')

from gformula import GFormula, CovariateSpec, InterventionSpec


def create_simple_data():
    """Create very simple test data."""
    np.random.seed(123)
    
    data_list = []
    n_subjects = 100
    n_times = 6
    
    for i in range(n_subjects):
        baseage = 40 + np.random.randint(0, 20)
        
        for t in range(n_times):
            # Simple covariates
            hbp = int(np.random.random() < 0.2)
            act = int(np.random.random() < 0.7) * 30
            
            # Simple outcome
            dia = int(np.random.random() < 0.1)
            dead = int(np.random.random() < 0.05)
            censlost = int(np.random.random() < 0.05)
            
            record = {
                'id': i,
                'time': t,
                'baseage': baseage,
                'hbp': hbp,
                'act': act,
                'dia': dia,
                'dead': dead,
                'censlost': censlost
            }
            
            data_list.append(record)
            
            # Stop if event
            if dia or dead or censlost:
                break
                
    df = pd.DataFrame(data_list)
    
    # Apply censoring
    df.loc[df['censlost'] == 1, ['dia', 'dead']] = np.nan
    df.loc[df['dead'] == 1, 'dia'] = np.nan
    
    return df


def test_simple():
    """Test with very simple data and model."""
    data = create_simple_data()
    
    print("Data shape:", data.shape)
    print("Data columns:", data.columns.tolist())
    print("Time range:", data['time'].min(), "to", data['time'].max())
    
    # Simple covariate specs - no advanced transformations
    covariates = [
        CovariateSpec(name='hbp', cov_type=2, predictor_type='lag1bin'),
        CovariateSpec(name='act', cov_type=4, predictor_type='lag1')
    ]
    
    # Simple intervention
    intervention = InterventionSpec(
        int_no=1,
        int_label='Exercise intervention',
        variables=['act'],
        int_types=[2],
        times=list(range(6)),
        min_values=[30]
    )
    
    # Run without time knots to avoid spline issues
    gf = GFormula(
        data=data,
        id_var='id',
        time_var='time',
        outcome='dia',
        outcome_type='binsurv',
        time_points=6,
        time_method='cat',  # Use categorical time to avoid splines
        covariates=covariates,
        fixed_covariates=['baseage'],
        competing_event='dead',
        censor='censlost',
        interventions=[intervention],
        n_simulations=10,  # Very small for testing
        seed=123
    )
    
    print("\nFitting model...")
    results = gf.fit(verbose=True)
    
    print("\nResults obtained!")
    print("Natural course risk:", results.counterfactual_risks[0].head())
    print("Intervention risk:", results.counterfactual_risks[1].head())


if __name__ == "__main__":
    test_simple()