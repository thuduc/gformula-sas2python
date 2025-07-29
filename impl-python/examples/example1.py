"""
Example 1: Binary survival outcome
Python implementation of example1.sas
"""

import numpy as np
import pandas as pd
import sys
sys.path.append('../src')

from gformula import GFormula, CovariateSpec, InterventionSpec


def create_sample_data():
    """Create sample data matching example1.sas."""
    np.random.seed(5027)
    
    data_list = []
    
    for i in range(1000):
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
    
    # Apply censoring logic from SAS
    # If censored, set dia and dead to missing
    df.loc[df['censlost'] == 1, ['dia', 'dead']] = np.nan
    
    # If dead, set dia to missing  
    df.loc[df['dead'] == 1, 'dia'] = np.nan
    
    return df


def main():
    """Run example 1 analysis."""
    print("GFORMULA Example 1: Binary Survival Outcome")
    print("=" * 50)
    
    # Create sample data
    print("\nCreating sample data...")
    data = create_sample_data()
    
    # Print data summary
    print("\nData summary:")
    print(f"Number of subjects: {data['id'].nunique()}")
    print(f"Number of records: {len(data)}")
    print(f"Time range: {data['time'].min()} to {data['time'].max()}")
    
    print("\nMeans of SAMPLE data:")
    print(data.describe())
    
    # Define covariates
    print("\nDefining covariates...")
    covariates = [
        CovariateSpec(name='hbp', cov_type=2, predictor_type='tsswitch1'),
        CovariateSpec(name='act', cov_type=4, predictor_type='lag2cub')
    ]
    
    # Define intervention
    print("\nDefining intervention...")
    intervention = InterventionSpec(
        int_no=1,
        int_label='All subjects exercise at least 30 minutes per day in all intervals',
        variables=['act'],
        int_types=[2],  # threshold intervention
        times=[0, 1, 2, 3, 4, 5],
        min_values=[30],
        probabilities=[1.0]  # Apply to all subjects
    )
    
    # Run G-Formula
    print("\nRunning G-Formula analysis...")
    print("-" * 50)
    
    gf = GFormula(
        data=data,
        id_var='id',
        time_var='time',
        outcome='dia',
        outcome_type='binsurv',
        time_points=6,
        time_method='concat',
        time_knots=[1, 2, 3, 4, 5],
        covariates=covariates,
        fixed_covariates=['baseage'],
        competing_event='dead',
        competing_event_cens=0,
        censor='censlost',
        interventions=[intervention],
        n_simulations=10000,
        n_samples=0,  # No bootstrap for base example
        seed=9458,
        check_cov_models=True,
        print_cov_means=True,
        print_log_stats=True
    )
    
    # Fit the model
    results = gf.fit(verbose=True)
    
    # Print results summary
    print("\n" + "=" * 50)
    print("RESULTS SUMMARY")
    print("=" * 50)
    print(results.summary())
    
    # Save results
    print("\nSaving results...")
    
    # Save risk differences
    results.risk_differences.to_csv('example1_risk_differences.csv', index=False)
    print("Risk differences saved to: example1_risk_differences.csv")
    
    # Save survival curves
    for int_no, survival_data in results.counterfactual_risks.items():
        filename = f'example1_survival_int{int_no}.csv'
        survival_data.to_csv(filename, index=False)
        print(f"Survival data for intervention {int_no} saved to: {filename}")
    
    # Create plots if matplotlib is available
    try:
        from gformula.utils.plotting import GFormulaPlotter
        
        print("\nCreating plots...")
        plotter = GFormulaPlotter(results)
        
        # Survival curves
        fig = plotter.plot_survival_curves(save_path='example1_survival_curves.png')
        print("Survival curves saved to: example1_survival_curves.png")
        
        # Risk differences
        fig = plotter.plot_risk_differences(save_path='example1_risk_differences.png')
        print("Risk differences saved to: example1_risk_differences.png")
        
        # Full report
        plotter.create_summary_report('example1_report.pdf')
        print("Full report saved to: example1_report.pdf")
        
    except ImportError:
        print("\nMatplotlib not available - skipping plots")
    
    print("\nExample 1 completed successfully!")


if __name__ == "__main__":
    main()