"""
Example 2: Multiple analyses with bootstrap and external graph construction
Python implementation of example2.sas
"""

import numpy as np
import pandas as pd
import sys
sys.path.append('../src')

from gformula import GFormula, CovariateSpec, InterventionSpec
from gformula.utils.plotting import GFormulaPlotter
from gformula.utils.bootstrap import Bootstrapper


def create_sample_data(event='dia', seed=5027):
    """Create sample data matching example2.sas."""
    np.random.seed(seed)
    
    data_list = []
    
    for i in range(1000):
        baseage = int(35 + 25 * np.random.uniform())
        
        # Generate arrays for hbp and bmi
        ahbp = np.zeros(12)
        abmi = np.zeros(12)
        
        for j in range(12):
            ahbp[j] = int(0.4 > np.random.uniform())
            if j > 1 and ahbp[j-1] == 1:
                ahbp[j] = 1
            abmi[j] = round(25 + 5 * np.random.normal(), 3)
        
        # Generate person-time data
        for j in range(2, 12):
            time = j - 2
            
            # Current and lagged values
            hbp = ahbp[j]
            hbp_l1 = ahbp[j-1]
            hbp_l2 = ahbp[j-2]
            hbp_b = ahbp[2]
            
            bmi = abmi[j]
            bmi_l1 = abmi[j-1]
            bmi_l2 = abmi[j-2]
            bmi_b = abmi[2]
            
            # Generate outcomes
            dia = int((j / 500) > np.random.uniform())
            
            if time < 9:
                censlost = int(0.05 > np.random.uniform())
            else:
                censlost = 0
                
            if event in ['dia', 'bin_e']:
                dead = int(0.05 > np.random.uniform())
            else:
                dead = 0
                
            if time == 9:
                cont_e = round(bmi + 5 * np.random.normal(), 2)
                bin_e = int(np.random.uniform() < 0.6)
            else:
                cont_e = np.nan
                bin_e = np.nan
            
            # Store record
            record = {
                'id': i,
                'time': time,
                'baseage': baseage,
                'hbp': hbp,
                'hbp_l1': hbp_l1,
                'hbp_l2': hbp_l2,
                'hbp_b': hbp_b,
                'bmi': bmi,
                'bmi_l1': bmi_l1,
                'bmi_l2': bmi_l2,
                'bmi_b': bmi_b,
                'dia': dia,
                'censlost': censlost,
                'dead': dead,
                'cont_e': cont_e,
                'bin_e': bin_e
            }
            
            data_list.append(record)
            
            # Stop based on event type
            if event == 'dia':
                if dia or censlost or dead:
                    break
            elif event == 'cont_e':
                if censlost:
                    break
            elif event == 'bin_e':
                if dead or censlost:
                    break
    
    # Create dataframe
    df = pd.DataFrame(data_list)
    
    # Apply censoring logic
    if event == 'dia':
        df.loc[df['censlost'] == 1, ['dia', 'dead']] = np.nan
        df.loc[df['dead'] == 1, 'dia'] = np.nan
    elif event == 'cont_e':
        df.loc[df['time'] < 9, 'cont_e'] = np.nan
        df.loc[df['censlost'] == 1, 'cont_e'] = np.nan
    elif event == 'bin_e':
        df.loc[df['censlost'] == 1, ['dead', 'bin_e']] = np.nan
        df.loc[df['time'] < 9, 'bin_e'] = np.nan
        df.loc[(df['censlost'] == 1) | (df['dead'] == 1), 'bin_e'] = np.nan
    
    return df


def main():
    """Run example 2 analyses."""
    print("GFORMULA Example 2: Multiple Analyses with Bootstrap")
    print("=" * 60)
    
    # Define interventions (same for all analyses)
    interventions = [
        InterventionSpec(
            int_no=1,
            int_label='BMI Less Than 25 and No HBP',
            variables=['bmi', 'hbp'],
            int_types=[2, 1],  # threshold for BMI, static for HBP
            times=list(range(10)),
            max_values=[25, None],
            values=[None, 0]
        ),
        InterventionSpec(
            int_no=2,
            int_label='50% Chance of 10% BMI Reduction on HBP Dx',
            variables=['bmi'],
            int_types=[3],  # proportional change
            times=list(range(10)),
            change_values=[-0.1],
            probabilities=[0.5],
            conditions='(hbp == 1 and hbp_l1 == 0)'
        )
    ]
    
    # Define covariates
    covariates = [
        CovariateSpec(name='hbp', cov_type=2, predictor_type='lag1bin'),
        CovariateSpec(name='bmi', cov_type=3, predictor_type='lag2cub')
    ]
    
    # =========================================================================
    # Analysis 1: Binary Survival Outcome (diabetes)
    # =========================================================================
    print("\n" + "-" * 60)
    print("Analysis 1: Binary Survival Outcome (Diabetes)")
    print("-" * 60)
    
    # Create data for diabetes outcome
    data_dia = create_sample_data(event='dia')
    print(f"\nData shape: {data_dia.shape}")
    print(f"Number of diabetes events: {data_dia['dia'].sum()}")
    
    # Run G-Formula
    gf_dia = GFormula(
        data=data_dia,
        id_var='id',
        time_var='time',
        outcome='dia',
        outcome_type='binsurv',
        time_points=10,
        time_method='concat',
        time_knots=[1, 2, 3, 4, 5, 6, 7, 8, 9],
        covariates=covariates,
        fixed_covariates=['hbp', 'bmi', 'baseage'],
        competing_event='dead',
        competing_event_cens=0,
        censor='censlost',
        interventions=interventions,
        n_simulations=1000,
        n_samples=0,  # No bootstrap for first run
        seed=9458,
        run_graphs=False
    )
    
    print("\nFitting G-Formula model...")
    results_dia = gf_dia.fit(verbose=False)
    
    print("\nRisk at end of follow-up:")
    for int_no, risks in results_dia.counterfactual_risks.items():
        final_risk = risks[risks['time'] == 9]['cumulative_incidence'].values[0]
        if int_no == 0:
            print(f"  Natural course: {final_risk:.3f}")
        else:
            int_label = next(i.int_label for i in interventions if i.int_no == int_no)
            print(f"  {int_label}: {final_risk:.3f}")
    
    # =========================================================================
    # Analysis 2: Continuous End-of-Follow-up Outcome
    # =========================================================================
    print("\n" + "-" * 60)
    print("Analysis 2: Continuous End-of-Follow-up Outcome")
    print("-" * 60)
    
    # Create data for continuous outcome
    data_cont = create_sample_data(event='cont_e')
    print(f"\nData shape: {data_cont.shape}")
    cont_final = data_cont[data_cont['time'] == 9]['cont_e']
    print(f"Mean continuous outcome at end: {cont_final.mean():.2f} (SD: {cont_final.std():.2f})")
    
    # Run G-Formula
    gf_cont = GFormula(
        data=data_cont,
        id_var='id',
        time_var='time',
        outcome='cont_e',
        outcome_type='conteofu',
        time_points=10,
        time_method='concat',
        time_knots=[1, 2, 3, 4, 5, 6, 7, 8, 9],
        covariates=covariates,
        fixed_covariates=['hbp', 'bmi', 'baseage'],
        censor='censlost',
        interventions=[],  # No interventions for this example
        n_simulations=1000,
        n_samples=0,
        seed=9458,
        run_graphs=False
    )
    
    print("\nFitting G-Formula model...")
    results_cont = gf_cont.fit(verbose=False)
    
    print("\nMean outcome:")
    print(f"  Natural course: {results_cont.counterfactual_risks[0]['mean'].values[0]:.2f}")
    
    # =========================================================================
    # Analysis 3: Bootstrap Example (split into parts)
    # =========================================================================
    print("\n" + "-" * 60)
    print("Analysis 3: Bootstrap in Parts")
    print("-" * 60)
    
    # Create fresh data for bootstrap example
    data_boot = create_sample_data(event='dia')
    
    # Part 1: Samples 0-10
    print("\nRunning bootstrap part 1 (samples 0-10)...")
    gf_boot1 = GFormula(
        data=data_boot,
        id_var='id',
        time_var='time',
        outcome='dia',
        outcome_type='binsurv',
        time_points=10,
        time_method='concat',
        time_knots=[1, 2, 3, 4, 5, 6, 7, 8, 9],
        covariates=covariates,
        fixed_covariates=['hbp', 'bmi', 'baseage'],
        competing_event='dead',
        competing_event_cens=0,
        censor='censlost',
        interventions=interventions,
        n_simulations=1000,
        n_samples=20,
        sample_start=0,
        sample_end=10,
        seed=9458,
        run_graphs=False
    )
    
    results_boot1 = gf_boot1.fit(verbose=False)
    
    # Part 2: Samples 11-20
    print("Running bootstrap part 2 (samples 11-20)...")
    gf_boot2 = GFormula(
        data=data_boot,
        id_var='id',
        time_var='time',
        outcome='dia',
        outcome_type='binsurv',
        time_points=10,
        time_method='concat',
        time_knots=[1, 2, 3, 4, 5, 6, 7, 8, 9],
        covariates=covariates,
        fixed_covariates=['hbp', 'bmi', 'baseage'],
        competing_event='dead',
        competing_event_cens=0,
        censor='censlost',
        interventions=interventions,
        n_simulations=1000,
        n_samples=20,
        sample_start=11,
        sample_end=20,
        seed=9458,
        run_graphs=False
    )
    
    results_boot2 = gf_boot2.fit(verbose=False)
    
    print("\nBootstrap results summary:")
    print("Part 1 completed: samples 0-10")
    print("Part 2 completed: samples 11-20")
    
    # In practice, you would combine the bootstrap results here
    # using the Bootstrapper.combine_partial_results method
    
    # =========================================================================
    # Create Plots
    # =========================================================================
    print("\n" + "-" * 60)
    print("Creating plots...")
    print("-" * 60)
    
    # Plot for diabetes analysis
    if results_dia is not None:
        plotter = GFormulaPlotter(results_dia)
        
        # Survival curves
        fig = plotter.plot_survival_curves(save_path='example2_survival_curves.png')
        print("Survival curves saved to: example2_survival_curves.png")
        
        # Risk differences
        fig = plotter.plot_risk_differences(save_path='example2_risk_differences.png')
        print("Risk differences saved to: example2_risk_differences.png")
        
        # Covariate means
        fig = plotter.plot_covariate_means('bmi', save_path='example2_bmi_means.png')
        print("BMI means plot saved to: example2_bmi_means.png")
        
        # Full report
        plotter.create_summary_report('example2_report.pdf')
        print("Full report saved to: example2_report.pdf")
    
    print("\n" + "=" * 60)
    print("Example 2 completed successfully!")
    print("=" * 60)


if __name__ == "__main__":
    main()