"""
Example 3: Binary end-of-follow-up outcome
Python implementation of example3.sas
"""

import numpy as np
import pandas as pd
import sys
sys.path.append('../src')

from gformula import GFormula, CovariateSpec, InterventionSpec
from gformula.utils.plotting import GFormulaPlotter


def create_sample_data(seed=5027):
    """Create sample data for binary end-of-follow-up outcome."""
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
                
            dead = int(0.05 > np.random.uniform())
            
            # Binary outcome only at end of follow-up
            if time == 9:
                bin_e = int(np.random.uniform() < 0.6)
            else:
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
                'bin_e': bin_e
            }
            
            data_list.append(record)
            
            # Stop if censored or dead
            if dead or censlost:
                break
    
    # Create dataframe
    df = pd.DataFrame(data_list)
    
    # Apply censoring logic for binary EOFU outcome
    df.loc[df['censlost'] == 1, ['dead', 'bin_e']] = np.nan
    df.loc[df['time'] < 9, 'bin_e'] = np.nan
    df.loc[(df['censlost'] == 1) | (df['dead'] == 1), 'bin_e'] = np.nan
    
    return df


def main():
    """Run example 3 analysis."""
    print("GFORMULA Example 3: Binary End-of-Follow-up Outcome")
    print("=" * 60)
    
    # Create sample data
    print("\nCreating sample data...")
    data = create_sample_data()
    
    # Print data summary
    print(f"\nData summary:")
    print(f"Number of subjects: {data['id'].nunique()}")
    print(f"Number of records: {len(data)}")
    print(f"Time range: {data['time'].min()} to {data['time'].max()}")
    
    # Check binary outcome at end
    final_data = data[data['time'] == 9]
    bin_e_counts = final_data['bin_e'].value_counts()
    print(f"\nBinary outcome at end of follow-up:")
    print(f"  Outcome = 0: {bin_e_counts.get(0.0, 0)} subjects")
    print(f"  Outcome = 1: {bin_e_counts.get(1.0, 0)} subjects")
    print(f"  Missing: {final_data['bin_e'].isna().sum()} subjects")
    
    # Define interventions
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
    # Run initial analysis without bootstrap
    # =========================================================================
    print("\n" + "-" * 60)
    print("Running G-Formula analysis (no bootstrap)...")
    print("-" * 60)
    
    gf = GFormula(
        data=data,
        id_var='id',
        time_var='time',
        outcome='bin_e',
        outcome_type='bineofu',
        time_points=10,
        time_method='concat',
        time_knots=[1, 2, 3, 4, 5, 6, 7, 8, 9],
        covariates=covariates,
        fixed_covariates=['hbp', 'bmi', 'baseage'],
        censor='censlost',
        interventions=interventions,
        n_simulations=1000,
        n_samples=0,  # No bootstrap
        seed=9458,
        check_cov_models=True,
        print_cov_means=False,
        save_raw_covmean=True
    )
    
    print("\nFitting models...")
    results = gf.fit(verbose=True)
    
    # Print results
    print("\n" + "-" * 60)
    print("RESULTS")
    print("-" * 60)
    
    print("\nProbability of binary outcome at end of follow-up:")
    for int_no, outcome_data in results.counterfactual_risks.items():
        prob = outcome_data['proportion'].values[0]
        if int_no == 0:
            print(f"  Natural course: {prob:.3f}")
        else:
            intervention = next(i for i in interventions if i.int_no == int_no)
            print(f"  {intervention.int_label}: {prob:.3f}")
    
    print("\nRisk differences vs natural course:")
    for _, row in results.risk_differences.iterrows():
        print(f"  {row['intervention_label']}: {row['difference']:.3f}")
    
    # =========================================================================
    # Run bootstrap analysis in parts
    # =========================================================================
    print("\n" + "-" * 60)
    print("Running bootstrap analysis...")
    print("-" * 60)
    
    # Part 1: Samples 0-10
    print("\nBootstrap part 1 (samples 0-10)...")
    gf_boot1 = GFormula(
        data=data,
        id_var='id',
        time_var='time',
        outcome='bin_e',
        outcome_type='bineofu',
        time_points=10,
        time_method='concat',
        time_knots=[1, 2, 3, 4, 5, 6, 7, 8, 9],
        covariates=covariates,
        fixed_covariates=['hbp', 'bmi', 'baseage'],
        censor='censlost',
        interventions=interventions,
        n_simulations=1000,
        n_samples=20,
        sample_start=0,
        sample_end=10,
        seed=9458
    )
    
    results_boot1 = gf_boot1.fit(verbose=False)
    
    # Part 2: Samples 11-20
    print("Bootstrap part 2 (samples 11-20)...")
    gf_boot2 = GFormula(
        data=data,
        id_var='id',
        time_var='time',
        outcome='bin_e',
        outcome_type='bineofu',
        time_points=10,
        time_method='concat',
        time_knots=[1, 2, 3, 4, 5, 6, 7, 8, 9],
        covariates=covariates,
        fixed_covariates=['hbp', 'bmi', 'baseage'],
        censor='censlost',
        interventions=interventions,
        n_simulations=1000,
        n_samples=20,
        sample_start=11,
        sample_end=20,
        seed=9458
    )
    
    results_boot2 = gf_boot2.fit(verbose=False)
    
    print("\nBootstrap completed:")
    print("  Part 1: samples 0-10 ✓")
    print("  Part 2: samples 11-20 ✓")
    
    # Print confidence intervals if available
    if results_boot2.confidence_intervals is not None:
        print("\nConfidence intervals for risk differences:")
        for _, row in results_boot2.confidence_intervals.iterrows():
            if row['intervention'] != 0:
                print(f"  Intervention {row['intervention']}: "
                      f"{row['risk_diff_mean']:.3f} "
                      f"({row['risk_diff_lower']:.3f}, {row['risk_diff_upper']:.3f})")
    
    # =========================================================================
    # Create plots
    # =========================================================================
    print("\n" + "-" * 60)
    print("Creating visualizations...")
    print("-" * 60)
    
    # Since this is a binary EOFU outcome, we'll create custom plots
    import matplotlib.pyplot as plt
    
    # Bar plot of probabilities
    fig, ax = plt.subplots(figsize=(10, 6))
    
    probabilities = []
    labels = []
    
    for int_no, outcome_data in results.counterfactual_risks.items():
        prob = outcome_data['proportion'].values[0]
        probabilities.append(prob)
        
        if int_no == 0:
            labels.append('Natural\nCourse')
        else:
            intervention = next(i for i in interventions if i.int_no == int_no)
            # Shorten label for plot
            if 'BMI Less Than 25' in intervention.int_label:
                labels.append('BMI<25\n& No HBP')
            else:
                labels.append('50% BMI\nReduction')
    
    bars = ax.bar(labels, probabilities, color=['gray', 'blue', 'green'])
    ax.set_ylabel('Probability of Binary Outcome')
    ax.set_title('Binary End-of-Follow-up Outcome by Intervention')
    ax.set_ylim(0, 1)
    
    # Add value labels on bars
    for bar, prob in zip(bars, probabilities):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                f'{prob:.3f}', ha='center', va='bottom')
    
    plt.tight_layout()
    plt.savefig('example3_probabilities.png', dpi=300, bbox_inches='tight')
    print("Probability plot saved to: example3_probabilities.png")
    
    # Create summary report
    plotter = GFormulaPlotter(results)
    plotter.create_summary_report('example3_report.pdf')
    print("Full report saved to: example3_report.pdf")
    
    print("\n" + "=" * 60)
    print("Example 3 completed successfully!")
    print("=" * 60)


if __name__ == "__main__":
    main()