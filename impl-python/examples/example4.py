"""
Example 4: Continuous end-of-follow-up outcome
Python implementation of example4.sas
"""

import numpy as np
import pandas as pd
import sys
sys.path.append('../src')

from gformula import GFormula, CovariateSpec, InterventionSpec
from gformula.utils.plotting import GFormulaPlotter


def create_sample_data(seed=5027):
    """Create sample data for continuous end-of-follow-up outcome."""
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
            
            # Note: No competing event (dead) for continuous outcome
            dead = np.nan
            
            # Continuous outcome only at end of follow-up
            if time == 9:
                # Continuous outcome related to BMI with noise
                cont_e = round(bmi + 5 * np.random.normal(), 2)
            else:
                cont_e = np.nan
            
            # Binary outcome (not used in this example)
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
            
            # Stop if censored
            if censlost:
                break
    
    # Create dataframe
    df = pd.DataFrame(data_list)
    
    # Apply censoring logic for continuous EOFU outcome
    df.loc[df['time'] < 9, 'cont_e'] = np.nan
    df.loc[df['censlost'] == 1, 'cont_e'] = np.nan
    
    return df


def main():
    """Run example 4 analysis."""
    print("GFORMULA Example 4: Continuous End-of-Follow-up Outcome")
    print("=" * 60)
    
    # Create sample data
    print("\nCreating sample data...")
    data = create_sample_data()
    
    # Print data summary
    print(f"\nData summary:")
    print(f"Number of subjects: {data['id'].nunique()}")
    print(f"Number of records: {len(data)}")
    print(f"Time range: {data['time'].min()} to {data['time'].max()}")
    
    # Check continuous outcome at end
    final_data = data[data['time'] == 9]
    cont_e_valid = final_data['cont_e'].dropna()
    print(f"\nContinuous outcome at end of follow-up:")
    print(f"  N with outcome: {len(cont_e_valid)}")
    print(f"  Mean: {cont_e_valid.mean():.2f}")
    print(f"  SD: {cont_e_valid.std():.2f}")
    print(f"  Min: {cont_e_valid.min():.2f}")
    print(f"  Max: {cont_e_valid.max():.2f}")
    print(f"  Missing: {final_data['cont_e'].isna().sum()} subjects")
    
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
        outcome='cont_e',
        outcome_type='conteofu',
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
    
    print("\nMean continuous outcome at end of follow-up:")
    for int_no, outcome_data in results.counterfactual_risks.items():
        mean_val = outcome_data['mean'].values[0]
        std_val = outcome_data['std'].values[0]
        if int_no == 0:
            print(f"  Natural course: {mean_val:.2f} (SD: {std_val:.2f})")
        else:
            intervention = next(i for i in interventions if i.int_no == int_no)
            print(f"  {intervention.int_label}: {mean_val:.2f} (SD: {std_val:.2f})")
    
    print("\nMean differences vs natural course:")
    for _, row in results.risk_differences.iterrows():
        print(f"  {row['intervention_label']}: {row['difference']:.2f}")
    
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
        outcome='cont_e',
        outcome_type='conteofu',
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
        outcome='cont_e',
        outcome_type='conteofu',
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
    
    # =========================================================================
    # Create plots
    # =========================================================================
    print("\n" + "-" * 60)
    print("Creating visualizations...")
    print("-" * 60)
    
    # Since this is a continuous EOFU outcome, we'll create custom plots
    import matplotlib.pyplot as plt
    
    # Bar plot of mean outcomes
    fig, ax = plt.subplots(figsize=(10, 6))
    
    means = []
    stds = []
    labels = []
    
    for int_no, outcome_data in results.counterfactual_risks.items():
        mean_val = outcome_data['mean'].values[0]
        std_val = outcome_data['std'].values[0]
        means.append(mean_val)
        stds.append(std_val)
        
        if int_no == 0:
            labels.append('Natural\nCourse')
        else:
            intervention = next(i for i in interventions if i.int_no == int_no)
            # Shorten label for plot
            if 'BMI Less Than 25' in intervention.int_label:
                labels.append('BMI<25\n& No HBP')
            else:
                labels.append('50% BMI\nReduction')
    
    # Create bar plot with error bars
    x = np.arange(len(labels))
    bars = ax.bar(x, means, yerr=stds, capsize=5, 
                   color=['gray', 'blue', 'green'],
                   error_kw={'linewidth': 2})
    
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel('Mean Continuous Outcome')
    ax.set_title('Continuous End-of-Follow-up Outcome by Intervention')
    
    # Add value labels on bars
    for bar, mean, std in zip(bars, means, stds):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + std + 0.5,
                f'{mean:.1f}', ha='center', va='bottom')
    
    plt.tight_layout()
    plt.savefig('example4_means.png', dpi=300, bbox_inches='tight')
    print("Mean outcome plot saved to: example4_means.png")
    
    # Plot covariate means over time
    plotter = GFormulaPlotter(results)
    fig = plotter.plot_covariate_means('bmi', save_path='example4_bmi_trajectory.png')
    print("BMI trajectory plot saved to: example4_bmi_trajectory.png")
    
    # Create summary report
    plotter.create_summary_report('example4_report.pdf')
    print("Full report saved to: example4_report.pdf")
    
    # =========================================================================
    # Additional analysis: Compare results from different bootstrap parts
    # =========================================================================
    print("\n" + "-" * 60)
    print("Comparing bootstrap results...")
    print("-" * 60)
    
    # This demonstrates how you might combine results from multiple bootstrap runs
    # In practice, the bootstrap_results function would handle this
    
    print("\nEstimates from different bootstrap parts:")
    print("(In practice, these would be combined for final CIs)")
    
    # Save intermediate results that could be combined later
    results_boot1.risk_differences.to_csv('example4_bootstrap_part1.csv', index=False)
    results_boot2.risk_differences.to_csv('example4_bootstrap_part2.csv', index=False)
    print("\nBootstrap results saved for potential combination:")
    print("  Part 1: example4_bootstrap_part1.csv")
    print("  Part 2: example4_bootstrap_part2.csv")
    
    print("\n" + "=" * 60)
    print("Example 4 completed successfully!")
    print("=" * 60)


if __name__ == "__main__":
    main()