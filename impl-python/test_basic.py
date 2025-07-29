"""Basic test to verify the Python implementation works."""

import numpy as np
import pandas as pd
from gformula import GFormula, CovariateSpec, InterventionSpec

# Create simple test data
np.random.seed(42)
n_subjects = 200
n_times = 5

data = []
for i in range(n_subjects):
    age = 40 + np.random.randint(-10, 10)
    for t in range(n_times):
        hbp = int(np.random.random() < 0.3)
        bmi = 25 + np.random.normal(0, 3)
        
        # Simple outcome model
        outcome_prob = 0.05 + 0.01 * t + 0.02 * hbp + 0.001 * max(0, bmi - 25)
        outcome = int(np.random.random() < outcome_prob)
        
        data.append({
            'id': i,
            'time': t,
            'age': age,
            'hbp': hbp,
            'bmi': bmi,
            'outcome': outcome
        })
        
        if outcome:
            break

df = pd.DataFrame(data)

print("Data shape:", df.shape)
print("\nData summary:")
print(df.groupby('time').agg({
    'outcome': 'mean',
    'hbp': 'mean',
    'bmi': 'mean'
}).round(3))

# Define simple analysis
covariates = [
    CovariateSpec(name='hbp', cov_type=2, predictor_type='lag1bin'),
    CovariateSpec(name='bmi', cov_type=4, predictor_type='lag1')
]

intervention = InterventionSpec(
    int_no=1,
    int_label='BMI < 25',
    variables=['bmi'],
    int_types=[2],
    times=list(range(5)),
    max_values=[25]
)

# Run G-Formula
print("\n" + "="*50)
print("Running G-Formula Analysis")
print("="*50)

gf = GFormula(
    data=df,
    id_var='id',
    time_var='time',
    outcome='outcome',
    outcome_type='binsurv',
    time_points=5,
    time_method='cat',  # Use categorical time
    covariates=covariates,
    fixed_covariates=['age'],
    interventions=[intervention],
    n_simulations=100,
    seed=42
)

results = gf.fit(verbose=True)

print("\n" + "="*50)
print("RESULTS")
print("="*50)
print("\nRisk Differences:")
print(results.risk_differences)

print("\nIntervention Effect Summary:")
rd_final = results.risk_differences[results.risk_differences['time'] == 4]
if len(rd_final) > 0:
    print(f"Risk difference at time 4: {rd_final['risk_difference'].values[0]:.3f}")
    print(f"Interpretation: The intervention {'reduces' if rd_final['risk_difference'].values[0] < 0 else 'increases'} risk by {abs(rd_final['risk_difference'].values[0]):.1%}")

print("\nAnalysis completed successfully!")