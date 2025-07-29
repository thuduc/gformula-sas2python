# GFORMULA Python Implementation

This is a Python implementation of the GFORMULA SAS macro (v4.0), providing the parametric g-formula for causal inference with time-varying treatments and confounders.

## Installation

1. Create a virtual environment:
```bash
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

2. Install the package:
```bash
pip install -r requirements.txt
pip install -e .
```

## Quick Start

```python
from gformula import GFormula, CovariateSpec, InterventionSpec

# Define covariates
covariates = [
    CovariateSpec(name='hbp', cov_type=2, predictor_type='lag1bin'),
    CovariateSpec(name='bmi', cov_type=4, predictor_type='lag2cub')
]

# Define intervention
intervention = InterventionSpec(
    int_no=1,
    int_label='BMI less than 25',
    variables=['bmi'],
    int_types=[2],  # threshold
    times=list(range(10)),
    max_values=[25]
)

# Run analysis
gf = GFormula(
    data=df,
    id_var='id',
    time_var='time',
    outcome='dia',
    outcome_type='binsurv',
    time_points=10,
    covariates=covariates,
    interventions=[intervention],
    n_simulations=1000
)

results = gf.fit()
print(results.summary())
```

## Features Implemented

### Core Functionality
- ✅ Parametric g-formula algorithm
- ✅ Monte Carlo simulation engine
- ✅ Multiple outcome types (binsurv, bineofu, conteofu)
- ✅ Time-varying covariates with various transformations
- ✅ Multiple intervention types (static, threshold, proportional, addition)
- ✅ Bootstrap confidence intervals
- ✅ Competing risks and censoring

### Covariate Transformations
- ✅ Lagged variables (lag1, lag2, lag3)
- ✅ Cumulative averages (cumavg, cumavglag1)
- ✅ Time switch variables (tsswitch0, tsswitch1)
- ✅ Polynomial terms (quadratic, cubic)
- ✅ Penalized splines (3, 4, or 5 knots)

### Models
- ✅ Logistic regression for binary outcomes
- ✅ Linear regression for continuous outcomes
- ✅ Pooled logistic regression for survival outcomes
- ✅ Truncated normal models
- ✅ Zero-inflated models (basic implementation)

### Output
- ✅ Risk differences and ratios
- ✅ Survival curves
- ✅ Covariate means over time
- ✅ Model diagnostics
- ✅ Plotting utilities

## Examples

See the `examples/` directory for Python implementations of the SAS examples:
- `example1.py`: Binary survival outcome
- More examples coming soon...

## Known Issues and Limitations

1. **Model Convergence**: Some models may have convergence issues with small datasets or rare outcomes. The implementation includes fallback methods but results should be validated.

2. **Performance**: The Python implementation may be slower than SAS for very large datasets. Consider using smaller `n_simulations` for testing.

3. **Categorical Outcomes**: Full support for categorical end-of-follow-up outcomes is still being implemented.

4. **Spline Implementation**: The spline implementation may differ slightly from SAS, potentially leading to different results.

## Testing

Run tests with:
```bash
pytest tests/
```

## Migration Notes from SAS

Key differences from the SAS implementation:
1. Uses pandas DataFrames instead of SAS datasets
2. Model formulas are handled internally (no need to specify)
3. Bootstrap can be parallelized using `n_jobs` parameter
4. Results are returned as Python objects with methods for plotting and export

## Contributing

This is an active migration project. Issues and pull requests are welcome.

## License

Same as the original GFORMULA SAS macro.