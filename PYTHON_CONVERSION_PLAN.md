# GFORMULA SAS to Python Migration Plan

## Implementation Status (Updated)

### ✅ Completed Components

1. **Project Structure** - Full Python package structure with proper organization
2. **Core Data Structures** - Dataclasses for parameters, covariates, interventions, and results
3. **Preprocessing Module** - Complete implementation of all covariate transformations
4. **Statistical Models** - Logistic, linear, and survival models with error handling
5. **Monte Carlo Simulation** - Full simulation engine with intervention support
6. **Bootstrap Module** - Confidence interval estimation with parallel support
7. **Plotting Utilities** - Comprehensive visualization tools
8. **Basic Testing** - Initial test suite and examples

### 🚧 In Progress

1. **Model Convergence** - Improving stability for edge cases
2. **Performance Optimization** - Vectorization and parallel processing
3. **Full Example Migration** - Converting all SAS examples to Python

### 📋 Remaining Work

1. **Categorical Outcomes** - Full multinomial support
2. **Advanced Bootstrap** - Multi-part bootstrap combination
3. **Documentation** - API reference and user guide
4. **Comprehensive Testing** - Full test coverage and validation

---

# GFORMULA SAS to Python Migration Plan

## Executive Summary

This document outlines a comprehensive plan to migrate the GFORMULA SAS macro (v4.0) to Python. The GFORMULA macro implements the parametric g-formula for causal inference, supporting time-varying treatments and confounders with various outcome types. The migration will create a modern, maintainable Python package while preserving all functionality.

## Project Overview

### Current State
- **Language**: SAS macro language
- **Version**: 4.0 (September 2024)
- **Size**: ~6,000 lines of SAS code
- **Components**: Single monolithic macro with multiple internal submacros
- **Dependencies**: Base SAS, SAS/STAT, SAS/GRAPH

### Target State
- **Language**: Python 3.8+
- **Architecture**: Modular object-oriented design
- **Dependencies**: NumPy, Pandas, SciPy, Statsmodels, Matplotlib
- **Testing**: Comprehensive unit and integration tests
- **Documentation**: Sphinx-based API documentation

## Technical Analysis

### Core Functionality to Migrate

1. **Data Processing**
   - Time-varying covariate handling
   - Lagged variable creation
   - Missing data handling
   - Data validation

2. **Statistical Models**
   - Logistic regression (binary outcomes)
   - Linear regression (continuous outcomes)
   - Pooled logistic regression (survival outcomes)
   - Spline functions
   - Interaction terms

3. **G-Formula Algorithm**
   - Monte Carlo simulation
   - Counterfactual prediction
   - Natural course estimation
   - Intervention implementation

4. **Outcome Types**
   - `binsurv`: Binary survival outcome
   - `bineofu`: Binary end-of-follow-up outcome
   - `cateofu`: Categorical end-of-follow-up outcome
   - `conteofu`: Continuous end-of-follow-up outcome

5. **Features**
   - Bootstrap confidence intervals
   - Competing risks
   - Censoring
   - Multiple interventions
   - Graphical output

### Key Challenges

1. **Macro Variable System**: SAS's extensive macro variable system needs careful translation
2. **Array Handling**: Dynamic array generation in SAS macros
3. **Statistical Procedures**: Mapping SAS PROC calls to Python equivalents
4. **Performance**: Ensuring Python implementation matches SAS performance
5. **Numerical Precision**: Maintaining statistical accuracy

## Migration Strategy

### Phase 1: Foundation (Weeks 1-2)

#### 1.1 Project Setup
```python
gformula-python/
├── src/
│   └── gformula/
│       ├── __init__.py
│       ├── core/
│       │   ├── __init__.py
│       │   ├── gformula.py
│       │   └── data_structures.py
│       ├── models/
│       │   ├── __init__.py
│       │   ├── base.py
│       │   ├── logistic.py
│       │   ├── linear.py
│       │   └── survival.py
│       ├── preprocessing/
│       │   ├── __init__.py
│       │   ├── covariates.py
│       │   └── validation.py
│       ├── simulation/
│       │   ├── __init__.py
│       │   ├── monte_carlo.py
│       │   └── interventions.py
│       └── utils/
│           ├── __init__.py
│           ├── bootstrap.py
│           └── plotting.py
├── tests/
├── docs/
├── examples/
└── setup.py
```

#### 1.2 Core Data Structures
```python
@dataclass
class GFormulaParams:
    """Parameters for G-Formula analysis"""
    outcome: str
    outcome_type: str
    time_points: int
    covariates: List[CovariateSpec]
    interventions: List[InterventionSpec]
    # ... additional parameters

@dataclass
class CovariateSpec:
    """Specification for a time-varying covariate"""
    name: str
    type: str  # binary, continuous, categorical
    predictor_type: str  # lag1bin, lag2cub, cumavg, etc.
    model_formula: Optional[str]
```

### Phase 2: Core Implementation (Weeks 3-6)

#### 2.1 Preprocessing Module
- Implement lagged variable creation
- Handle spline transformations
- Create cumulative averages
- Validate input data

#### 2.2 Modeling Module
- Base model class with fit/predict interface
- Specific implementations for each outcome type
- Handle fixed and time-varying covariates
- Support for interaction terms

#### 2.3 Simulation Engine
- Monte Carlo simulation framework
- Natural course simulation
- Intervention application logic
- Parallel processing support

### Phase 3: Advanced Features (Weeks 7-9)

#### 3.1 Bootstrap Module
- Implement bootstrap sampling
- Parallel bootstrap execution
- Confidence interval calculation
- Support for multi-part bootstrapping

#### 3.2 Visualization
- Survival curves
- Covariate means over time
- Risk difference plots
- Customizable plotting options

#### 3.3 Utilities
- Results aggregation
- Data export functions
- Logging and diagnostics
- Performance profiling

### Phase 4: Testing and Validation (Weeks 10-11)

#### 4.1 Unit Tests
```python
# Example test structure
class TestGFormula:
    def test_binary_survival_outcome(self):
        """Test basic binary survival analysis"""
        # Load example data
        # Run analysis
        # Compare with SAS results
        
    def test_intervention_application(self):
        """Test intervention implementation"""
        # Test static interventions
        # Test threshold interventions
        # Test dynamic interventions
```

#### 4.2 Integration Tests
- Replicate all four example analyses
- Compare results with SAS output
- Validate numerical precision
- Performance benchmarking

#### 4.3 Statistical Validation
- Cross-validate against published results
- Ensure bootstrap CIs match
- Verify handling of edge cases

### Phase 5: Documentation and Release (Week 12)

#### 5.1 Documentation
- API reference documentation
- Migration guide from SAS
- Theoretical background
- Example notebooks

#### 5.2 Packaging
- PyPI package setup
- Conda package (optional)
- Docker container
- CI/CD pipeline

## Implementation Details

### Key Python Mappings

| SAS Concept | Python Implementation |
|-------------|----------------------|
| Macro variables | Class attributes/config dict |
| Arrays | NumPy arrays/Pandas Series |
| PROC LOGISTIC | statsmodels.api.Logit |
| PROC REG | statsmodels.api.OLS |
| PROC MEANS | pandas.DataFrame.describe() |
| Data steps | Pandas operations |
| Macro loops | Python loops/vectorization |

### Example API Design

```python
from gformula import GFormula, Covariate, Intervention

# Define covariates
covariates = [
    Covariate('hbp', type='binary', predictor_type='lag1bin'),
    Covariate('bmi', type='continuous', predictor_type='lag2cub')
]

# Define intervention
intervention = Intervention(
    variable='bmi',
    type='threshold',
    max_value=25,
    times=range(10)
)

# Create and run analysis
gf = GFormula(
    data=df,
    id_col='id',
    time_col='time',
    outcome='dia',
    outcome_type='binsurv',
    covariates=covariates,
    interventions=[intervention],
    n_simulations=1000,
    n_bootstraps=500
)

results = gf.fit()
results.plot_survival_curves()
```

## Testing Strategy

### 1. Numerical Validation
- Create test datasets with known properties
- Compare Python results with SAS output
- Tolerance: < 0.001 for point estimates
- Bootstrap CI overlap validation

### 2. Unit Test Coverage
- Target: 95% code coverage
- Test each module independently
- Mock external dependencies
- Edge case testing

### 3. Integration Tests
```python
def test_example1_replication():
    """Replicate example1.sas results"""
    # Generate sample data
    data = create_example1_data()
    
    # Run analysis
    gf = GFormula(...)
    results = gf.fit()
    
    # Load SAS results
    sas_results = load_sas_results('example1_results.csv')
    
    # Compare
    assert np.allclose(results.risk_diff, sas_results.risk_diff, rtol=1e-3)
```

### 4. Performance Benchmarks
- Baseline: SAS execution time
- Target: Within 2x of SAS performance
- Profile bottlenecks
- Optimize critical paths

## Migration Timeline

### Month 1
- Week 1-2: Foundation and setup
- Week 3-4: Core preprocessing and data structures

### Month 2
- Week 5-6: Statistical models and simulation engine
- Week 7-8: Advanced features and bootstrap

### Month 3
- Week 9-10: Testing and validation
- Week 11: Documentation
- Week 12: Release preparation

## Risk Mitigation

### Technical Risks
1. **Statistical Accuracy**
   - Mitigation: Extensive validation against SAS
   - Continuous integration testing

2. **Performance**
   - Mitigation: Profile early and often
   - Use NumPy vectorization
   - Implement parallel processing

3. **Missing Features**
   - Mitigation: Careful analysis of SAS macro
   - Maintain feature parity checklist

### Project Risks
1. **Scope Creep**
   - Mitigation: Strict adherence to SAS functionality
   - Future enhancements in v2.0

2. **Testing Complexity**
   - Mitigation: Automated test generation
   - Use SAS to generate test cases

## Success Criteria

1. **Functional Parity**: All SAS features implemented
2. **Numerical Accuracy**: Results match within tolerance
3. **Performance**: Execution time within 2x of SAS
4. **Test Coverage**: >95% code coverage
5. **Documentation**: Complete API docs and examples
6. **User Acceptance**: Successfully replicate published analyses

## Next Steps

1. Review and approve migration plan
2. Set up development environment
3. Create detailed task breakdown
4. Begin Phase 1 implementation
5. Establish validation datasets

## Appendix: Technical Specifications

### Supported Predictor Types
- `lag1bin`, `lag2bin`: Lagged binary variables
- `lag1, lag2, lag3`: Lagged continuous variables
- `cumavg, cumavglag1`: Cumulative averages
- `tsswitch0, tsswitch1`: Time-switch variables
- `lag1quad, lag2quad`: Quadratic terms
- `lag1cub, lag2cub`: Cubic terms
- `pspline3, pspline4, pspline5`: Penalized splines

### Intervention Types
1. **Static** (type=1): Set to fixed value
2. **Threshold** (type=2): Enforce min/max bounds
3. **Proportional** (type=3): Multiply by factor
4. **Addition** (type=4): Add constant value

### Output Datasets
- Survival curves (observed and counterfactual)
- Covariate means over time
- Risk differences with confidence intervals
- Model coefficients
- Diagnostic statistics