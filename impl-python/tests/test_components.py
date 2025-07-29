"""Unit tests for individual components."""

import pytest
import numpy as np
import pandas as pd
from gformula.core.data_structures import CovariateSpec, InterventionSpec, GFormulaParams
from gformula.preprocessing.covariates import CovariateProcessor
from gformula.preprocessing.validation import DataValidator
from gformula.models.logistic import LogisticModel
from gformula.models.linear import LinearModel


def test_covariate_spec():
    """Test CovariateSpec creation and validation."""
    # Valid covariate
    cov = CovariateSpec(name='hbp', cov_type=2, predictor_type='lag1bin')
    assert cov.name == 'hbp'
    assert cov.cov_type == 2
    assert cov.predictor_type == 'lag1bin'
    
    # Invalid cov_type
    with pytest.raises(ValueError):
        CovariateSpec(name='hbp', cov_type=10, predictor_type='lag1bin')
    
    # Invalid predictor_type
    with pytest.raises(ValueError):
        CovariateSpec(name='hbp', cov_type=2, predictor_type='invalid')


def test_intervention_spec():
    """Test InterventionSpec creation and validation."""
    # Valid intervention
    int_spec = InterventionSpec(
        int_no=1,
        int_label='Test intervention',
        variables=['bmi'],
        int_types=[2],
        times=[0, 1, 2],
        max_values=[25]
    )
    assert int_spec.int_no == 1
    assert int_spec.variables == ['bmi']
    assert int_spec.int_types == [2]
    
    # Mismatched lengths
    with pytest.raises(ValueError):
        InterventionSpec(
            int_no=1,
            int_label='Bad intervention',
            variables=['bmi', 'hbp'],
            int_types=[2],  # Only one type for two variables
            times=[0, 1, 2]
        )


def test_covariate_processor():
    """Test covariate processing functionality."""
    # Create simple data
    data = pd.DataFrame({
        'id': [1, 1, 1, 2, 2, 2],
        'time': [0, 1, 2, 0, 1, 2],
        'hbp': [0, 0, 1, 1, 1, 1],
        'bmi': [25, 26, 27, 30, 29, 28]
    })
    
    processor = CovariateProcessor(data, 'id', 'time')
    
    # Test lag creation
    df_lag = processor.create_lagged_variables('hbp', 1)
    assert 'hbp_l1' in df_lag.columns
    assert df_lag.loc[1, 'hbp_l1'] == 0  # Lag of time 1 for id 1
    assert pd.isna(df_lag.loc[0, 'hbp_l1'])  # No lag at time 0
    
    # Test cumulative average
    df_cum = processor.create_cumulative_average('bmi', 0)
    assert 'bmi_cumavg' in df_cum.columns
    # For id 1: cumavg at time 2 should be (25+26+27)/3 = 26
    assert abs(df_cum.loc[2, 'bmi_cumavg'] - 26) < 0.01


def test_data_validator():
    """Test data validation functionality."""
    # Valid data
    data = pd.DataFrame({
        'id': [1, 1, 2, 2],
        'time': [0, 1, 0, 1],
        'outcome': [0, 1, 0, 0],
        'cov1': [1, 0, 1, 1]
    })
    
    validator = DataValidator(data, 'id', 'time')
    
    # Test structure validation
    is_valid, errors = validator.validate_structure()
    assert is_valid
    assert len(errors) == 0
    
    # Test outcome validation
    is_valid, errors = validator.validate_outcome('outcome', 'binsurv')
    assert is_valid
    
    # Test with missing time points
    data_gaps = pd.DataFrame({
        'id': [1, 1, 1],
        'time': [0, 1, 3],  # Missing time 2
        'outcome': [0, 0, 1]
    })
    
    validator_gaps = DataValidator(data_gaps, 'id', 'time')
    is_valid, errors = validator_gaps.validate_structure()
    assert not is_valid
    assert any('missing time points' in e for e in errors)


def test_logistic_model():
    """Test logistic model fitting and prediction."""
    # Create simple data
    np.random.seed(42)
    n = 100
    data = pd.DataFrame({
        'x1': np.random.normal(0, 1, n),
        'x2': np.random.binomial(1, 0.5, n),
        'y': np.random.binomial(1, 0.3, n)
    })
    
    model = LogisticModel()
    model.fit(data, 'y', ['x1', 'x2'])
    
    # Check model fitted
    assert model.results is not None
    assert model.coefficients is not None
    
    # Test prediction
    probs = model.predict(data, type='response')
    assert len(probs) == n
    assert all(0 <= p <= 1 for p in probs)


def test_linear_model():
    """Test linear model fitting and prediction."""
    # Create simple data
    np.random.seed(42)
    n = 100
    data = pd.DataFrame({
        'x1': np.random.normal(0, 1, n),
        'x2': np.random.normal(5, 2, n)
    })
    data['y'] = 2 * data['x1'] + 0.5 * data['x2'] + np.random.normal(0, 0.5, n)
    
    model = LinearModel()
    model.fit(data, 'y', ['x1', 'x2'])
    
    # Check model fitted
    assert model.results is not None
    assert model.coefficients is not None
    
    # Test prediction
    preds = model.predict(data)
    assert len(preds) == n
    
    # Check coefficients are roughly correct
    coef_x1 = model.coefficients.loc['x1', 'coefficient']
    assert abs(coef_x1 - 2.0) < 0.5  # Should be close to 2


def test_gformula_params():
    """Test GFormulaParams validation."""
    # Create minimal valid data
    data = pd.DataFrame({
        'id': [1, 2],
        'time': [0, 0],
        'outcome': [0, 1]
    })
    
    # Valid params
    params = GFormulaParams(
        data=data,
        id_var='id',
        time_var='time',
        outcome='outcome',
        outcome_type='binsurv',
        time_points=5
    )
    assert params.outcome == 'outcome'
    assert params.outcome_type == 'binsurv'
    
    # Invalid outcome type
    with pytest.raises(ValueError):
        GFormulaParams(
            data=data,
            id_var='id',
            time_var='time',
            outcome='outcome',
            outcome_type='invalid',
            time_points=5
        )


if __name__ == "__main__":
    # Run tests
    pytest.main([__file__, '-v'])