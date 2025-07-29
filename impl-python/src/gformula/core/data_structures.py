"""Data structures for the gformula package."""

from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any, Union, Tuple
import numpy as np
import pandas as pd


@dataclass
class CovariateSpec:
    """Specification for a time-varying covariate.
    
    Attributes:
        name: Name of the covariate
        cov_type: Type of covariate (1=binary, 2=binary, 3=continuous zero-inflated, 
                  4=continuous non-zero, 5=continuous truncated normal)
        predictor_type: Type of predictor (lag1bin, lag2bin, lag1, lag2, cumavg, etc.)
        model_formula: Optional custom model formula
        transform: Optional transformation function
        ref_value: Reference value for the covariate
        restriction: Optional restriction on covariate values
    """
    name: str
    cov_type: int
    predictor_type: str
    model_formula: Optional[str] = None
    transform: Optional[str] = None
    ref_value: Optional[float] = None
    restriction: Optional[str] = None
    
    def __post_init__(self):
        valid_types = [1, 2, 3, 4, 5]
        if self.cov_type not in valid_types:
            raise ValueError(f"cov_type must be one of {valid_types}")
        
        valid_predictors = [
            'lag1bin', 'lag2bin', 'lag1', 'lag2', 'lag3',
            'cumavg', 'cumavglag1', 'tsswitch0', 'tsswitch1',
            'lag1quad', 'lag2quad', 'lag1cub', 'lag2cub',
            'pspline3', 'pspline4', 'pspline5'
        ]
        if self.predictor_type not in valid_predictors:
            raise ValueError(f"predictor_type must be one of {valid_predictors}")


@dataclass
class InterventionSpec:
    """Specification for an intervention.
    
    Attributes:
        int_no: Intervention number
        int_label: Label for the intervention
        variables: List of variables to intervene on
        int_types: List of intervention types for each variable
        times: Time points when intervention applies
        values: Values for static interventions
        min_values: Minimum values for threshold interventions
        max_values: Maximum values for threshold interventions
        change_values: Change values for proportional/addition interventions
        probabilities: Probabilities for random interventions
        conditions: Optional conditions for intervention application
    """
    int_no: int
    int_label: str
    variables: List[str]
    int_types: List[int]  # 1=static, 2=threshold, 3=proportional, 4=addition
    times: List[int]
    values: Optional[List[float]] = None
    min_values: Optional[List[float]] = None
    max_values: Optional[List[float]] = None
    change_values: Optional[List[float]] = None
    probabilities: Optional[List[float]] = None
    conditions: Optional[str] = None
    
    def __post_init__(self):
        if len(self.variables) != len(self.int_types):
            raise ValueError("Number of variables must match number of intervention types")
        
        for int_type in self.int_types:
            if int_type not in [1, 2, 3, 4]:
                raise ValueError("Intervention type must be 1, 2, 3, or 4")


@dataclass
class GFormulaParams:
    """Parameters for G-Formula analysis.
    
    Core parameters from SAS macro.
    """
    # Data parameters
    data: pd.DataFrame
    id_var: str
    time_var: str
    
    # Outcome parameters
    outcome: str
    outcome_type: str  # binsurv, bineofu, cateofu, conteofu
    outcome_model: Optional[str] = None
    
    # Time parameters
    time_points: int = 0
    time_method: str = "concat"  # concat, cat
    time_knots: Optional[List[int]] = None
    
    # Covariates
    covariates: List[CovariateSpec] = field(default_factory=list)
    fixed_covariates: Optional[List[str]] = None
    
    # Competing events and censoring
    competing_event: Optional[str] = None
    competing_event_cens: int = 0
    censor: Optional[str] = None
    
    # Simulation parameters
    n_simulations: int = 10000
    seed: Optional[int] = None
    
    # Bootstrap parameters
    n_samples: int = 0
    sample_start: int = 0
    sample_end: int = -1
    
    # Intervention parameters
    interventions: List[InterventionSpec] = field(default_factory=list)
    ref_int: int = 0
    
    # Output parameters
    save_results: bool = True
    print_log_stats: bool = True
    check_cov_models: bool = True
    print_cov_means: bool = True
    save_raw_covmean: bool = False
    
    # Graph parameters
    run_graphs: bool = True
    graph_file: Optional[str] = None
    
    def __post_init__(self):
        valid_outcome_types = ['binsurv', 'bineofu', 'cateofu', 'conteofu']
        if self.outcome_type not in valid_outcome_types:
            raise ValueError(f"outcome_type must be one of {valid_outcome_types}")
        
        if self.time_method not in ['concat', 'cat']:
            raise ValueError("time_method must be 'concat' or 'cat'")


@dataclass
class GFormulaResults:
    """Results from G-Formula analysis."""
    # Survival/outcome estimates
    observed_risk: pd.DataFrame
    counterfactual_risks: Dict[int, pd.DataFrame]
    
    # Risk differences
    risk_differences: pd.DataFrame
    risk_ratios: pd.DataFrame
    
    # Covariate means
    observed_covariate_means: pd.DataFrame
    counterfactual_covariate_means: Dict[int, pd.DataFrame]
    
    # Model information
    model_coefficients: Dict[str, pd.DataFrame]
    model_diagnostics: Dict[str, Any]
    
    # Bootstrap results (if applicable)
    bootstrap_samples: Optional[np.ndarray] = None
    confidence_intervals: Optional[pd.DataFrame] = None
    
    # Metadata
    n_subjects: int = 0
    n_time_points: int = 0
    n_simulations: int = 0
    interventions: List[InterventionSpec] = field(default_factory=list)
    
    def summary(self) -> str:
        """Generate a summary of the results."""
        summary_lines = [
            f"G-Formula Results Summary",
            f"========================",
            f"Number of subjects: {self.n_subjects}",
            f"Number of time points: {self.n_time_points}",
            f"Number of simulations: {self.n_simulations}",
            f"Number of interventions: {len(self.interventions)}",
            "",
            "Risk Differences (vs Natural Course):",
            str(self.risk_differences),
        ]
        
        if self.confidence_intervals is not None:
            summary_lines.extend([
                "",
                "Confidence Intervals:",
                str(self.confidence_intervals),
            ])
        
        return "\n".join(summary_lines)