# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

GFORMULA-SAS implements the parametric g-formula in SAS to estimate the risk or mean of an outcome under sustained treatment strategies. The macro is designed for causal inference in epidemiological studies with time-varying confounding.

## Core Architecture

### Main Components

1. **gformula4.0.sas** - The main macro file containing the g-formula implementation
   - Handles various outcome types: binary survival (binsurv), binary end-of-follow-up (bineofu), categorical (cateofu), continuous (conteofu variants)
   - Supports time-varying covariates with different predictor types
   - Implements parametric models for covariates and outcomes
   - Handles competing risks and censoring

2. **Example Files** (example1.sas through example4.sas)
   - Demonstrate different outcome types and intervention strategies
   - Include data generation and macro invocation patterns
   - Show how to specify interventions using the %let statements

### Key Concepts

- **Predictor Types (ptype)**: Define how covariate histories are included in models (e.g., lag1, lag2, cumavg)
- **Interventions**: Specified using intno, nintvar, intvar, inttype parameters
- **Time-Varying Covariates**: Modeled parametrically with specified functional forms
- **Outcome Types**: Different modeling approaches for survival, end-of-follow-up outcomes

## Common Development Tasks

### Running Examples
```sas
%include 'gformula4.0.sas';
%include 'example1.sas';  /* For binary survival outcome example */
```

### Testing Changes
- Run individual example files to test specific outcome types
- Check log for macro resolution issues with: `options mprint mlogic;`
- Verify intervention specifications produce expected results

### Key Macro Parameters
- `data=` - Input dataset
- `id=` - Subject identifier
- `time=` - Time variable
- `outc=` - Outcome variable
- `outctype=` - Outcome type (binsurv, bineofu, cateofu, conteofu)
- `ncov=` - Number of time-varying covariates
- `cov1=, cov2=, ...` - Covariate specifications

## Important Notes

- The macro uses SAS macro language extensively - changes require understanding of macro variable resolution
- Time-varying covariate models are built dynamically based on ptype specifications
- Intervention definitions use a specific format with intno, nintvar, intvar, inttype, intmin/intmax parameters
- The macro generates intermediate datasets during simulation - ensure adequate disk space