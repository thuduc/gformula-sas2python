"""Plotting utilities for G-Formula results."""

from typing import Dict, List, Optional, Tuple, Any
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.backends.backend_pdf import PdfPages

from ..core.data_structures import GFormulaResults


class GFormulaPlotter:
    """Create plots for G-Formula results."""
    
    def __init__(self, results: GFormulaResults, time_var: str = 'time'):
        """Initialize plotter.
        
        Args:
            results: G-Formula results object
            time_var: Name of time variable
        """
        self.results = results
        self.time_var = time_var
        
        # Set plot style
        sns.set_style("whitegrid")
        plt.rcParams['figure.figsize'] = (10, 6)
        
    def plot_survival_curves(self, save_path: Optional[str] = None,
                           show_ci: bool = True) -> plt.Figure:
        """Plot survival curves for all interventions.
        
        Args:
            save_path: Path to save plot
            show_ci: Whether to show confidence intervals
            
        Returns:
            Matplotlib figure
        """
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # Plot observed survival
        obs_data = self.results.observed_risk
        if 'survival' in obs_data.columns:
            ax.plot(obs_data[self.time_var], obs_data['survival'],
                   'k-', linewidth=2, label='Observed')
                   
        # Plot counterfactual survival curves
        colors = plt.cm.Set1(np.linspace(0, 1, len(self.results.counterfactual_risks)))
        
        for i, (int_no, cf_risk) in enumerate(self.results.counterfactual_risks.items()):
            if 'survival' not in cf_risk.columns:
                continue
                
            # Get intervention label
            if int_no == 0:
                label = "Natural Course"
            else:
                intervention = next(
                    (inv for inv in self.results.interventions if inv.int_no == int_no),
                    None
                )
                label = intervention.int_label if intervention else f"Intervention {int_no}"
                
            # Plot survival curve
            ax.plot(cf_risk[self.time_var], cf_risk['survival'],
                   color=colors[i], linewidth=2, label=label)
                   
            # Add confidence intervals if available
            if show_ci and self.results.confidence_intervals is not None:
                ci_data = self.results.confidence_intervals[
                    self.results.confidence_intervals['intervention'] == int_no
                ]
                if len(ci_data) > 0 and 'survival_lower' in ci_data.columns:
                    ax.fill_between(
                        ci_data[self.time_var],
                        ci_data['survival_lower'],
                        ci_data['survival_upper'],
                        color=colors[i],
                        alpha=0.2
                    )
                    
        # Formatting
        ax.set_xlabel('Time')
        ax.set_ylabel('Survival Probability')
        ax.set_title('Survival Curves by Intervention')
        ax.legend(loc='best')
        ax.set_ylim(0, 1)
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            
        return fig
    
    def plot_risk_differences(self, save_path: Optional[str] = None,
                            show_ci: bool = True) -> plt.Figure:
        """Plot risk differences over time.
        
        Args:
            save_path: Path to save plot
            show_ci: Whether to show confidence intervals
            
        Returns:
            Matplotlib figure
        """
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # Get unique interventions (excluding natural course)
        risk_diff_data = self.results.risk_differences
        interventions = risk_diff_data['intervention'].unique()
        interventions = [i for i in interventions if i != 0]
        
        colors = plt.cm.Set1(np.linspace(0, 1, len(interventions)))
        
        for i, int_no in enumerate(interventions):
            int_data = risk_diff_data[risk_diff_data['intervention'] == int_no]
            
            if self.time_var in int_data.columns:
                # Time-varying outcome
                ax.plot(int_data[self.time_var], int_data['risk_difference'],
                       color=colors[i], linewidth=2, 
                       label=int_data['intervention_label'].iloc[0])
                       
                # Add confidence intervals
                if show_ci and self.results.confidence_intervals is not None:
                    ci_data = self.results.confidence_intervals[
                        self.results.confidence_intervals['intervention'] == int_no
                    ]
                    if len(ci_data) > 0:
                        ax.fill_between(
                            ci_data[self.time_var],
                            ci_data['risk_diff_lower'],
                            ci_data['risk_diff_upper'],
                            color=colors[i],
                            alpha=0.2
                        )
                        
        # Add reference line at 0
        ax.axhline(y=0, color='black', linestyle='--', alpha=0.5)
        
        # Formatting
        ax.set_xlabel('Time')
        ax.set_ylabel('Risk Difference')
        ax.set_title('Risk Differences vs Natural Course')
        ax.legend(loc='best')
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            
        return fig
    
    def plot_covariate_means(self, covariate: str,
                           save_path: Optional[str] = None) -> plt.Figure:
        """Plot covariate means over time.
        
        Args:
            covariate: Name of covariate to plot
            save_path: Path to save plot
            
        Returns:
            Matplotlib figure
        """
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # Plot observed means
        obs_means = self.results.observed_covariate_means
        mean_col = f"{covariate}_mean"
        
        if mean_col in obs_means.columns:
            ax.plot(obs_means[self.time_var], obs_means[mean_col],
                   'k-', linewidth=2, label='Observed')
                   
        # Plot counterfactual means
        colors = plt.cm.Set1(np.linspace(0, 1, len(self.results.counterfactual_covariate_means)))
        
        for i, (int_no, cf_means) in enumerate(self.results.counterfactual_covariate_means.items()):
            if mean_col not in cf_means.columns:
                continue
                
            # Get intervention label
            if int_no == 0:
                label = "Natural Course"
            else:
                intervention = next(
                    (inv for inv in self.results.interventions if inv.int_no == int_no),
                    None
                )
                label = intervention.int_label if intervention else f"Intervention {int_no}"
                
            ax.plot(cf_means[self.time_var], cf_means[mean_col],
                   color=colors[i], linewidth=2, label=label)
                   
        # Formatting
        ax.set_xlabel('Time')
        ax.set_ylabel(f'Mean {covariate}')
        ax.set_title(f'{covariate} Mean Over Time by Intervention')
        ax.legend(loc='best')
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            
        return fig
    
    def create_summary_report(self, pdf_path: str) -> None:
        """Create a PDF report with all standard plots.
        
        Args:
            pdf_path: Path for PDF output
        """
        with PdfPages(pdf_path) as pdf:
            # Title page
            fig = plt.figure(figsize=(8.5, 11))
            fig.text(0.5, 0.7, 'G-Formula Analysis Report', 
                    ha='center', va='center', size=24, weight='bold')
            fig.text(0.5, 0.6, f'Generated: {pd.Timestamp.now().strftime("%Y-%m-%d %H:%M")}',
                    ha='center', va='center', size=12)
            fig.text(0.5, 0.5, f'Number of subjects: {self.results.n_subjects}',
                    ha='center', va='center', size=12)
            fig.text(0.5, 0.45, f'Number of time points: {self.results.n_time_points}',
                    ha='center', va='center', size=12)
            fig.text(0.5, 0.4, f'Number of simulations: {self.results.n_simulations}',
                    ha='center', va='center', size=12)
            pdf.savefig(fig)
            plt.close(fig)
            
            # Survival curves (if applicable)
            if 'survival' in self.results.observed_risk.columns:
                fig = self.plot_survival_curves(show_ci=True)
                pdf.savefig(fig)
                plt.close(fig)
                
            # Risk differences
            if len(self.results.risk_differences) > 0:
                fig = self.plot_risk_differences(show_ci=True)
                pdf.savefig(fig)
                plt.close(fig)
                
            # Covariate means
            if len(self.results.observed_covariate_means) > 0:
                # Plot first few covariates
                cov_cols = [c for c in self.results.observed_covariate_means.columns 
                           if c.endswith('_mean')]
                for cov_col in cov_cols[:3]:  # Limit to first 3 covariates
                    cov_name = cov_col.replace('_mean', '')
                    fig = self.plot_covariate_means(cov_name)
                    pdf.savefig(fig)
                    plt.close(fig)
                    
            # Results summary table
            fig, ax = plt.subplots(figsize=(8.5, 11))
            ax.axis('tight')
            ax.axis('off')
            
            # Create summary text
            summary_text = self.results.summary()
            ax.text(0.1, 0.9, summary_text, transform=ax.transAxes,
                   fontsize=10, verticalalignment='top', fontfamily='monospace')
            pdf.savefig(fig)
            plt.close(fig)