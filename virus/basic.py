"""
Normal Distribution Sampling Analysis

This script demonstrates the law of large numbers by:
1. Generating samples from a normal distribution with varying sample sizes
2. Computing sample statistics (mean and standard deviation)
3. Visualizing how these statistics converge to true parameters
4. Using logarithmic scaling to handle large range of sample sizes

Key components:
- Sample sizes: 10^1 through 10^6
- Normal distribution parameters: μ=0, σ=1
- Visualization using plotly with dual y-axes
"""

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

def generate_sample_sizes():
    """Generate array of sample sizes from 10^1 to 10^6 in half magnitudes."""
    # Create exponents from 1 to 6 in 0.5 steps
    exponents = np.arange(1, 6.1, 0.5)
    return np.array([10**i for i in exponents]).astype(int)

def generate_samples(mu, sigma, sample_sizes, n_iterations=10):
    """
    Generate multiple samples for each sample size from normal distribution.
    
        mu (float): Mean of normal distribution
        sigma (float): Standard deviation of normal distribution
        sample_sizes (np.array): Array of sample sizes
        n_iterations (int): Number of times to sample at each sample size
    
    Returns:
        tuple: Arrays of sample means and standard deviations, each with shape (n_sample_sizes, n_iterations)
    """
    sample_means = np.zeros((len(sample_sizes), n_iterations))
    sample_stds = np.zeros((len(sample_sizes), n_iterations))
    
    for i, n in enumerate(sample_sizes):
        for j in range(n_iterations):
            samples = np.random.normal(mu, sigma, n)
            sample_means[i, j] = np.mean(samples)
            sample_stds[i, j] = np.std(samples)
    
    return sample_means, sample_stds

def create_plot(sample_sizes, sample_means, sample_stds):
    """
    Create interactive plot of absolute differences with distribution in log-log scale.
    """
    # Calculate absolute differences from true parameters
    mean_diff = np.abs(sample_means - 0)
    std_diff = np.abs(sample_stds - 1)
    
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    
    # Add mean difference traces - one for each iteration
    for i in range(mean_diff.shape[1]):
        fig.add_trace(
            go.Scatter(x=sample_sizes, y=mean_diff[:, i], 
                      name=f"Mean Error {i+1}",
                      line=dict(color='blue', width=1),
                      opacity=0.3,
                      showlegend=False),
            secondary_y=False,
        )
    
    # Add std difference traces - one for each iteration
    for i in range(std_diff.shape[1]):
        fig.add_trace(
            go.Scatter(x=sample_sizes, y=std_diff[:, i],
                      name=f"Std Error {i+1}",
                      line=dict(color='red', width=1),
                      opacity=0.3,
                      showlegend=False),
            secondary_y=True,
        )
    
    # Add mean of errors
    fig.add_trace(
        go.Scatter(x=sample_sizes, y=np.mean(mean_diff, axis=1),
                  name="Mean Error (average)",
                  line=dict(color='blue', width=3)),
        secondary_y=False,
    )
    
    fig.add_trace(
        go.Scatter(x=sample_sizes, y=np.mean(std_diff, axis=1),
                  name="Std Error (average)",
                  line=dict(color='red', width=3)),
        secondary_y=True,
    )
    
    # Update layout
    fig.update_layout(
        title="Distribution of Absolute Errors vs Sample Size (Log-Log Scale)",
        xaxis_type="log",
        xaxis_title="Number of Samples",
    )
    
    fig.update_yaxes(title_text="Mean Absolute Error", type="log", secondary_y=False)
    fig.update_yaxes(title_text="Standard Deviation Absolute Error", type="log", secondary_y=True)
    
    fig.show()

def create_violin_plot(sample_sizes, sample_means, sample_stds):
    """
    Create violin plots of absolute differences for each sample size.
    """
    # Calculate absolute differences from true parameters
    mean_diff = np.abs(sample_means - 0)
    std_diff = np.abs(sample_stds - 1)
    
    # Create two subplots
    fig = make_subplots(rows=2, cols=1, 
                       subplot_titles=("Mean Absolute Error", "Standard Deviation Absolute Error"))
    
    # Prepare data for violin plots
    for i, n in enumerate(sample_sizes):
        # Mean error violin plot
        fig.add_trace(
            go.Violin(x=[f"n={n}"] * len(mean_diff[i]), 
                     y=mean_diff[i],
                     name=f"n={n}",
                     box_visible=True,
                     meanline_visible=True,
                     line_color='blue',
                     fillcolor='lightblue',
                     opacity=0.6,
                     showlegend=False),
            row=1, col=1
        )
        
        # Std error violin plot
        fig.add_trace(
            go.Violin(x=[f"n={n}"] * len(std_diff[i]), 
                     y=std_diff[i],
                     name=f"n={n}",
                     box_visible=True,
                     meanline_visible=True,
                     line_color='red',
                     fillcolor='pink',
                     opacity=0.6,
                     showlegend=False),
            row=2, col=1
        )
    
    # Update layout
    fig.update_layout(
        title="Distribution of Errors by Sample Size",
        height=800,
        showlegend=False
    )
    
    # Update y-axes to log scale
    fig.update_yaxes(type="log", title_text="Error", row=1, col=1)
    fig.update_yaxes(type="log", title_text="Error", row=2, col=1)
    
    fig.show()

def main():
    """Main function to run the analysis."""
    # Set distribution parameters
    mu = 0
    sigma = 1
    
    # Generate sample sizes
    sample_sizes = generate_sample_sizes()
    
    # Generate samples and compute statistics (increase n_iterations for smoother violins)
    sample_means, sample_stds = generate_samples(mu, sigma, sample_sizes, n_iterations=100)
    
    # Create and display both plots
    create_plot(sample_sizes, sample_means, sample_stds)
    create_violin_plot(sample_sizes, sample_means, sample_stds)

if __name__ == "__main__":
    main()
