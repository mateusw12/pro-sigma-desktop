"""
Variability Analysis Utilities
Functions for creating variability charts with multiple X and Y axes
Similar to JMP Variability Chart
"""
from typing import List, Optional, Tuple
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.figure import Figure


def create_variability_chart(
    df: pd.DataFrame,
    x_columns: List[str],
    y_column: str,
    show_mean_lines: bool = True,
    show_std_bands: bool = False,
    show_connect_lines: bool = True,
    figsize: Tuple[int, int] = (12, 8)
) -> Figure:
    """
    Creates a variability chart with multiple X factors and one Y response
    
    Args:
        df: DataFrame with data
        x_columns: List of column names to use as X factors (multiple levels)
        y_column: Column name for Y response variable
        show_mean_lines: Show mean lines for each group
        show_std_bands: Show standard deviation bands
        show_connect_lines: Connect points within groups
        figsize: Figure size
        
    Returns:
        Matplotlib Figure object
    """
    if not x_columns or y_column not in df.columns:
        raise ValueError("Invalid columns specified")
    
    # Check if all x_columns exist in dataframe
    for col in x_columns:
        if col not in df.columns:
            raise ValueError(f"Column '{col}' not found in dataframe")
    
    # Create figure with subplots for each X factor
    n_factors = len(x_columns)
    fig, axes = plt.subplots(n_factors, 1, figsize=figsize, sharex=False)
    
    # If only one factor, make axes a list for consistency
    if n_factors == 1:
        axes = [axes]
    
    # Color palette
    colors = plt.cm.tab10(np.linspace(0, 1, 10))
    
    # Process each X factor
    for idx, x_col in enumerate(x_columns):
        ax = axes[idx]
        
        # Group data by X factor
        grouped = df.groupby(x_col)[y_column]
        groups = sorted(df[x_col].unique())
        
        # Calculate statistics for each group
        means = []
        stds = []
        positions = []
        
        for i, group_name in enumerate(groups):
            group_data = grouped.get_group(group_name)
            
            # Create x positions with slight jitter for visibility
            x_pos = np.ones(len(group_data)) * i
            x_jitter = np.random.normal(0, 0.02, len(group_data))
            x_positions = x_pos + x_jitter
            
            # Plot individual points
            ax.scatter(
                x_positions,
                group_data,
                alpha=0.6,
                s=50,
                color=colors[i % len(colors)],
                edgecolors='black',
                linewidth=0.5,
                zorder=3
            )
            
            # Store statistics
            positions.append(i)
            means.append(group_data.mean())
            stds.append(group_data.std())
            
            # Connect points within group if requested
            if show_connect_lines and len(group_data) > 1:
                # Sort by y value for cleaner lines
                sorted_data = group_data.sort_values()
                sorted_x = x_pos[0] + np.random.normal(0, 0.02, len(sorted_data))
                ax.plot(
                    sorted_x,
                    sorted_data,
                    color=colors[i % len(colors)],
                    alpha=0.3,
                    linewidth=0.5,
                    zorder=1
                )
        
        # Plot mean lines
        if show_mean_lines:
            ax.plot(
                positions,
                means,
                color='red',
                linewidth=2,
                marker='D',
                markersize=8,
                label='Mean',
                zorder=4
            )
            
            # Add mean values as text
            for pos, mean in zip(positions, means):
                ax.text(
                    pos,
                    mean,
                    f'{mean:.2f}',
                    ha='center',
                    va='bottom',
                    fontsize=8,
                    color='red',
                    fontweight='bold'
                )
        
        # Plot standard deviation bands
        if show_std_bands:
            means_array = np.array(means)
            stds_array = np.array(stds)
            ax.fill_between(
                positions,
                means_array - stds_array,
                means_array + stds_array,
                alpha=0.2,
                color='gray',
                label='±1 Std Dev',
                zorder=2
            )
        
        # Styling
        ax.set_ylabel(y_column, fontsize=10, fontweight='bold')
        ax.set_xticks(positions)
        ax.set_xticklabels(groups, rotation=45, ha='right')
        ax.set_title(f'{y_column} by {x_col}', fontsize=12, fontweight='bold')
        ax.grid(True, alpha=0.3, linestyle='--', linewidth=0.5)
        
        # Add legend only to first subplot
        if idx == 0:
            ax.legend(loc='upper right', fontsize=9)
        
        # Add horizontal line at overall mean
        overall_mean = df[y_column].mean()
        ax.axhline(
            y=overall_mean,
            color='blue',
            linestyle='--',
            linewidth=1,
            alpha=0.5,
            label='Overall Mean' if idx == 0 else ''
        )
    
    # Main title
    fig.suptitle(
        f'Variability Chart: {y_column} by {", ".join(x_columns)}',
        fontsize=14,
        fontweight='bold',
        y=0.995
    )
    
    plt.tight_layout()
    return fig


def create_nested_variability_chart(
    df: pd.DataFrame,
    x_columns: List[str],
    y_column: str,
    show_mean_lines: bool = True,
    show_group_separators: bool = True,
    separator_levels: Optional[List[int]] = None,
    figsize: Tuple[int, int] = (14, 8)
) -> Figure:
    """
    Creates a nested variability chart showing hierarchical grouping
    All X factors displayed in a single plot with nested structure
    
    Args:
        df: DataFrame with data
        x_columns: List of column names in hierarchical order (outer to inner)
        y_column: Column name for Y response variable
        show_mean_lines: Show mean lines for each level
        show_group_separators: Show vertical lines separating groups
        separator_levels: List of indices (0-based) indicating which X factor levels to show separators for.
                         If None and show_group_separators is True, defaults to [0] (first level only)
        figsize: Figure size
        
    Returns:
        Matplotlib Figure object
    """
    if not x_columns or y_column not in df.columns:
        raise ValueError("Invalid columns specified")
    
    # Create figure
    fig, ax = plt.subplots(1, 1, figsize=figsize)
    
    # Create hierarchical grouping
    df_sorted = df.sort_values(by=x_columns)
    
    # Create nested group identifiers
    df_sorted['_group_id'] = df_sorted.groupby(x_columns).ngroup()
    df_sorted['_x_position'] = range(len(df_sorted))
    
    # Get color palette
    colors = plt.cm.tab10(np.linspace(0, 1, 10))
    
    # Plot individual points
    for group_id in df_sorted['_group_id'].unique():
        group_data = df_sorted[df_sorted['_group_id'] == group_id]
        
        ax.scatter(
            group_data['_x_position'],
            group_data[y_column],
            alpha=0.6,
            s=50,
            color=colors[group_id % len(colors)],
            edgecolors='black',
            linewidth=0.5,
            zorder=3
        )
        
        # Connect points within group
        ax.plot(
            group_data['_x_position'],
            group_data[y_column],
            color=colors[group_id % len(colors)],
            alpha=0.3,
            linewidth=1,
            zorder=1
        )
    
    # Add mean lines for each hierarchical level
    if show_mean_lines:
        # Innermost level means
        for i, x_col in enumerate(reversed(x_columns)):
            grouped = df_sorted.groupby(x_col)
            
            for group_name, group_data in grouped:
                x_positions = group_data['_x_position']
                y_mean = group_data[y_column].mean()
                
                ax.plot(
                    [x_positions.min(), x_positions.max()],
                    [y_mean, y_mean],
                    color='red',
                    linewidth=2 - i*0.3,
                    alpha=0.7,
                    zorder=4
                )
    
    # Add group separators
    if show_group_separators and len(x_columns) > 0:
        # Determine which levels to show separators for
        if separator_levels is None:
            separator_levels = [0]  # Default to first level only
        
        # Filter valid levels
        valid_levels = [lvl for lvl in separator_levels if 0 <= lvl < len(x_columns)]
        
        # Track separator positions to avoid duplicates
        separator_positions = set()
        
        # Add separators for each requested level
        for level_idx in valid_levels:
            col = x_columns[level_idx]
            # Use different line styles for different levels
            linestyle = '--' if level_idx == 0 else ':'
            linewidth = 2.0 - (level_idx * 0.3)
            alpha = 0.6 - (level_idx * 0.1)
            
            for group_name in df_sorted[col].unique()[:-1]:
                group_data = df_sorted[df_sorted[col] == group_name]
                separator_pos = group_data['_x_position'].max() + 0.5
                
                # Only add if not already added at this position
                if separator_pos not in separator_positions:
                    separator_positions.add(separator_pos)
                    ax.axvline(
                        x=separator_pos,
                        color='gray',
                        linestyle=linestyle,
                        linewidth=linewidth,
                        alpha=alpha,
                        zorder=2
                    )
    
    # Create hierarchical x-axis labels (JMP style - multiple levels)
    # Disable default x-axis
    ax.set_xticks([])
    ax.set_xticklabels([])
    
    # Calculate bottom margin needed for hierarchical labels
    n_levels = len(x_columns)
    bottom_margin = 0.15 + (n_levels * 0.05)  # Increase margin for each level
    
    # Create text labels for each hierarchical level
    for level_idx, x_col in enumerate(x_columns):
        # Calculate y position for this level (below the plot)
        y_pos = -0.08 - (level_idx * 0.08)
        
        # Get unique groups for this level
        grouped = df_sorted.groupby(x_col)
        
        # Track positions to avoid label overlap
        prev_end = -1
        
        for group_name in df_sorted[x_col].unique():
            group_data = df_sorted[df_sorted[x_col] == group_name]
            x_start = group_data['_x_position'].min()
            x_end = group_data['_x_position'].max()
            x_center = (x_start + x_end) / 2
            
            # Normalize position to axis coordinates (0-1)
            x_range = df_sorted['_x_position'].max() - df_sorted['_x_position'].min()
            x_norm = (x_center - df_sorted['_x_position'].min()) / x_range if x_range > 0 else 0.5
            
            # Add text label
            ax.text(
                x_norm,
                y_pos,
                str(group_name),
                transform=ax.transAxes,
                ha='center',
                va='top',
                fontsize=10,
                fontweight='bold' if level_idx == 0 else 'normal',
                bbox=dict(boxstyle='round,pad=0.3', facecolor='lightgray' if level_idx % 2 == 0 else 'white', 
                         edgecolor='gray', alpha=0.7)
            )
        
        # Add level label on the left
        ax.text(
            -0.02,
            y_pos,
            x_col + ':',
            transform=ax.transAxes,
            ha='right',
            va='top',
            fontsize=9,
            fontweight='bold',
            style='italic',
            color='darkblue'
        )
    
    # Labels and title
    ax.set_ylabel(y_column, fontsize=12, fontweight='bold')
    ax.set_xlabel('')  # Remove x label as we have hierarchical labels
    ax.set_title(
        f'Variability Chart: {y_column}',
        fontsize=14,
        fontweight='bold'
    )
    
    # Add overall mean line
    overall_mean = df[y_column].mean()
    ax.axhline(
        y=overall_mean,
        color='blue',
        linestyle='--',
        linewidth=1.5,
        alpha=0.6,
        label=f'Overall Mean: {overall_mean:.2f}'
    )
    
    # Grid and legend
    ax.grid(True, alpha=0.3, linestyle='--', linewidth=0.5)
    ax.legend(loc='upper right', fontsize=10)
    
    plt.tight_layout()
    return fig


def calculate_variability_statistics(
    df: pd.DataFrame,
    x_columns: List[str],
    y_column: str
) -> pd.DataFrame:
    """
    Calculate statistics for variability analysis
    
    Args:
        df: DataFrame with data
        x_columns: List of X factor columns
        y_column: Y response column
        
    Returns:
        DataFrame with statistics for each group
    """
    results = []
    
    # Overall statistics
    overall_stats = {
        'Factor': 'Overall',
        'Group': 'All Data',
        'N': len(df),
        'Mean': df[y_column].mean(),
        'StdDev': df[y_column].std(),
        'Min': df[y_column].min(),
        'Max': df[y_column].max(),
        'Range': df[y_column].max() - df[y_column].min()
    }
    results.append(overall_stats)
    
    # Statistics for each X factor
    for x_col in x_columns:
        grouped = df.groupby(x_col)[y_column]
        
        for group_name, group_data in grouped:
            stats = {
                'Factor': x_col,
                'Group': str(group_name),
                'N': len(group_data),
                'Mean': group_data.mean(),
                'StdDev': group_data.std(),
                'Min': group_data.min(),
                'Max': group_data.max(),
                'Range': group_data.max() - group_data.min()
            }
            results.append(stats)
    
    return pd.DataFrame(results)
