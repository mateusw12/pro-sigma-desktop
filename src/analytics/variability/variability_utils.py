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
    lsl: Optional[float] = None,
    usl: Optional[float] = None,
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
        lsl: Lower Specification Limit (optional)
        usl: Upper Specification Limit (optional)
        figsize: Figure size
        
    Returns:
        Matplotlib Figure object
    """
    if not x_columns or y_column not in df.columns:
        raise ValueError("Invalid columns specified")
    
    # Adjust figure size based on data volume
    n_points = len(df)
    total_groups = sum(df[col].nunique() for col in x_columns)
    
    # Increase width for many groups
    if total_groups > 50:
        figsize = (min(figsize[0] * 1.5, 24), figsize[1])
    elif total_groups > 30:
        figsize = (min(figsize[0] * 1.2, 20), figsize[1])
    
    # Create figure
    fig, ax = plt.subplots(1, 1, figsize=figsize)
    
    # Add warning text if too many data points
    if n_points > 1000:
        fig.text(
            0.5, 0.98,
            f'Atenção: Visualizando {n_points} pontos. Considere filtrar os dados para melhor visualização.',
            ha='center', va='top', fontsize=9, color='red', style='italic'
        )
    
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
        
        # Track separator positions to avoid duplicates per level
        separator_positions_by_level = {}
        
        # Color palette for different levels
        level_colors = ['#e74c3c', '#3498db', '#2ecc71', '#f39c12', '#9b59b6', '#1abc9c']
        
        # Add separators for each requested level
        for level_idx in valid_levels:
            col = x_columns[level_idx]
            
            # Different appearance for each level
            line_color = level_colors[level_idx % len(level_colors)]
            linewidth = 1.2 - (level_idx * 0.15)  # Thinner lines
            alpha = 0.7
            
            # Calculate extension for this specific level only
            label_extension_start = 0
            label_extension_end = -0.08 - (level_idx * 0.08) - 0.04
            
            if level_idx not in separator_positions_by_level:
                separator_positions_by_level[level_idx] = set()
            
            for group_name in df_sorted[col].unique()[:-1]:
                group_data = df_sorted[df_sorted[col] == group_name]
                separator_pos = group_data['_x_position'].max() + 0.5
                
                # Only add if not already added at this position for this level
                if separator_pos not in separator_positions_by_level[level_idx]:
                    separator_positions_by_level[level_idx].add(separator_pos)
                    
                    # Normalize x position for consistent rendering
                    x_range = df_sorted['_x_position'].max() - df_sorted['_x_position'].min()
                    x_norm = (separator_pos - df_sorted['_x_position'].min()) / x_range if x_range > 0 else 0.5
                    
                    # Draw continuous line from top of plot through to label area
                    # Using single line for consistency
                    y_top = 1.0  # Top of plot in axes coordinates
                    y_bottom = label_extension_end  # Bottom of label area
                    
                    ax.plot(
                        [x_norm, x_norm],
                        [y_bottom, y_top],
                        color=line_color,
                        linestyle='-',
                        linewidth=linewidth,
                        alpha=alpha,
                        transform=ax.transAxes,
                        clip_on=False,
                        zorder=2,
                        solid_capstyle='butt'
                    )
    
    # Create hierarchical x-axis labels (JMP style - multiple levels)
    # Disable default x-axis
    ax.set_xticks([])
    ax.set_xticklabels([])
    
    # Calculate bottom margin needed for hierarchical labels
    n_levels = len(x_columns)
    bottom_margin = 0.15 + (n_levels * 0.05)  # Increase margin for each level
    
    # Determine font size based on number of data points and groups
    n_points = len(df_sorted)
    total_groups = sum(df_sorted[col].nunique() for col in x_columns)
    
    # Adjust font sizes based on data volume
    if n_points > 500 or total_groups > 50:
        label_fontsize = 7
        level_fontsize = 7
        pad_size = 0.2
    elif n_points > 200 or total_groups > 30:
        label_fontsize = 8
        level_fontsize = 8
        pad_size = 0.25
    else:
        label_fontsize = 9
        level_fontsize = 9
        pad_size = 0.3
    
    # Create text labels for each hierarchical level
    for level_idx, x_col in enumerate(x_columns):
        # Calculate y position for this level (below the plot)
        y_pos = -0.08 - (level_idx * 0.08)
        
        # Get unique groups for this level
        grouped = df_sorted.groupby(x_col)
        unique_groups = df_sorted[x_col].unique()
        n_groups = len(unique_groups)
        
        # Track positions to avoid label overlap
        prev_end = -1
        
        for group_name in unique_groups:
            group_data = df_sorted[df_sorted[x_col] == group_name]
            x_start = group_data['_x_position'].min()
            x_end = group_data['_x_position'].max()
            x_center = (x_start + x_end) / 2
            
            # Normalize position to axis coordinates (0-1)
            x_range = df_sorted['_x_position'].max() - df_sorted['_x_position'].min()
            x_norm = (x_center - df_sorted['_x_position'].min()) / x_range if x_range > 0 else 0.5
            
            # Calculate label width to check for overlap
            label_width = (x_end - x_start) / x_range if x_range > 0 else 0.1
            
            # Decide rotation based on space available
            rotation = 0
            if label_width < 0.05 or n_groups > 20:  # Tight space
                rotation = 45
                ha = 'right'
            else:
                ha = 'center'
            
            # Truncate long labels if necessary
            label_text = str(group_name)
            if len(label_text) > 15 and n_groups > 10:
                label_text = label_text[:12] + '...'
            
            # Add text label
            ax.text(
                x_norm,
                y_pos,
                label_text,
                transform=ax.transAxes,
                ha=ha,
                va='top',
                fontsize=label_fontsize,
                fontweight='bold' if level_idx == 0 else 'normal',
                rotation=rotation,
                bbox=dict(boxstyle=f'round,pad={pad_size}', facecolor='lightgray' if level_idx % 2 == 0 else 'white', 
                         edgecolor='gray', alpha=0.6, linewidth=0.5)
            )
        
        # Add level label on the left
        ax.text(
            -0.02,
            y_pos,
            x_col + ':',
            transform=ax.transAxes,
            ha='right',
            va='top',
            fontsize=level_fontsize,
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
        label=f'Média Geral: {overall_mean:.2f}'
    )
    
    # Add specification limits if provided
    if lsl is not None:
        ax.axhline(
            y=lsl,
            color='red',
            linestyle='--',
            linewidth=2,
            alpha=0.8,
            label=f'LSL: {lsl:.2f}',
            zorder=5
        )
        # Add shaded area below LSL
        ax.axhspan(
            ax.get_ylim()[0],
            lsl,
            alpha=0.1,
            color='red',
            zorder=0
        )
    
    if usl is not None:
        ax.axhline(
            y=usl,
            color='red',
            linestyle='--',
            linewidth=2,
            alpha=0.8,
            label=f'USL: {usl:.2f}',
            zorder=5
        )
        # Add shaded area above USL
        ax.axhspan(
            usl,
            ax.get_ylim()[1],
            alpha=0.1,
            color='red',
            zorder=0
        )
    
    # Add green zone between limits if both are specified
    if lsl is not None and usl is not None:
        ax.axhspan(
            lsl,
            usl,
            alpha=0.05,
            color='green',
            zorder=0
        )
    
    # Grid and legend
    ax.grid(True, alpha=0.3, linestyle='--', linewidth=0.5)
    ax.legend(loc='upper right', fontsize=10, framealpha=0.9)
    
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
