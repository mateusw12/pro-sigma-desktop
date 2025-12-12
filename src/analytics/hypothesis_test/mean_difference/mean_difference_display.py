"""
Mean Difference Test Display
"""
import customtkinter as ctk
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from typing import Dict


def display_mean_difference_results(parent, results: Dict):
    """Display mean difference test results"""
    mse_data = results.get("mse", {})
    
    if not mse_data:
        ctk.CTkLabel(
            parent,
            text="Nenhum resultado disponível",
            font=ctk.CTkFont(size=14)
        ).pack(pady=20)
        return
    
    # Create tabs for each response
    if len(mse_data) > 1:
        tabview = ctk.CTkTabview(parent)
        tabview.pack(fill="both", expand=True, padx=10, pady=10)
        
        for response_name, result_data in mse_data.items():
            tab = tabview.add(response_name)
            display_mean_difference_for_response(tab, result_data, response_name)
    else:
        response_name = list(mse_data.keys())[0]
        result_data = mse_data[response_name]
        display_mean_difference_for_response(parent, result_data, response_name)


def display_mean_difference_for_response(parent, result_data: Dict, response_name: str):
    """Display mean difference results for a single response"""
    content_frame = ctk.CTkFrame(parent)
    content_frame.pack(fill="both", expand=True, padx=10, pady=10)
    
    # Left - Table
    left_frame = ctk.CTkFrame(content_frame)
    left_frame.pack(side="left", fill="both", expand=True, padx=(0, 10))
    
    # Right - Chart
    right_frame = ctk.CTkFrame(content_frame)
    right_frame.pack(side="left", fill="both", expand=True)
    
    # Display table
    display_mean_difference_table(left_frame, result_data)
    
    # Display chart
    display_mean_difference_chart(right_frame, result_data, response_name)


def display_mean_difference_table(parent, result_data: Dict):
    """Display mean difference table"""
    table_frame = ctk.CTkFrame(parent)
    table_frame.pack(fill="both", expand=True, pady=10)
    
    ctk.CTkLabel(
        table_frame,
        text="Diferenças de Média Ordenadas",
        font=ctk.CTkFont(size=16, weight="bold")
    ).pack(pady=10)
    
    # Create scrollable frame
    table_scroll = ctk.CTkScrollableFrame(table_frame, height=400)
    table_scroll.pack(fill="both", expand=True, padx=10, pady=(0, 10))
    
    # Headers
    headers = ["Comparação", "Diferença", "Erro Padrão", "F-Ratio", "p-Value", "IC Inferior", "IC Superior"]
    header_frame = ctk.CTkFrame(table_scroll)
    header_frame.pack(fill="x", pady=5)
    
    for col, header in enumerate(headers):
        ctk.CTkLabel(
            header_frame,
            text=header,
            font=ctk.CTkFont(weight="bold"),
            width=90
        ).grid(row=0, column=col, padx=3, pady=5)
    
    # Data rows
    mean_diff_data = result_data.get("meanDifference", {})
    
    for idx, (key, values) in enumerate(mean_diff_data.items()):
        row_frame = ctk.CTkFrame(table_scroll)
        row_frame.pack(fill="x", pady=2)
        
        # Color significant p-values
        is_significant = values.get("pValue", 1) < 0.05
        text_color = "red" if is_significant else None
        
        ctk.CTkLabel(row_frame, text=key, width=90).grid(row=0, column=0, padx=3, pady=3)
        ctk.CTkLabel(row_frame, text=f"{values['difference']:.4f}", width=90).grid(row=0, column=1, padx=3, pady=3)
        ctk.CTkLabel(row_frame, text=f"{values['stdErrorDifference']:.4f}", width=90).grid(row=0, column=2, padx=3, pady=3)
        ctk.CTkLabel(row_frame, text=f"{values['fRatio']:.4f}", width=90).grid(row=0, column=3, padx=3, pady=3)
        
        p_value_text = f"{values['pValue']:.4f}" if values['pValue'] >= 0.0001 else "< 0.0001*"
        ctk.CTkLabel(row_frame, text=p_value_text, width=90, text_color=text_color).grid(row=0, column=4, padx=3, pady=3)
        
        ctk.CTkLabel(row_frame, text=f"{values['ciInferior']:.4f}", width=90).grid(row=0, column=5, padx=3, pady=3)
        ctk.CTkLabel(row_frame, text=f"{values['ciSuperior']:.4f}", width=90).grid(row=0, column=6, padx=3, pady=3)


def display_mean_difference_chart(parent, result_data: Dict, response_name: str):
    """Display mean difference confidence interval chart"""
    chart_frame = ctk.CTkFrame(parent)
    chart_frame.pack(fill="both", expand=True)
    
    ctk.CTkLabel(
        chart_frame,
        text=f"Intervalos de Confiança - {response_name}",
        font=ctk.CTkFont(size=14, weight="bold")
    ).pack(pady=5)
    
    mean_diff_data = result_data.get("meanDifference", {})
    
    if not mean_diff_data:
        ctk.CTkLabel(chart_frame, text="Sem dados para exibir").pack(pady=20)
        return
    
    # Prepare data
    labels = []
    differences = []
    ci_lowers = []
    ci_uppers = []
    colors = []
    
    for key, values in mean_diff_data.items():
        labels.append(key)
        differences.append(values['difference'])
        ci_lowers.append(values['ciInferior'])
        ci_uppers.append(values['ciSuperior'])
        # Red if significant
        colors.append('red' if values['pValue'] < 0.05 else 'blue')
    
    # Create chart
    fig = Figure(figsize=(6, 5), dpi=100)
    ax = fig.add_subplot(111)
    
    y_positions = range(len(labels))
    
    # Plot CI lines and points
    for i, (label, diff, ci_low, ci_up, color) in enumerate(zip(labels, differences, ci_lowers, ci_uppers, colors)):
        # CI line
        ax.plot([ci_low, ci_up], [i, i], color=color, linewidth=2, alpha=0.6)
        # Mean point
        ax.plot(diff, i, 'o', color=color, markersize=8)
    
    # Reference line at zero
    ax.axvline(0, color='gray', linestyle='--', linewidth=1, alpha=0.7)
    
    ax.set_yticks(y_positions)
    ax.set_yticklabels(labels)
    ax.set_xlabel('Diferença', fontsize=10, fontweight='bold')
    ax.set_ylabel('Comparações', fontsize=10, fontweight='bold')
    ax.set_title('Intervalos de Confiança (95%)', fontsize=12, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='x')
    
    fig.tight_layout()
    
    canvas = FigureCanvasTkAgg(fig, chart_frame)
    canvas.draw()
    canvas.get_tk_widget().pack(fill="both", expand=True, padx=10, pady=10)
