"""
Process Capability Analysis Window
Allows user to perform Cp/Cpk and Pp/Ppk analysis on selected data
"""
import customtkinter as ctk
from tkinter import messagebox
from src.utils.lazy_imports import get_pandas, get_numpy, get_matplotlib_figure, get_matplotlib_backend, get_matplotlib, get_scipy_stats

from .capability_utils import (
    calculate_pp_ppk,
    calculate_cp_cpk,
    calculate_process_summary,
    calculate_rate,
    calculate_rate_inferior,
    calculate_rate_superior,
    calculate_se,
    calculate_inferior_limit,
    calculate_superior_limit,
    data_frame_split_by_columns,
    remove_last_column,
    calculate_ppk_not_normal,
    calculate_cpk_not_normal,
    fit_weibull,
)

# Lazy-loaded libraries (carregadas apenas quando window é criada)
_pd = None
_np = None
_plt = None
_Figure = None
_FigureCanvasTkAgg = None
_chi2 = None
_weibull_min = None
_norm = None

def _ensure_libs():
    """Garante que bibliotecas pesadas estão carregadas"""
    global _pd, _np, _plt, _Figure, _FigureCanvasTkAgg, _chi2, _weibull_min, _norm
    if _pd is None:
        _pd = get_pandas()
        _np = get_numpy()
        _plt = get_matplotlib()
        _Figure = get_matplotlib_figure()
        _FigureCanvasTkAgg = get_matplotlib_backend()
        stats = get_scipy_stats()
        _chi2 = stats.chi2
        _weibull_min = stats.weibull_min
        _norm = stats.norm
    return _pd, _np, _plt, _Figure, _FigureCanvasTkAgg, _chi2, _weibull_min, _norm


class CapabilityWindow(ctk.CTkToplevel):
    def __init__(self, parent, df):
        super().__init__(parent)
        
        # Carrega bibliotecas pesadas (lazy)
        pd, np, plt, Figure, FigureCanvasTkAgg, chi2, weibull_min, norm = _ensure_libs()
        self.pd = pd
        self.np = np
        self.plt = plt
        self.Figure = Figure
        self.FigureCanvasTkAgg = FigureCanvasTkAgg
        self.chi2 = chi2
        self.weibull_min = weibull_min
        self.norm = norm
        
        self.df = df
        self.results = None
        
        # Window configuration
        self.title("Análise de Capacidade de Processo")
        
        # Allow resizing
        self.resizable(True, True)
        self.minsize(800, 600)
        
        # Start maximized (full screen)
        self.state('zoomed')  # Windows maximized
        
        # Make modal
        self.transient(parent)
        self.grab_set()
        
        self.create_widgets()
    
    def create_widgets(self):
        # Main container with scrollable frame
        self.main_container = ctk.CTkScrollableFrame(self)
        self.main_container.pack(fill="both", expand=True, padx=20, pady=20)
        
        # Title
        title = ctk.CTkLabel(
            self.main_container,
            text="📊 Análise de Capacidade de Processo",
            font=ctk.CTkFont(size=24, weight="bold")
        )
        title.pack(pady=(0, 20))
        
        # Configuration Frame
        config_frame = ctk.CTkFrame(self.main_container)
        config_frame.pack(fill="x", pady=(0, 20))
        
        # Column Selection Section
        col_section = ctk.CTkFrame(config_frame)
        col_section.pack(fill="x", padx=20, pady=20)
        
        ctk.CTkLabel(
            col_section,
            text="Seleção de Colunas",
            font=ctk.CTkFont(size=16, weight="bold")
        ).pack(anchor="w", pady=(0, 10))
        
        # X Column (Phase) - Optional
        x_frame = ctk.CTkFrame(col_section, fg_color="transparent")
        x_frame.pack(fill="x", pady=5)
        
        ctk.CTkLabel(x_frame, text="Coluna X (Fase/Grupo) - Opcional:", width=200).pack(side="left", padx=(0, 10))
        
        columns_with_none = ["Nenhuma"] + list(self.df.columns)
        self.x_column_var = ctk.StringVar(value="Nenhuma")
        self.x_column_dropdown = ctk.CTkComboBox(
            x_frame,
            values=columns_with_none,
            variable=self.x_column_var,
            width=300,
            command=self.on_column_change
        )
        self.x_column_dropdown.pack(side="left")
        
        # Y Column (Response) - Required
        y_frame = ctk.CTkFrame(col_section, fg_color="transparent")
        y_frame.pack(fill="x", pady=5)
        
        ctk.CTkLabel(y_frame, text="Coluna Y (Resposta) *:", width=200).pack(side="left", padx=(0, 10))
        
        self.y_column_var = ctk.StringVar(value=self.df.columns[0] if len(self.df.columns) > 0 else "")
        self.y_column_dropdown = ctk.CTkComboBox(
            y_frame,
            values=list(self.df.columns),
            variable=self.y_column_var,
            width=300
        )
        self.y_column_dropdown.pack(side="left")
        
        # Analysis Type Section
        analysis_section = ctk.CTkFrame(config_frame)
        analysis_section.pack(fill="x", padx=20, pady=(0, 20))
        
        ctk.CTkLabel(
            analysis_section,
            text="Tipo de Análise",
            font=ctk.CTkFont(size=16, weight="bold")
        ).pack(anchor="w", pady=(0, 10))
        
        # Distribution Type
        dist_frame = ctk.CTkFrame(analysis_section, fg_color="transparent")
        dist_frame.pack(fill="x", pady=5)
        
        ctk.CTkLabel(dist_frame, text="Distribuição dos Dados:", width=200).pack(side="left", padx=(0, 10))
        
        self.distribution_var = ctk.StringVar(value="normal")
        
        radio_container = ctk.CTkFrame(dist_frame, fg_color="transparent")
        radio_container.pack(side="left")
        
        ctk.CTkRadioButton(
            radio_container,
            text="Normal",
            variable=self.distribution_var,
            value="normal"
        ).pack(side="left", padx=10)
        
        ctk.CTkRadioButton(
            radio_container,
            text="Não Normal",
            variable=self.distribution_var,
            value="not_normal"
        ).pack(side="left", padx=10)
        
        # Tolerance Type
        tolerance_frame = ctk.CTkFrame(analysis_section, fg_color="transparent")
        tolerance_frame.pack(fill="x", pady=5)
        
        ctk.CTkLabel(tolerance_frame, text="Tipo de Tolerância:", width=200).pack(side="left", padx=(0, 10))
        
        self.tolerance_var = ctk.StringVar(value="bilateral")
        self.tolerance_dropdown = ctk.CTkComboBox(
            tolerance_frame,
            values=["bilateral", "superiorUnilateral", "inferiorUnilateral"],
            variable=self.tolerance_var,
            width=300,
            command=self.on_tolerance_change
        )
        self.tolerance_dropdown.pack(side="left")
        
        # Specification Limits Section
        limits_section = ctk.CTkFrame(config_frame)
        limits_section.pack(fill="x", padx=20, pady=(0, 20))
        
        ctk.CTkLabel(
            limits_section,
            text="Limites de Especificação",
            font=ctk.CTkFont(size=16, weight="bold")
        ).pack(anchor="w", pady=(0, 10))
        
        # LSE (Upper Spec Limit)
        lse_frame = ctk.CTkFrame(limits_section, fg_color="transparent")
        lse_frame.pack(fill="x", pady=5)
        
        self.lse_label = ctk.CTkLabel(lse_frame, text="LSE (Limite Superior) *:", width=200)
        self.lse_label.pack(side="left", padx=(0, 10))
        
        self.lse_entry = ctk.CTkEntry(lse_frame, width=300, placeholder_text="Ex: 100.0")
        self.lse_entry.pack(side="left")
        
        # LIE (Lower Spec Limit)
        lie_frame = ctk.CTkFrame(limits_section, fg_color="transparent")
        lie_frame.pack(fill="x", pady=5)
        
        self.lie_label = ctk.CTkLabel(lie_frame, text="LIE (Limite Inferior) *:", width=200)
        self.lie_label.pack(side="left", padx=(0, 10))
        
        self.lie_entry = ctk.CTkEntry(lie_frame, width=300, placeholder_text="Ex: 90.0")
        self.lie_entry.pack(side="left")
        
        # Action Buttons
        button_frame = ctk.CTkFrame(config_frame, fg_color="transparent")
        button_frame.pack(fill="x", padx=20, pady=20)
        
        self.calculate_btn = ctk.CTkButton(
            button_frame,
            text="🔍 Calcular Capacidade",
            command=self.calculate_capability,
            height=40,
            font=ctk.CTkFont(size=14, weight="bold"),
            fg_color="#2E86DE"
        )
        self.calculate_btn.pack(side="left", padx=10)
        
        self.clear_btn = ctk.CTkButton(
            button_frame,
            text="🗑️ Limpar Resultados",
            command=self.clear_results,
            height=40,
            font=ctk.CTkFont(size=14),
            fg_color="#95A5A6"
        )
        self.clear_btn.pack(side="left", padx=10)
        
        # Results Container (initially empty)
        self.results_container = ctk.CTkFrame(self.main_container)
        self.results_container.pack(fill="both", expand=True, pady=(0, 20))
        
        # Initialize tolerance state
        self.on_tolerance_change(self.tolerance_var.get())
    
    def on_tolerance_change(self, value):
        """Update limit fields based on tolerance type"""
        if value == "superiorUnilateral":
            self.lse_label.configure(text="LSE (Limite Superior) *:")
            self.lie_label.configure(text="LIE (Limite Inferior):")
            self.lse_entry.configure(state="normal")
            self.lie_entry.configure(state="disabled")
            self.lie_entry.delete(0, "end")
            self.lie_entry.insert(0, "0")
        elif value == "inferiorUnilateral":
            self.lse_label.configure(text="LSE (Limite Superior):")
            self.lie_label.configure(text="LIE (Limite Inferior) *:")
            self.lse_entry.configure(state="disabled")
            self.lie_entry.configure(state="normal")
            self.lse_entry.delete(0, "end")
            self.lse_entry.insert(0, "999999")
        else:  # bilateral
            self.lse_label.configure(text="LSE (Limite Superior) *:")
            self.lie_label.configure(text="LIE (Limite Inferior) *:")
            self.lse_entry.configure(state="normal")
            self.lie_entry.configure(state="normal")
    
    def on_column_change(self, value):
        """Handle X column selection change"""
        pass
    
    def validate_inputs(self):
        """Validate user inputs before calculation"""
        # Check Y column
        y_col = self.y_column_var.get()
        if not y_col or y_col not in self.df.columns:
            messagebox.showerror("Erro", "Selecione uma coluna Y válida!")
            return False
        
        # Check limits based on tolerance type
        tolerance = self.tolerance_var.get()
        
        try:
            lse = float(self.lse_entry.get()) if self.lse_entry.get() else None
            lie = float(self.lie_entry.get()) if self.lie_entry.get() else None
            
            if tolerance == "bilateral":
                if lse is None or lie is None:
                    messagebox.showerror("Erro", "Para análise bilateral, informe LSE e LIE!")
                    return False
                if lse <= lie:
                    messagebox.showerror("Erro", "LSE deve ser maior que LIE!")
                    return False
            elif tolerance == "superiorUnilateral":
                if lse is None:
                    messagebox.showerror("Erro", "Para análise unilateral superior, informe LSE!")
                    return False
            elif tolerance == "inferiorUnilateral":
                if lie is None:
                    messagebox.showerror("Erro", "Para análise unilateral inferior, informe LIE!")
                    return False
        except ValueError:
            messagebox.showerror("Erro", "Os limites devem ser números válidos!")
            return False
        
        return True
    
    def calculate_capability(self):
        """Main calculation function - routes to appropriate analysis"""
        if not self.validate_inputs():
            return
        
        self.calculate_btn.configure(state="disabled", text="⏳ Calculando...")
        self.update()
        
        try:
            # Get configuration
            x_col = self.x_column_var.get()
            y_col = self.y_column_var.get()
            distribution = self.distribution_var.get()
            tolerance = self.tolerance_var.get()
            lse = float(self.lse_entry.get())
            lie = float(self.lie_entry.get())
            
            has_phase = x_col != "Nenhuma" and x_col in self.df.columns
            
            # Prepare data
            if has_phase:
                # Analysis with phase
                analysis_df = self.df[[x_col, y_col]].copy()
            else:
                # Analysis without phase
                analysis_df = self.df[[y_col]].copy()
            
            # Remove any NaN values
            analysis_df = analysis_df.dropna()
            
            if len(analysis_df) < 2:
                messagebox.showerror("Erro", "Dados insuficientes para análise (mínimo 2 valores)!")
                return
            
            # Route to appropriate calculation
            if has_phase:
                if distribution == "normal":
                    self.results = self.calculate_normal_with_phase(analysis_df, lse, lie, tolerance)
                else:
                    self.results = self.calculate_not_normal_with_phase(analysis_df, lse, lie, tolerance)
            else:
                if distribution == "normal":
                    self.results = self.calculate_normal(analysis_df, lse, lie, tolerance)
                else:
                    self.results = self.calculate_not_normal(analysis_df, lse, lie, tolerance)
            
            # Display results
            self.display_results()
            
        except Exception as e:
            messagebox.showerror("Erro no Cálculo", f"Erro ao calcular capacidade:\n{str(e)}")
            print(f"Capability calculation error: {e}")
            import traceback
            traceback.print_exc()
        finally:
            self.calculate_btn.configure(state="normal", text="🔍 Calcular Capacidade")
    
    def calculate_normal(self, df, lse: float, lie: float, tolerance_type: str):
        """Calculate capability for normal distribution without phase"""
        # Calcula PP e PPK
        pp, ppk, ppu, ppl = calculate_pp_ppk(lse, lie, df)
        
        # Calcula o Process Summary
        mean, within_sigma, overall_sigma, stability = calculate_process_summary(df)
        
        # Calcula CP e CPK
        cp, cpk, cpl, cpu = calculate_cp_cpk(lse, lie, df, within_sigma)
        
        observations = df.shape[0]
        
        # Calculate I-MR Chart data
        y_values = df.iloc[:, 0].tolist()
        mr_data = self.calculate_moving_range_data(y_values, mean, within_sigma, lse, lie)
        
        # Confidence intervals
        pp_lower = calculate_inferior_limit(pp, observations)
        pp_upper = calculate_superior_limit(pp, observations)
        ppk_lower = calculate_inferior_limit(ppk, observations)
        ppk_upper = calculate_superior_limit(ppk, observations)
        
        degress = observations - 1
        quiqua_0975 = self.chi2.ppf(0.975, degress)
        quiqua_0025 = self.chi2.ppf(0.025, degress)
        se = calculate_se(observations, within_sigma, lie, lse, mean)
        
        cp_lower = cp * ((observations - 1) / quiqua_0975) ** 0.5
        cp_upper = cp_lower * ((observations - 1) / quiqua_0025) ** 0.5
        
        cpu_lower = cpu - 1.96 * se
        cpl_lower = cpl - 1.96 * se
        cpu_upper = cpu + 1.96 * se
        cpl_upper = cpl + 1.96 * se
        
        cpk_lower = cpu_lower if cpl > cpu else cpl_lower
        cpk_upper = cpu_upper if cpl > cpu else cpl_upper
        
        # Calculate PPM rates
        rate_cpu = calculate_rate_superior(lse, mean, within_sigma)
        rate_cpl = calculate_rate_inferior(lie, mean, within_sigma)
        rate_cp = calculate_rate(lse, lie, mean, within_sigma)
        rate_cpk = rate_cpu if cpu > cpl else rate_cpl
        
        rate_ppu = calculate_rate_superior(lse, mean, overall_sigma)
        rate_ppl = calculate_rate_inferior(lie, mean, overall_sigma)
        rate_pp = calculate_rate(lse, lie, mean, overall_sigma)
        rate_ppk = rate_ppu if ppu > ppl else rate_ppl
        
        return {
            "type": "normal",
            "phase": False,
            "cp": cp,
            "cpk": cpk,
            "cpu": cpu,
            "cpl": cpl,
            "pp": pp,
            "ppk": ppk,
            "ppu": ppu,
            "ppl": ppl,
            "cp_lower": cp_lower,
            "cp_upper": cp_upper,
            "cpk_lower": cpk_lower,
            "cpk_upper": cpk_upper,
            "pp_lower": pp_lower,
            "pp_upper": pp_upper,
            "ppk_lower": ppk_lower,
            "ppk_upper": ppk_upper,
            "mean": mean,
            "within_sigma": within_sigma,
            "overall_sigma": overall_sigma,
            "stability": stability,
            "observations": observations,
            "rate_cp": rate_cp,
            "rate_cpk": rate_cpk,
            "rate_cpu": rate_cpu,
            "rate_cpl": rate_cpl,
            "rate_pp": rate_pp,
            "rate_ppk": rate_ppk,
            "rate_ppu": rate_ppu,
            "rate_ppl": rate_ppl,
            "y_values": y_values,
            "lse": lse,
            "lie": lie,
            "tolerance_type": tolerance_type,
            "mr_data": mr_data
        }
    
    def calculate_not_normal(self, df, lse: float, lie: float, tolerance_type: str):
        """Calculate capability for non-normal distribution without phase"""
        col = df.iloc[:, 0]
        
        P99865 = self.np.percentile(col, 99.865)
        P000135 = self.np.percentile(col, 0.135)
        P50 = self.np.percentile(col, 50)
        
        mean, within_sigma, overall_sigma, stability = calculate_process_summary(df)
        
        cp = (lse - lie) / (P99865 - P000135)
        cpu = (lse - P50) / (P99865 - P50)
        cpl = (P50 - lie) / (P50 - P000135)
        
        # Fit Weibull distribution
        shape, scale = fit_weibull(col.tolist())
        
        # Calculate PPM
        if tolerance_type == "superiorUnilateral":
            ppm = (1 - self.weibull_min.cdf(lse, c=shape, scale=scale)) * 1000000
        elif tolerance_type == "inferiorUnilateral":
            ppm = self.weibull_min.cdf(lie, c=shape, scale=scale) * 1000000
        else:
            ppm_upper = (1 - self.weibull_min.cdf(lse, c=shape, scale=scale)) * 1000000
            ppm_lower = self.weibull_min.cdf(lie, c=shape, scale=scale) * 1000000
            ppm = ppm_upper + ppm_lower
        
        observations = df.shape[0]
        y_values = col.tolist()
        
        return {
            "type": "not_normal",
            "phase": False,
            "pp": cp,
            "ppu": cpu,
            "ppl": cpl,
            "mean": mean,
            "within_sigma": within_sigma,
            "overall_sigma": overall_sigma,
            "stability": stability,
            "ppm": ppm,
            "observations": observations,
            "y_values": y_values,
            "lse": lse,
            "lie": lie,
            "tolerance_type": tolerance_type,
            "P50": P50,
            "P99865": P99865,
            "P000135": P000135
        }
    
    def calculate_normal_with_phase(self, df, lse: float, lie: float, tolerance_type: str):
        """Calculate capability for normal distribution with phase"""
        # Split by phase
        split_dfs = data_frame_split_by_columns(df)
        
        results_by_phase = {}
        
        for split_df in split_dfs:
            split_df = self.pd.DataFrame(split_df)
            first_value = str(split_df.iloc[0, 0])
            phase_col = split_df.columns[0]
            
            # Remove phase column
            remove_last_column(split_df, phase_col)
            
            # Calculate for this phase
            pp, ppk, ppu, ppl = calculate_pp_ppk(lse, lie, split_df)
            mean, within_sigma, overall_sigma, stability = calculate_process_summary(split_df)
            cp, cpk, cpl, cpu = calculate_cp_cpk(lse, lie, split_df, within_sigma)
            
            observations = split_df.shape[0]
            y_values = split_df.iloc[:, 0].tolist()
            
            # Calculate rates
            rate_cpu = calculate_rate_superior(lse, mean, within_sigma)
            rate_cpl = calculate_rate_inferior(lie, mean, within_sigma)
            rate_cp = calculate_rate(lse, lie, mean, within_sigma)
            rate_cpk = rate_cpu if cpu > cpl else rate_cpl
            
            phase_key = f"{phase_col}_{first_value}"
            results_by_phase[phase_key] = {
                "cp": cp,
                "cpk": cpk,
                "cpu": cpu,
                "cpl": cpl,
                "pp": pp,
                "ppk": ppk,
                "ppu": ppu,
                "ppl": ppl,
                "mean": mean,
                "within_sigma": within_sigma,
                "overall_sigma": overall_sigma,
                "stability": stability,
                "observations": observations,
                "rate_cp": rate_cp,
                "rate_cpk": rate_cpk,
                "y_values": y_values
            }
        
        return {
            "type": "normal",
            "phase": True,
            "phases": results_by_phase,
            "lse": lse,
            "lie": lie,
            "tolerance_type": tolerance_type
        }
    
    def calculate_not_normal_with_phase(self, df, lse: float, lie: float, tolerance_type: str):
        """Calculate capability for non-normal distribution with phase"""
        split_dfs = data_frame_split_by_columns(df)
        
        results_by_phase = {}
        
        for split_df in split_dfs:
            split_df = self.pd.DataFrame(split_df)
            first_value = str(split_df.iloc[0, 0])
            phase_col = split_df.columns[0]
            
            remove_last_column(split_df, phase_col)
            
            col = split_df.iloc[:, 0]
            P50 = self.np.percentile(col, 50)
            P99865 = self.np.percentile(col, 99.865)
            P000135 = self.np.percentile(col, 0.135)
            
            mean, within_sigma, overall_sigma, stability = calculate_process_summary(split_df)
            
            ppu, ppl, pp, ppk_value = calculate_ppk_not_normal(
                lse, lie, tolerance_type, P50, P99865, P000135, overall_sigma
            )
            
            cp, cpu, cpl, cpk_value = calculate_cpk_not_normal(
                lse, lie, tolerance_type, mean, within_sigma
            )
            
            # Fit Weibull
            shape, scale = fit_weibull(col.tolist())
            
            if tolerance_type == "superiorUnilateral":
                ppm = (1 - self.weibull_min.cdf(lse, c=shape, scale=scale)) * 1000000
            elif tolerance_type == "inferiorUnilateral":
                ppm = self.weibull_min.cdf(lie, c=shape, scale=scale) * 1000000
            else:
                ppm_upper = (1 - self.weibull_min.cdf(lse, c=shape, scale=scale)) * 1000000
                ppm_lower = self.weibull_min.cdf(lie, c=shape, scale=scale) * 1000000
                ppm = ppm_upper + ppm_lower
            
            phase_key = f"{phase_col}_{first_value}"
            results_by_phase[phase_key] = {
                "cp": cp,
                "cpk": cpk_value,
                "cpu": cpu,
                "cpl": cpl,
                "pp": pp,
                "ppk": ppk_value,
                "ppu": ppu,
                "ppl": ppl,
                "mean": mean,
                "within_sigma": within_sigma,
                "overall_sigma": overall_sigma,
                "stability": stability,
                "ppm": ppm,
                "y_values": col.tolist()
            }
        
        return {
            "type": "not_normal",
            "phase": True,
            "phases": results_by_phase,
            "lse": lse,
            "lie": lie,
            "tolerance_type": tolerance_type
        }
    
    def display_results(self):
        """Display calculation results with charts and tables"""
        # Clear previous results
        for widget in self.results_container.winfo_children():
            widget.destroy()
        
        if not self.results:
            return
        
        # Results title
        title = ctk.CTkLabel(
            self.results_container,
            text="📈 Resultados da Análise",
            font=ctk.CTkFont(size=20, weight="bold")
        )
        title.pack(pady=(10, 20))
        
        if self.results.get("phase"):
            self.display_phase_results()
        else:
            self.display_single_results()
    
    def display_single_results(self):
        """Display results for analysis without phase"""
        # First row - Control Charts (I-MR)
        if self.results["type"] == "normal" and "mr_data" in self.results:
            charts_row = ctk.CTkFrame(self.results_container)
            charts_row.pack(fill="both", expand=True, padx=10, pady=(0, 10))
            
            # Moving Range Chart (left)
            mr_frame = ctk.CTkFrame(charts_row)
            mr_frame.pack(side="left", fill="both", expand=True, padx=(0, 5))
            self.display_moving_range_chart(mr_frame)
            
            # Individual Chart (right)
            i_frame = ctk.CTkFrame(charts_row)
            i_frame.pack(side="left", fill="both", expand=True, padx=(5, 0))
            self.display_individual_chart(i_frame)
        
        # Second row - Tables and Analysis Charts
        content_frame = ctk.CTkFrame(self.results_container)
        content_frame.pack(fill="both", expand=True, padx=10, pady=10)
        
        # Left column - Tables
        left_frame = ctk.CTkFrame(content_frame)
        left_frame.pack(side="left", fill="both", expand=True, padx=(0, 10))
        
        # Right column - Charts
        right_frame = ctk.CTkFrame(content_frame)
        right_frame.pack(side="left", fill="both", expand=True)
        
        # Display tables
        self.display_process_summary_table(left_frame)
        if self.results["type"] == "normal":
            self.display_within_sigma_table(left_frame)
            self.display_overall_sigma_table(left_frame)
        else:
            self.display_overall_sigma_table_not_normal(left_frame)
        
        # Display charts
        self.display_histogram_chart(right_frame)
        self.display_capability_chart(right_frame)
    
    def display_phase_results(self):
        """Display results for analysis with phase (multiple groups)"""
        phases = self.results.get("phases", {})
        
        # Create tabs for each phase
        tabview = ctk.CTkTabview(self.results_container)
        tabview.pack(fill="both", expand=True, padx=10, pady=10)
        
        for phase_name, phase_data in phases.items():
            # Clean phase name for tab
            display_name = phase_name.replace("_", " ")
            tab = tabview.add(display_name)
            
            # Create layout for this phase
            content_frame = ctk.CTkFrame(tab)
            content_frame.pack(fill="both", expand=True, padx=10, pady=10)
            
            left_frame = ctk.CTkFrame(content_frame)
            left_frame.pack(side="left", fill="both", expand=True, padx=(0, 10))
            
            right_frame = ctk.CTkFrame(content_frame)
            right_frame.pack(side="left", fill="both", expand=True)
            
            # Display tables and charts for this phase
            self.display_capability_table_phase(left_frame, phase_data)
            self.display_process_summary_table_phase(left_frame, phase_data)
            self.display_histogram_chart_phase(right_frame, phase_data, phase_name)
    
    def display_within_sigma_table(self, parent):
        """Display Within Sigma Capability table"""
        table_frame = ctk.CTkFrame(parent)
        table_frame.pack(fill="x", pady=(0, 20))
        
        ctk.CTkLabel(
            table_frame,
            text="Within Sigma Capability",
            font=ctk.CTkFont(size=16, weight="bold")
        ).pack(pady=10)
        
        # Create table
        table_data_frame = ctk.CTkFrame(table_frame)
        table_data_frame.pack(fill="x", padx=10, pady=(0, 10))
        
        # Header
        headers = ["Termo", "Estimativa", "IC Inferior", "IC Superior", "PPM"]
        for col, header in enumerate(headers):
            ctk.CTkLabel(
                table_data_frame,
                text=header,
                font=ctk.CTkFont(weight="bold"),
                width=80
            ).grid(row=0, column=col, padx=5, pady=5, sticky="ew")
        
        # Data rows - reversed order (Cpk, Cpl, Cpu, Cp)
        indices = [
            ("Cpk", "cpk", "cpk_lower", "cpk_upper", "rate_cpk"),
            ("Cpl", "cpl", None, None, "rate_cpl"),
            ("Cpu", "cpu", None, None, "rate_cpu"),
            ("Cp", "cp", "cp_lower", "cp_upper", "rate_cp")
        ]
        
        for row, (label, val_key, lower_key, upper_key, rate_key) in enumerate(indices, start=1):
            ctk.CTkLabel(table_data_frame, text=label, width=80).grid(row=row, column=0, padx=5, pady=3)
            ctk.CTkLabel(table_data_frame, text=f"{self.results[val_key]:.4f}", width=80).grid(row=row, column=1, padx=5, pady=3)
            
            # IC Inferior
            if lower_key and lower_key in self.results:
                ctk.CTkLabel(table_data_frame, text=f"{self.results[lower_key]:.4f}", width=80).grid(row=row, column=2, padx=5, pady=3)
            else:
                ctk.CTkLabel(table_data_frame, text="-", width=80).grid(row=row, column=2, padx=5, pady=3)
            
            # IC Superior
            if upper_key and upper_key in self.results:
                ctk.CTkLabel(table_data_frame, text=f"{self.results[upper_key]:.4f}", width=80).grid(row=row, column=3, padx=5, pady=3)
            else:
                ctk.CTkLabel(table_data_frame, text="-", width=80).grid(row=row, column=3, padx=5, pady=3)
            
            # PPM
            if rate_key and rate_key in self.results:
                ppm_val = int(self.results[rate_key])
                ctk.CTkLabel(table_data_frame, text=f"{ppm_val:,}", width=80).grid(row=row, column=4, padx=5, pady=3)
            else:
                ctk.CTkLabel(table_data_frame, text="-", width=80).grid(row=row, column=4, padx=5, pady=3)
    
    def display_overall_sigma_table(self, parent):
        """Display Overall Sigma Capability table"""
        table_frame = ctk.CTkFrame(parent)
        table_frame.pack(fill="x", pady=(0, 20))
        
        ctk.CTkLabel(
            table_frame,
            text="Overall Sigma Capability",
            font=ctk.CTkFont(size=16, weight="bold")
        ).pack(pady=10)
        
        # Create table
        table_data_frame = ctk.CTkFrame(table_frame)
        table_data_frame.pack(fill="x", padx=10, pady=(0, 10))
        
        # Header
        headers = ["Termo", "Estimativa", "IC Inferior", "IC Superior", "PPM"]
        for col, header in enumerate(headers):
            ctk.CTkLabel(
                table_data_frame,
                text=header,
                font=ctk.CTkFont(weight="bold"),
                width=80
            ).grid(row=0, column=col, padx=5, pady=5, sticky="ew")
        
        # Data rows - reversed order (Ppk, Ppl, Ppu, Pp)
        indices = [
            ("Ppk", "ppk", "ppk_lower", "ppk_upper", "rate_ppk"),
            ("Ppl", "ppl", None, None, "rate_ppl"),
            ("Ppu", "ppu", None, None, "rate_ppu"),
            ("Pp", "pp", "pp_lower", "pp_upper", "rate_pp")
        ]
        
        for row, (label, val_key, lower_key, upper_key, rate_key) in enumerate(indices, start=1):
            ctk.CTkLabel(table_data_frame, text=label, width=80).grid(row=row, column=0, padx=5, pady=3)
            ctk.CTkLabel(table_data_frame, text=f"{self.results[val_key]:.4f}", width=80).grid(row=row, column=1, padx=5, pady=3)
            
            # IC Inferior
            if lower_key and lower_key in self.results:
                ctk.CTkLabel(table_data_frame, text=f"{self.results[lower_key]:.4f}", width=80).grid(row=row, column=2, padx=5, pady=3)
            else:
                ctk.CTkLabel(table_data_frame, text="-", width=80).grid(row=row, column=3, padx=5, pady=3)
            
            # IC Superior
            if upper_key and upper_key in self.results:
                ctk.CTkLabel(table_data_frame, text=f"{self.results[upper_key]:.4f}", width=80).grid(row=row, column=3, padx=5, pady=3)
            else:
                ctk.CTkLabel(table_data_frame, text="-", width=80).grid(row=row, column=3, padx=5, pady=3)
            
            # PPM
            if rate_key and rate_key in self.results:
                ppm_val = int(self.results[rate_key])
                ctk.CTkLabel(table_data_frame, text=f"{ppm_val:,}", width=80).grid(row=row, column=4, padx=5, pady=3)
            else:
                ctk.CTkLabel(table_data_frame, text="-", width=80).grid(row=row, column=4, padx=5, pady=3)
    
    def display_overall_sigma_table_not_normal(self, parent):
        """Display Overall Sigma Capability table for non-normal distribution"""
        table_frame = ctk.CTkFrame(parent)
        table_frame.pack(fill="x", pady=(0, 20))
        
        ctk.CTkLabel(
            table_frame,
            text="Overall Sigma Capability",
            font=ctk.CTkFont(size=16, weight="bold")
        ).pack(pady=10)
        
        # Create table
        table_data_frame = ctk.CTkFrame(table_frame)
        table_data_frame.pack(fill="x", padx=10, pady=(0, 10))
        
        # Header
        headers = ["Termo", "Estimativa", "PPM"]
        for col, header in enumerate(headers):
            ctk.CTkLabel(
                table_data_frame,
                text=header,
                font=ctk.CTkFont(weight="bold"),
                width=120
            ).grid(row=0, column=col, padx=5, pady=5, sticky="ew")
        
        # Filter based on tolerance type
        tolerance = self.results["tolerance_type"]
        indices = []
        
        if tolerance == "bilateral":
            indices = [
                ("Pp", "pp", True),
                ("Ppu", "ppu", False),
                ("Ppl", "ppl", False)
            ]
        elif tolerance == "superiorUnilateral":
            indices = [("Ppu", "ppu", True)]
        elif tolerance == "inferiorUnilateral":
            indices = [("Ppl", "ppl", True)]
        
        for row, (label, val_key, show_ppm) in enumerate(indices, start=1):
            ctk.CTkLabel(table_data_frame, text=label, width=120).grid(row=row, column=0, padx=5, pady=3)
            ctk.CTkLabel(table_data_frame, text=f"{self.results[val_key]:.4f}", width=120).grid(row=row, column=1, padx=5, pady=3)
            
            # PPM - only show for first row or specific tolerance types
            if show_ppm:
                ppm_val = int(self.results["ppm"])
                ctk.CTkLabel(table_data_frame, text=f"{ppm_val:,}", width=120).grid(row=row, column=2, padx=5, pady=3)
            else:
                ctk.CTkLabel(table_data_frame, text="-", width=120).grid(row=row, column=2, padx=5, pady=3)
    
    def display_process_summary_table(self, parent):
        """Display process summary statistics table"""
        table_frame = ctk.CTkFrame(parent)
        table_frame.pack(fill="x", pady=(0, 20))
        
        ctk.CTkLabel(
            table_frame,
            text="Resumo do Processo",
            font=ctk.CTkFont(size=16, weight="bold")
        ).pack(pady=10)
        
        table_data_frame = ctk.CTkFrame(table_frame)
        table_data_frame.pack(fill="x", padx=10, pady=(0, 10))
        
        # Header
        ctk.CTkLabel(
            table_data_frame,
            text="Métrica",
            font=ctk.CTkFont(weight="bold"),
            width=200
        ).grid(row=0, column=0, padx=5, pady=5, sticky="w")
        
        ctk.CTkLabel(
            table_data_frame,
            text="Valor",
            font=ctk.CTkFont(weight="bold"),
            width=150
        ).grid(row=0, column=1, padx=5, pady=5, sticky="w")
        
        # Data
        metrics = [
            ("Média", f"{self.results['mean']:.4f}"),
            ("Sigma Within", f"{self.results['within_sigma']:.4f}"),
            ("Sigma Overall", f"{self.results['overall_sigma']:.4f}"),
            ("Estabilidade", f"{self.results['stability']:.4f}"),
            ("Observações", f"{self.results['observations']}")
        ]
        
        if self.results["type"] == "not_normal":
            metrics.append(("PPM (Defeitos)", f"{self.results['ppm']:.0f}"))
        
        for row, (label, value) in enumerate(metrics, start=1):
            ctk.CTkLabel(table_data_frame, text=label, width=200).grid(row=row, column=0, padx=5, pady=3, sticky="w")
            ctk.CTkLabel(table_data_frame, text=value, width=150).grid(row=row, column=1, padx=5, pady=3, sticky="w")
    
    def display_capability_table_phase(self, parent, phase_data):
        """Display capability table for a specific phase"""
        table_frame = ctk.CTkFrame(parent)
        table_frame.pack(fill="x", pady=(0, 20))
        
        ctk.CTkLabel(
            table_frame,
            text="Índices de Capacidade",
            font=ctk.CTkFont(size=16, weight="bold")
        ).pack(pady=10)
        
        table_data_frame = ctk.CTkFrame(table_frame)
        table_data_frame.pack(fill="x", padx=10, pady=(0, 10))
        
        headers = ["Índice", "Valor"]
        for col, header in enumerate(headers):
            ctk.CTkLabel(
                table_data_frame,
                text=header,
                font=ctk.CTkFont(weight="bold"),
                width=150
            ).grid(row=0, column=col, padx=5, pady=5, sticky="ew")
        
        indices = [
            ("Cp", "cp"),
            ("Cpk", "cpk"),
            ("Pp", "pp"),
            ("Ppk", "ppk")
        ]
        
        for row, (label, val_key) in enumerate(indices, start=1):
            if val_key in phase_data:
                ctk.CTkLabel(table_data_frame, text=label, width=150).grid(row=row, column=0, padx=5, pady=3)
                ctk.CTkLabel(table_data_frame, text=f"{phase_data[val_key]:.4f}", width=150).grid(row=row, column=1, padx=5, pady=3)
    
    def display_process_summary_table_phase(self, parent, phase_data):
        """Display process summary for a specific phase"""
        table_frame = ctk.CTkFrame(parent)
        table_frame.pack(fill="x", pady=(0, 20))
        
        ctk.CTkLabel(
            table_frame,
            text="Resumo do Processo",
            font=ctk.CTkFont(size=16, weight="bold")
        ).pack(pady=10)
        
        table_data_frame = ctk.CTkFrame(table_frame)
        table_data_frame.pack(fill="x", padx=10, pady=(0, 10))
        
        ctk.CTkLabel(
            table_data_frame,
            text="Métrica",
            font=ctk.CTkFont(weight="bold"),
            width=200
        ).grid(row=0, column=0, padx=5, pady=5, sticky="w")
        
        ctk.CTkLabel(
            table_data_frame,
            text="Valor",
            font=ctk.CTkFont(weight="bold"),
            width=150
        ).grid(row=0, column=1, padx=5, pady=5, sticky="w")
        
        metrics = [
            ("Média", f"{phase_data['mean']:.4f}"),
            ("Sigma Within", f"{phase_data['within_sigma']:.4f}"),
            ("Sigma Overall", f"{phase_data['overall_sigma']:.4f}"),
            ("Estabilidade", f"{phase_data['stability']:.4f}")
        ]
        
        if "ppm" in phase_data:
            metrics.append(("PPM (Defeitos)", f"{phase_data['ppm']:.0f}"))
        
        for row, (label, value) in enumerate(metrics, start=1):
            ctk.CTkLabel(table_data_frame, text=label, width=200).grid(row=row, column=0, padx=5, pady=3, sticky="w")
            ctk.CTkLabel(table_data_frame, text=value, width=150).grid(row=row, column=1, padx=5, pady=3, sticky="w")
    
    def display_moving_range_chart(self, parent):
        """Display Moving Range chart with control limits"""
        chart_frame = ctk.CTkFrame(parent)
        chart_frame.pack(fill="both", expand=True)
        
        ctk.CTkLabel(
            chart_frame,
            text="Carta de Amplitude Móvel (MR)",
            font=ctk.CTkFont(size=14, weight="bold")
        ).pack(pady=5)
        
        mr_data = self.results["mr_data"]
        moving_range = mr_data['moving_range']
        mr_bar = mr_data['mr_bar']
        ucl_mr = mr_data['ucl_mr']
        lcl_mr = mr_data['lcl_mr']
        
        fig = self.Figure(figsize=(6, 3.5), dpi=100)
        ax = fig.add_subplot(111)
        
        observations = list(range(1, len(moving_range) + 1))
        
        # Plot moving range
        ax.plot(observations, moving_range, 'o-', color='black', linewidth=1.5, markersize=4, label='MR')
        
        # Plot MR bar (center line)
        ax.axhline(mr_bar, color='green', linestyle='-', linewidth=2, label=f'MR̄ = {mr_bar:.4f}')
        
        # Plot control limits
        ax.axhline(ucl_mr, color='red', linestyle='--', linewidth=2, label=f'UCL = {ucl_mr:.4f}')
        ax.axhline(lcl_mr, color='red', linestyle='--', linewidth=2, label=f'LCL = {lcl_mr:.4f}')
        
        ax.set_xlabel('Observação')
        ax.set_ylabel('Amplitude Móvel')
        ax.legend(fontsize=8, loc='best')
        ax.grid(True, alpha=0.3)
        
        canvas = self.FigureCanvasTkAgg(fig, chart_frame)
        canvas.draw()
        canvas.get_tk_widget().pack(fill="both", expand=True, padx=10, pady=5)
    
    def display_individual_chart(self, parent):
        """Display Individual (I) chart with control and specification limits"""
        chart_frame = ctk.CTkFrame(parent)
        chart_frame.pack(fill="both", expand=True)
        
        ctk.CTkLabel(
            chart_frame,
            text="Carta de Medidas Individuais (I)",
            font=ctk.CTkFont(size=14, weight="bold")
        ).pack(pady=5)
        
        y_values = self.results["y_values"]
        mr_data = self.results["mr_data"]
        mean = mr_data['mean']
        ucl_i = mr_data['ucl_i']
        lcl_i = mr_data['lcl_i']
        lse = self.results["lse"]
        lie = self.results["lie"]
        
        fig = self.Figure(figsize=(6, 3.5), dpi=100)
        ax = fig.add_subplot(111)
        
        observations = list(range(1, len(y_values) + 1))
        
        # Plot individual values
        ax.plot(observations, y_values, 'o-', color='black', linewidth=1.5, markersize=4, label='Medidas')
        
        # Plot center line (mean)
        ax.axhline(mean, color='green', linestyle='-', linewidth=2, label=f'Média = {mean:.4f}')
        
        # Plot control limits
        ax.axhline(ucl_i, color='red', linestyle='--', linewidth=1.5, label=f'UCL = {ucl_i:.4f}')
        ax.axhline(lcl_i, color='red', linestyle='--', linewidth=1.5, label=f'LCL = {lcl_i:.4f}')
        
        # Plot specification limits
        ax.axhline(lse, color='blue', linestyle='--', linewidth=1.5, label=f'LSE = {lse:.2f}')
        ax.axhline(lie, color='blue', linestyle='--', linewidth=1.5, label=f'LIE = {lie:.2f}')
        
        ax.set_xlabel('Observação')
        ax.set_ylabel('Valor')
        ax.legend(fontsize=7, loc='best', ncol=2)
        ax.grid(True, alpha=0.3)
        
        canvas = self.FigureCanvasTkAgg(fig, chart_frame)
        canvas.draw()
        canvas.get_tk_widget().pack(fill="both", expand=True, padx=10, pady=5)
    
    def display_histogram_chart(self, parent):
        """Display histogram with normal curve and specification limits"""
        chart_frame = ctk.CTkFrame(parent)
        chart_frame.pack(fill="both", expand=True, pady=(0, 10))
        
        ctk.CTkLabel(
            chart_frame,
            text="Histograma com Limites de Especificação",
            font=ctk.CTkFont(size=14, weight="bold")
        ).pack(pady=5)
        
        # Create matplotlib figure
        fig = self.Figure(figsize=(6, 4), dpi=100)
        ax = fig.add_subplot(111)
        
        y_values = self.results["y_values"]
        lse = self.results["lse"]
        lie = self.results["lie"]
        
        # Histogram
        ax.hist(y_values, bins=20, density=True, alpha=0.6, color='skyblue', edgecolor='black')
        
        # Normal curve using Overall Sigma
        mu = self.results["mean"]
        sigma = self.results["overall_sigma"]
        x = self.np.linspace(min(y_values), max(y_values), 100)
        ax.plot(x, self.norm.pdf(x, mu, sigma), 'r-', linewidth=2, label='Distribuição Normal')
        
        # Specification limits
        ax.axvline(lse, color='red', linestyle='--', linewidth=2, label=f'LSE = {lse}')
        ax.axvline(lie, color='red', linestyle='--', linewidth=2, label=f'LIE = {lie}')
        ax.axvline(mu, color='green', linestyle='-', linewidth=2, label=f'Média = {mu:.2f}')
        
        ax.set_xlabel('Valores')
        ax.set_ylabel('Densidade')
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
        
        # Embed in tkinter
        canvas = self.FigureCanvasTkAgg(fig, chart_frame)
        canvas.draw()
        canvas.get_tk_widget().pack(fill="both", expand=True, padx=10, pady=10)
    
    def display_capability_chart(self, parent):
        """Display capability indices as bar chart"""
        chart_frame = ctk.CTkFrame(parent)
        chart_frame.pack(fill="both", expand=True)
        
        ctk.CTkLabel(
            chart_frame,
            text="Comparação de Índices",
            font=ctk.CTkFont(size=14, weight="bold")
        ).pack(pady=5)
        
        fig = self.Figure(figsize=(6, 4), dpi=100)
        ax = fig.add_subplot(111)
        
        if self.results["type"] == "normal":
            indices = ['Cp', 'Cpk', 'Pp', 'Ppk']
            values = [
                self.results['cp'],
                self.results['cpk'],
                self.results['pp'],
                self.results['ppk']
            ]
        else:
            indices = ['Pp', 'Ppu', 'Ppl']
            values = [
                self.results['pp'],
                self.results['ppu'],
                self.results['ppl']
            ]
        
        colors = ['green' if v >= 1.33 else 'orange' if v >= 1.0 else 'red' for v in values]
        
        bars = ax.bar(indices, values, color=colors, alpha=0.7, edgecolor='black')
        
        # Add reference lines
        ax.axhline(1.33, color='green', linestyle='--', linewidth=1, alpha=0.5, label='Excelente (≥1.33)')
        ax.axhline(1.0, color='orange', linestyle='--', linewidth=1, alpha=0.5, label='Aceitável (≥1.0)')
        
        # Add value labels on bars
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                    f'{height:.3f}',
                    ha='center', va='bottom', fontsize=9)
        
        ax.set_ylabel('Valor do Índice')
        ax.set_title('Índices de Capacidade do Processo')
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3, axis='y')
        
        canvas = self.FigureCanvasTkAgg(fig, chart_frame)
        canvas.draw()
        canvas.get_tk_widget().pack(fill="both", expand=True, padx=10, pady=10)
    
    def display_histogram_chart_phase(self, parent, phase_data, phase_name):
        """Display histogram for a specific phase"""
        chart_frame = ctk.CTkFrame(parent)
        chart_frame.pack(fill="both", expand=True, pady=(0, 10))
        
        ctk.CTkLabel(
            chart_frame,
            text="Histograma com Limites",
            font=ctk.CTkFont(size=14, weight="bold")
        ).pack(pady=5)
        
        fig = self.Figure(figsize=(6, 4), dpi=100)
        ax = fig.add_subplot(111)
        
        y_values = phase_data["y_values"]
        lse = self.results["lse"]
        lie = self.results["lie"]
        
        ax.hist(y_values, bins=15, density=True, alpha=0.6, color='skyblue', edgecolor='black')
        
        mu = phase_data["mean"]
        sigma = phase_data["overall_sigma"]
        x = self.np.linspace(min(y_values), max(y_values), 100)
        ax.plot(x, self.norm.pdf(x, mu, sigma), 'r-', linewidth=2, label='Distribuição')
        
        ax.axvline(lse, color='red', linestyle='--', linewidth=2, label=f'LSE = {lse}')
        ax.axvline(lie, color='red', linestyle='--', linewidth=2, label=f'LIE = {lie}')
        ax.axvline(mu, color='green', linestyle='-', linewidth=2, label=f'Média = {mu:.2f}')
        
        ax.set_xlabel('Valores')
        ax.set_ylabel('Densidade')
        ax.set_title(f'{phase_name}')
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
        
        canvas = self.FigureCanvasTkAgg(fig, chart_frame)
        canvas.draw()
        canvas.get_tk_widget().pack(fill="both", expand=True, padx=10, pady=10)
    
    def clear_results(self):
        """Clear all results"""
        for widget in self.results_container.winfo_children():
            widget.destroy()
        
        self.results = None

    
    def calculate_moving_range_data(self, y_values, mean, within_sigma, lse, lie):
        """Calculate data for I-MR charts (Individual and Moving Range)"""
        # Calculate Moving Range
        moving_range = []
        for i in range(len(y_values) - 1):
            mr = abs(y_values[i] - y_values[i + 1])
            moving_range.append(mr)
        
        # MRbar (average of moving ranges)
        mr_bar = self.np.mean(moving_range) if moving_range else 0
        
        # Constants for subgroup size = 2
        d2 = 1.128
        D3 = 0
        D4 = 3.267
        
        # Control limits for I chart (Individual values)
        ucl_i = mean + 3 * (mr_bar / d2)
        lcl_i = mean - 3 * (mr_bar / d2)
        
        # Control limits for MR chart
        ucl_mr = D4 * mr_bar
        lcl_mr = D3 * mr_bar
        
        return {
            'moving_range': moving_range,
            'mr_bar': mr_bar,
            'ucl_i': ucl_i,
            'lcl_i': lcl_i,
            'ucl_mr': ucl_mr,
            'lcl_mr': lcl_mr,
            'mean': mean
        }