"""
Descriptive Statistics analysis window.
Optional grouping (X/phase) and multiple Y targets.
Shows summary table plus histogram and boxplot.
"""
import customtkinter as ctk
from tkinter import ttk, messagebox
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg


class DescriptiveStatsWindow(ctk.CTkToplevel):
    """UI for descriptive statistics with optional grouping and multiple targets."""

    def __init__(self, parent, data: pd.DataFrame):
        super().__init__(parent)
        self.title("Descriptive Statistics")
        self.geometry("1200x800")
        self.minsize(900, 650)

        # Data setup
        self.df = data.copy()
        self.numeric_cols = list(self.df.select_dtypes(include="number").columns)
        self.group_options = ["None"] + list(self.df.columns)

        # State
        self.y_checks = {}
        self.group_var = ctk.StringVar(value="None")

        # Build UI
        self._create_widgets()

    def _create_widgets(self):
        """Build layout: left controls, right outputs."""
        main = ctk.CTkFrame(self)
        main.pack(fill="both", expand=True, padx=16, pady=16)

        left_container = ctk.CTkFrame(main, width=280)
        left_container.pack(side="left", fill="y")
        left_container.pack_propagate(False)
        
        left = ctk.CTkScrollableFrame(left_container)
        left.pack(fill="both", expand=True)

        # Right panel with scroll
        right_container = ctk.CTkFrame(main)
        right_container.pack(side="right", fill="both", expand=True, padx=(12, 0))
        
        right = ctk.CTkScrollableFrame(right_container)
        right.pack(fill="both", expand=True)

        # Header
        title = ctk.CTkLabel(left, text="Descriptive Statistics", font=ctk.CTkFont(size=18, weight="bold"))
        title.pack(pady=(12, 4))
        subtitle = ctk.CTkLabel(
            left,
            text="Select optional X (phase) and one or more Y columns.",
            font=ctk.CTkFont(size=12),
            text_color="gray"
        )
        subtitle.pack(pady=(0, 12), padx=8)

        # Group selection (optional X)
        group_label = ctk.CTkLabel(left, text="Optional X / Phase", font=ctk.CTkFont(size=12, weight="bold"))
        group_label.pack(anchor="w", padx=12)
        group_menu = ctk.CTkOptionMenu(left, values=self.group_options, variable=self.group_var, width=220)
        group_menu.pack(padx=12, pady=(4, 12))

        # Y selection (multiple)
        y_label = ctk.CTkLabel(left, text="Select Y columns", font=ctk.CTkFont(size=12, weight="bold"))
        y_label.pack(anchor="w", padx=12)

        y_scroll = ctk.CTkScrollableFrame(left, width=240, height=260)
        y_scroll.pack(padx=12, pady=(4, 8), fill="both", expand=True)

        if not self.numeric_cols:
            ctk.CTkLabel(y_scroll, text="No numeric columns found.", text_color="red").pack(pady=6)
        else:
            for col in self.numeric_cols:
                var = ctk.BooleanVar(value=False)
                cb = ctk.CTkCheckBox(y_scroll, text=col, variable=var)
                cb.pack(anchor="w", pady=2, padx=4)
                self.y_checks[col] = var

        btn_row = ctk.CTkFrame(left, fg_color="transparent")
        btn_row.pack(fill="x", padx=12, pady=(6, 12))

        select_all_btn = ctk.CTkButton(btn_row, text="Select all", command=self._select_all, height=28, fg_color="#95A5A6")
        select_all_btn.pack(side="left", expand=True, fill="x", padx=(0, 4))

        clear_btn = ctk.CTkButton(btn_row, text="Clear", command=self._clear_selection, height=28, fg_color="#7F8C8D")
        clear_btn.pack(side="right", expand=True, fill="x", padx=(4, 0))

        run_btn = ctk.CTkButton(left, text="Analyze", command=self._run_analysis, height=40, font=ctk.CTkFont(size=14, weight="bold"))
        run_btn.pack(fill="x", padx=12, pady=(4, 12))
        
        # Bottom spacer for footer margin
        ctk.CTkFrame(left, fg_color="transparent", height=20).pack()

        # Right side: stats table and plots
        table_frame = ctk.CTkFrame(right)
        table_frame.pack(fill="x", padx=8, pady=(8, 6))

        cols = ["Group", "Column", "Count", "Mean", "Std", "Min", "P25", "Median", "P75", "Max", "IQR", "Skew", "Kurtosis", "Missing"]
        self.tree = ttk.Treeview(table_frame, columns=cols, show="headings", height=8)
        for col in cols:
            self.tree.heading(col, text=col)
            self.tree.column(col, anchor="center", stretch=True, width=90)
        self.tree.pack(fill="x", padx=6, pady=6)

        # Plots
        self.plot_frame = ctk.CTkFrame(right)
        self.plot_frame.pack(fill="both", expand=True, padx=8, pady=(0, 8))

        self.figure, (self.ax_hist, self.ax_box) = plt.subplots(1, 2, figsize=(10, 4))
        self.canvas = FigureCanvasTkAgg(self.figure, master=self.plot_frame)
        self.canvas.get_tk_widget().pack(fill="both", expand=True)
        self.figure.tight_layout()

    def _select_all(self):
        for var in self.y_checks.values():
            var.set(True)

    def _clear_selection(self):
        for var in self.y_checks.values():
            var.set(False)

    def _run_analysis(self):
        selected_y = [col for col, var in self.y_checks.items() if var.get()]
        if not selected_y:
            messagebox.showwarning("Select columns", "Please select at least one numeric Y column.")
            return

        group_col = self.group_var.get()
        if group_col == "None":
            group_col = None

        # Compute summary and update UI
        summary_rows = self._compute_summary(selected_y, group_col)
        self._render_table(summary_rows)
        self._update_plots(selected_y, group_col)

    def _compute_summary(self, y_cols, group_col=None):
        """Compute descriptive metrics. Returns list of dicts."""
        rows = []
        df = self.df

        def calc_stats(series: pd.Series):
            clean = series.dropna()
            if clean.empty:
                return None
            q1 = clean.quantile(0.25)
            q3 = clean.quantile(0.75)
            return {
                "count": int(clean.count()),
                "mean": clean.mean(),
                "std": clean.std(),
                "min": clean.min(),
                "p25": q1,
                "median": clean.median(),
                "p75": q3,
                "max": clean.max(),
                "iqr": q3 - q1,
                "skew": clean.skew(),
                "kurt": clean.kurtosis(),
                "missing": int(series.isna().sum()),
            }

        if group_col:
            for grp, grp_df in df.groupby(group_col):
                for col in y_cols:
                    stats = calc_stats(grp_df[col])
                    if stats:
                        rows.append({
                            "Group": str(grp),
                            "Column": col,
                            "Count": stats["count"],
                            "Mean": stats["mean"],
                            "Std": stats["std"],
                            "Min": stats["min"],
                            "P25": stats["p25"],
                            "Median": stats["median"],
                            "P75": stats["p75"],
                            "Max": stats["max"],
                            "IQR": stats["iqr"],
                            "Skew": stats["skew"],
                            "Kurtosis": stats["kurt"],
                            "Missing": stats["missing"],
                        })
        else:
            for col in y_cols:
                stats = calc_stats(df[col])
                if stats:
                    rows.append({
                        "Group": "-",
                        "Column": col,
                        "Count": stats["count"],
                        "Mean": stats["mean"],
                        "Std": stats["std"],
                        "Min": stats["min"],
                        "P25": stats["p25"],
                        "Median": stats["median"],
                        "P75": stats["p75"],
                        "Max": stats["max"],
                        "IQR": stats["iqr"],
                        "Skew": stats["skew"],
                        "Kurtosis": stats["kurt"],
                        "Missing": stats["missing"],
                    })

        return rows

    def _render_table(self, rows):
        # Clear existing
        for item in self.tree.get_children():
            self.tree.delete(item)

        for row in rows:
            values = [
                row.get("Group", "-"),
                row.get("Column", ""),
                row.get("Count", ""),
                self._fmt(row.get("Mean")),
                self._fmt(row.get("Std")),
                self._fmt(row.get("Min")),
                self._fmt(row.get("P25")),
                self._fmt(row.get("Median")),
                self._fmt(row.get("P75")),
                self._fmt(row.get("Max")),
                self._fmt(row.get("IQR")),
                self._fmt(row.get("Skew")),
                self._fmt(row.get("Kurtosis")),
                row.get("Missing", 0),
            ]
            self.tree.insert("", "end", values=values)

    def _fmt(self, value):
        if value is None:
            return "-"
        try:
            return f"{value:.4g}"
        except Exception:
            return str(value)

    def _update_plots(self, y_cols, group_col=None):
        # Clear axes
        self.figure.clear()
        self.ax_hist, self.ax_box = self.figure.subplots(1, 2)

        # Histogram
        palette = sns.color_palette("tab10")
        if group_col:
            base_col = y_cols[0]
            for idx, (grp, grp_df) in enumerate(self.df.groupby(group_col)):
                vals = grp_df[base_col].dropna()
                if vals.empty:
                    continue
                self.ax_hist.hist(vals, bins=20, alpha=0.5, label=str(grp), color=palette[idx % len(palette)])
            self.ax_hist.set_title(f"Histogram: {base_col} by {group_col}")
            self.ax_hist.legend()
        else:
            for idx, col in enumerate(y_cols[:3]):
                vals = self.df[col].dropna()
                if vals.empty:
                    continue
                self.ax_hist.hist(vals, bins=20, alpha=0.5, label=col, color=palette[idx % len(palette)])
            self.ax_hist.set_title("Histogram")
            if len(y_cols) > 1:
                self.ax_hist.legend()

        self.ax_hist.set_xlabel("Value")
        self.ax_hist.set_ylabel("Frequency")

        # Boxplot
        if group_col:
            melted = self.df[[group_col] + y_cols].melt(id_vars=group_col, var_name="variable", value_name="value")
            sns.boxplot(data=melted.dropna(), x=group_col, y="value", hue="variable", ax=self.ax_box)
            self.ax_box.set_title("Boxplot by group")
            self.ax_box.legend(title="Y")
        else:
            melted = self.df[y_cols].melt(var_name="variable", value_name="value")
            sns.boxplot(data=melted.dropna(), x="variable", y="value", ax=self.ax_box)
            self.ax_box.set_title("Boxplot")
            self.ax_box.set_xlabel("Y")

        self.ax_box.set_ylabel("Value")
        self.figure.tight_layout()
        self.canvas.draw_idle()
