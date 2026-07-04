"""
Página Inicial do Pro Sigma
Importação de dados e seleção de ferramentas
"""
import tkinter as tk
import customtkinter as ctk
from tkinter import filedialog, messagebox
import pandas as pd
import threading
from pathlib import Path
from typing import Optional
from src.utils.file_history import FileHistory
from src.utils.performance_utils import resize_optimizer, optimize_frame_resize
from src.ui.inline_spreadsheet import InlineSpreadsheet


class HomePage(ctk.CTkFrame):
    """Página inicial com importação de dados e menu de ferramentas"""
    
    def __init__(self, parent, license_data: dict, on_license_change=None, **kwargs):
        """
        Inicializa a página inicial
        
        Args:
            parent: Widget pai
            license_data: Dados da licença do usuário
            on_license_change: Callback quando a licença é alterada
        """
        super().__init__(parent, **kwargs)
        
        self.parent = parent
        self.license_data = license_data
        self.current_data: Optional[pd.DataFrame] = None
        self.current_file_path: Optional[str] = None
        self.on_license_change = on_license_change
        self.file_history = FileHistory()
        
        # Otimizações de performance
        self._debounced_resize = resize_optimizer.debounce(self._handle_resize)
        
        self.create_widgets()
        
        # Bind para otimizar redimensionamento
        self.bind('<Configure>', self._on_configure)
    
    def create_widgets(self):
        """Cria os widgets da interface"""
        
        # Container principal com menu lateral
        main_container = ctk.CTkFrame(self, fg_color="transparent")
        main_container.pack(fill="both", expand=True)
        optimize_frame_resize(main_container)
        
        # ===== MENU LATERAL =====
        self.sidebar = ctk.CTkFrame(main_container, width=220, corner_radius=0)
        self.sidebar.pack(side="left", fill="y", padx=0, pady=0)
        self.sidebar.pack_propagate(False)  # Mant\u00e9m largura fixa
        

        # Logo/Título no sidebar
        sidebar_title = ctk.CTkLabel(
            self.sidebar,
            text="Pro Sigma",
            font=ctk.CTkFont(size=20, weight="bold")
        )
        sidebar_title.pack(pady=(30, 10))
        
        sidebar_subtitle = ctk.CTkLabel(
            self.sidebar,
            text="Six Sigma Analytics",
            font=ctk.CTkFont(size=11),
            text_color="gray"
        )
        sidebar_subtitle.pack(pady=(0, 30))
        
        # Info do plano
        plan_info_frame = ctk.CTkFrame(self.sidebar)
        plan_info_frame.pack(fill="x", padx=10, pady=(0, 20))
        
        plan_title = ctk.CTkLabel(
            plan_info_frame,
            text="Seu Plano",
            font=ctk.CTkFont(size=11, weight="bold")
        )
        plan_title.pack(pady=(10, 5))
        
        plan_name = ctk.CTkLabel(
            plan_info_frame,
            text=self.license_data['plan_name'],
            font=ctk.CTkFont(size=14, weight="bold"),
            text_color="#2E86DE"
        )
        plan_name.pack(pady=(0, 5))
        
        plan_expiry = ctk.CTkLabel(
            plan_info_frame,
            text=f"Válido até:\n{self.license_data['expiratedDate']}",
            font=ctk.CTkFont(size=10),
            text_color="gray"
        )
        plan_expiry.pack(pady=(0, 5))

        plan_version = ctk.CTkLabel(
            plan_info_frame,
            text="v0.1.0",
            font=ctk.CTkFont(size=9),
            text_color="gray60"
        )
        plan_version.pack(pady=(0, 10))
        
        # Separador
        separator1 = ctk.CTkFrame(self.sidebar, height=2, fg_color="#CCCCCC")
        separator1.pack(fill="x", padx=20, pady=20)
        
        # Botões do menu
        menu_label = ctk.CTkLabel(
            self.sidebar,
            text="MENU",
            font=ctk.CTkFont(size=10, weight="bold"),
            text_color="gray"
        )
        menu_label.pack(pady=(0, 10))
        
        # Botão Renovar Plano
        renew_button = ctk.CTkButton(
            self.sidebar,
            text="🔄 Renovar/Alterar Plano",
            command=self.show_renew_options,
            fg_color="#2E86DE",
            hover_color="#1E5BA8",
            height=40,
            font=ctk.CTkFont(size=12)
        )
        renew_button.pack(fill="x", padx=10, pady=5)
        
        # Botão Histórico
        history_button = ctk.CTkButton(
            self.sidebar,
            text="📚 Histórico de Arquivos",
            command=self.show_history,
            fg_color="#E8EDF2",
            hover_color="#D0D8E4",
            text_color="#2D3748",
            height=40,
            font=ctk.CTkFont(size=12)
        )
        history_button.pack(fill="x", padx=10, pady=5)

        # Botão Inserir Nova Licença
        new_license_button = ctk.CTkButton(
            self.sidebar,
            text="🔑 Inserir Nova Licença",
            command=self.insert_new_license,
            fg_color="#E8EDF2",
            hover_color="#D0D8E4",
            text_color="#2D3748",
            height=40,
            font=ctk.CTkFont(size=12)
        )
        new_license_button.pack(fill="x", padx=10, pady=5)

        # Botão Suporte
        support_button = ctk.CTkButton(
            self.sidebar,
            text="💬 Suporte",
            command=self.open_support,
            fg_color="#E8EDF2",
            hover_color="#D0D8E4",
            text_color="#2D3748",
            height=40,
            font=ctk.CTkFont(size=12)
        )
        support_button.pack(fill="x", padx=10, pady=5)

        # Botão Sobre
        about_button = ctk.CTkButton(
            self.sidebar,
            text="ℹ️ Sobre",
            command=self.show_about,
            fg_color="#E8EDF2",
            hover_color="#D0D8E4",
            text_color="#2D3748",
            height=40,
            font=ctk.CTkFont(size=12)
        )
        about_button.pack(fill="x", padx=10, pady=5)
        
        # Espaçador
        spacer = ctk.CTkFrame(self.sidebar, fg_color="transparent")
        spacer.pack(fill="both", expand=True)
        
        # Versão no rodapé
        version_label = ctk.CTkLabel(
            self.sidebar,
            text="v0.1.0",
            font=ctk.CTkFont(size=9),
            text_color="gray50"
        )
        version_label.pack(pady=(0, 10))
        
        # ===== ÁREA PRINCIPAL =====
        content_area = ctk.CTkFrame(main_container, fg_color="transparent")
        content_area.pack(side="right", fill="both", expand=True)

        # ── Toolbar de importação (topo compacto) ────────────────────────────
        toolbar = ctk.CTkFrame(content_area, fg_color="transparent")
        toolbar.pack(fill="x", padx=15, pady=(12, 0))

        self.import_excel_btn = ctk.CTkButton(
            toolbar, text="📊 Importar Excel",
            command=self.import_excel,
            width=145, height=36,
            font=ctk.CTkFont(size=12, weight="bold"),
            fg_color="#2E86DE", hover_color="#1E5BA8", corner_radius=7
        )
        self.import_excel_btn.pack(side="left", padx=(0, 8))

        self.import_csv_btn = ctk.CTkButton(
            toolbar, text="📄 Importar CSV",
            command=self.import_csv,
            width=140, height=36,
            font=ctk.CTkFont(size=12, weight="bold"),
            fg_color="#2E86DE", hover_color="#1E5BA8", corner_radius=7
        )
        self.import_csv_btn.pack(side="left", padx=(0, 8))

        self.import_status_label = ctk.CTkLabel(
            toolbar, text="Cole dados do Excel com Ctrl+V na tabela abaixo",
            font=ctk.CTkFont(size=11), text_color="gray", anchor="w"
        )
        self.import_status_label.pack(side="left", padx=(8, 0))

        # ── Spreadsheet ocupando toda a área principal ───────────────────────
        sheet_outer = tk.Frame(content_area, bg="white", bd=1, relief="solid")
        sheet_outer.pack(fill="both", expand=True, padx=15, pady=(8, 10))

        self.spreadsheet = InlineSpreadsheet(
            sheet_outer,
            on_data_change=self._on_spreadsheet_data_change,
            fg_color="transparent"
        )
        self.spreadsheet.pack(fill="both", expand=True)

        # Cria menu bar com acesso às ferramentas
        self._build_menu_bar()

    def _get_tools_definition(self) -> dict:
        """Retorna a definição completa de ferramentas"""
        return {
            'process_capability': {
                'title': 'Process Capability',
                'description': 'Cálculo de Cp, Cpk, Pp, Ppk',
                'plan': 'basic',
                'in_development': False
            },
            'hypothesis_test': {
                'title': 'Testes de Hipótese',
                'description': 'Testes T, Z, ANOVA, Qui-quadrado',
                'plan': 'basic',
                'in_development': False
            },
            'distribution_test': {
                'title': 'Teste de Distribuição',
                'description': 'Ajuste de distribuições (Normal, Weibull, etc)',
                'plan': 'basic',
                'in_development': False
            },
            'cov_ems': {
                'title': 'COV',
                'description': 'Análise de coeficiente de variação',
                'plan': 'basic',
                'in_development': False
            },
            'normalization_test': {
                'title': 'Testes de Normalidade',
                'description': 'Shapiro-Wilk, Jarque-Bera, Kolmogorov-Smirnov',
                'plan': 'intermediate',
                'in_development': False
            },
            'control_charts': {
                'title': 'Cartas de Controle',
                'description': 'X-bar, R, S, P, NP, C, U',
                'plan': 'intermediate',
                'in_development': False
            },
            'dashboard': {
                'title': 'Dashboard',
                'description': 'Visualização de métricas',
                'plan': 'intermediate',
                'in_development': True  # Em desenvolvimento
            },
            'monte_carlo': {
                'title': 'Monte Carlo',
                'description': 'Simulações Monte Carlo',
                'plan': 'intermediate',
                'in_development': False
            },
            'variability': {
                'title': 'Análise de Variabilidade',
                'description': 'Gráficos de variabilidade com múltiplos fatores X e Y',
                'plan': 'basic',
                'in_development': False
            },
            'text_analysis': {
                'title': 'Análise de Texto',
                'description': 'Mineração de texto e processamento de linguagem natural',
                'plan': 'intermediate',
                'in_development': False
            },
            'doe': {
                'title': 'DOE',
                'description': 'Design of Experiments',
                'plan': 'intermediate',
                'in_development': True  # Em desenvolvimento
            },
            'simple_regression': {
                'title': 'Regressão Simples',
                'description': 'Regressão linear simples (1 X e 1 Y)',
                'plan': 'pro',
                'in_development': False
            },
            'multiple_regression': {
                'title': 'Regressão Múltipla',
                'description': 'Regressão com múltiplos X e Y, interações e seleção de modelo',
                'plan': 'pro',
                'in_development': False
            },
            'multivariate': {
                'title': 'Análise Multivariada',
                'description': 'Matriz de Correlação e Scatter Plot Matrix',
                'plan': 'pro',
                'in_development': False
            },
            'stackup': {
                'title': 'StackUp',
                'description': 'Análise de tolerâncias 2D',
                'plan': 'intermediate',
                'in_development': False  # Implementado
            },
            'space_filling': {
                'title': 'Space Filling Design',
                'description': 'Latin Hypercube, Sphere Packing e Análise',
                'plan': 'pro',
                'in_development': False  # Implementado
            },
            'neural_networks': {
                'title': 'Redes Neurais',
                'description': 'MLP para Classificação e Regressão com Holdout e K-Fold',
                'plan': 'pro',
                'in_development': False  # Implementado
            },
            'tree_models': {
                'title': 'Modelos de Árvore',
                'description': 'Decision Tree, Random Forest e Gradient Boosting',
                'plan': 'pro',
                'in_development': False  # Implementado
            },
            'gage_rr': {
                'title': 'Gage R&R',
                'description': 'Measurement System Analysis - Repetibilidade e Reprodutibilidade',
                'plan': 'basic',
                'in_development': False  # Implementado
            },
            'run_chart': {
                'title': 'Run Chart',
                'description': 'Gráfico de Sequência com detecção de padrões e tendências',
                'plan': 'basic',
                'in_development': False  # Implementado
            },
            'pareto': {
                'title': 'Gráfico de Pareto',
                'description': 'Análise 80/20 e classificação ABC',
                'plan': 'basic',
                'in_development': False  # Implementado
            },
            'nonlinear': {
                'title': 'Regressão Não Linear',
                'description': 'Análise de regressão não linear',
                'plan': 'pro',
                'in_development': False 
            },
            'ccd': {
                'title': 'Central Composite Design',
                'description': 'Gerar experimentos CCD/Box-Behnken e análise com ANOVA',
                'plan': 'pro',
                'in_development': False
            },
            'k_means': {
                'title': 'K-Means Clustering',
                'description': 'Agrupamento não supervisionado com método do cotovelo',
                'plan': 'pro',
                'in_development': False  # Implementado
            },
            'gaussian_process': {
                'title': 'Gaussian Process',
                'description': 'Regressão probabilística com intervalos de confiança',
                'plan': 'pro',
                'in_development': False  # Implementado
            },
            'logistic_regression': {
                'title': 'Regressão Logística',
                'description': 'Classificação binária com GLM e análise de probabilidades',
                'plan': 'pro',
                'in_development': False  # Implementado
            },
            'mixture_design': {
                'title': 'Design de Mistura',
                'description': 'Design experimental para misturas com restrições de soma',
                'plan': 'pro',
                'in_development': False
            },
            'box_cox': {
                'title': 'Transformação Box-Cox',
                'description': 'Encontra o lambda ótimo para normalização de dados',
                'plan': 'intermediate',
                'in_development': False
            },
            'sample_size_explorer': {
                'title': 'Tamanho de Amostra',
                'description': 'Calcula N, margem de erro e nível de confiança',
                'plan': 'basic',
                'in_development': False
            },
            'time_series': {
                'title': 'Séries Temporais',
                'description': 'Decomposição, diagnóstico e teste ADF de estacionariedade',
                'plan': 'intermediate',
                'in_development': False
            },
            'data_editor': {
                'title': 'Editor de Dados',
                'description': 'Crie, edite e gere dados por fórmulas e distribuições',
                'plan': 'basic',
                'in_development': False
            },
        }

    def _build_menu_bar(self):
        """Cria o menu bar nativo com ferramentas agrupadas por domínio."""
        tools_def = self._get_tools_definition()
        available  = self.license_data.get('features', [])

        # Agrupamento por domínio (sem rótulos de plano)
        DOMAIN_GROUPS = [
            ("Estatística", [
                "hypothesis_test", "distribution_test",
                "normality_test", "cov_ems", "box_cox", "sample_size_explorer",
                "multivariate",
            ]),
            ("Qualidade", [
                "process_capability", "gage_rr", "pareto", "variability",
            ]),
            ("Controle", [
                "control_charts", "run_chart", "monte_carlo", "time_series",
            ]),
            ("Regressão", [
                "simple_regression", "multiple_regression", "nonlinear",
                "logistic_regression", "gaussian_process",
            ]),
            ("Machine Learning", [
                "neural_networks", "tree_models", "k_means",
            ]),
            ("DOE", [
                "ccd", "space_filling", "mixture_design", "stackup",
            ]),
            ("Dados", [
                "text_analysis",
            ]),
        ]

        menubar = tk.Menu(self.parent)

        # ── Arquivo ───────────────────────────────────────────────────────────
        arq = tk.Menu(menubar, tearoff=0)
        arq.add_command(label="Importar Excel",        command=self.import_excel)
        arq.add_command(label="Importar CSV",          command=self.import_csv)
        arq.add_separator()
        arq.add_command(label="Histórico de Arquivos", command=self.show_history)
        arq.add_separator()
        arq.add_command(label="Editor de Dados",       command=self._open_data_editor)
        menubar.add_cascade(label="Arquivo", menu=arq)

        # ── Domínios ──────────────────────────────────────────────────────────
        for domain_label, feature_ids in DOMAIN_GROUPS:
            domain_menu = tk.Menu(menubar, tearoff=0)
            has_any = False
            for fid in feature_ids:
                if fid not in available:
                    continue
                info = tools_def.get(fid)
                if not info:
                    continue
                title = info["title"]
                in_dev = info.get("in_development", False)
                if in_dev:
                    title += "  [Em breve]"
                domain_menu.add_command(
                    label=title,
                    command=lambda f=fid: self.open_tool(f),
                    state="disabled" if in_dev else "normal"
                )
                has_any = True
            if has_any:
                menubar.add_cascade(label=domain_label, menu=domain_menu)

        # ── Ajuda ─────────────────────────────────────────────────────────────
        ajuda = tk.Menu(menubar, tearoff=0)
        ajuda.add_command(label="Suporte", command=self.open_support)
        ajuda.add_command(label="Sobre",   command=self.show_about)
        menubar.add_cascade(label="Ajuda", menu=ajuda)

        self.parent.config(menu=menubar)
        self._menubar = menubar

    def _open_data_editor(self):
        from src.analytics.data_editor.data_editor_window import DataEditorWindow

        def _on_editor_data(df):
            self.current_data = df
            self.spreadsheet.load_dataframe(df)
            self.import_status_label.configure(
                text=f"✓ Editor de Dados: {len(df)} linhas × {len(df.columns)} colunas",
                text_color="#2D8A4E"
            )

        DataEditorWindow(self, initial_df=self.current_data, on_use_data=_on_editor_data)

    def import_excel(self):
        """Importa arquivo Excel"""
        file_path = filedialog.askopenfilename(
            title="Selecione um arquivo Excel",
            filetypes=[
                ("Excel files", "*.xlsx *.xls"),
                ("All files", "*.*")
            ]
        )
        
        if file_path:
            self.load_file(file_path, 'excel')
    
    def import_csv(self):
        """Importa arquivo CSV"""
        file_path = filedialog.askopenfilename(
            title="Selecione um arquivo CSV",
            filetypes=[
                ("CSV files", "*.csv"),
                ("All files", "*.*")
            ]
        )
        
        if file_path:
            self.load_file(file_path, 'csv')
    
    def load_file(self, file_path: str, file_type: str):
        """Carrega arquivo de dados em thread separada para não travar a UI"""
        self.show_loading(True)

        def _read_in_thread():
            try:
                if file_type == 'excel':
                    data = pd.read_excel(file_path)
                else:
                    data = pd.read_csv(file_path)
                self.after(0, lambda: self._on_file_loaded(data, file_path, file_type))
            except Exception as e:
                self.after(0, lambda: self._on_file_error(str(e)))

        threading.Thread(target=_read_in_thread, daemon=True).start()

    def _on_file_loaded(self, data: pd.DataFrame, file_path: str, file_type: str):
        """Callback (na thread da UI) após leitura bem-sucedida"""
        self.current_data = data
        self.current_file_path = file_path

        file_name = Path(file_path).name
        rows, cols = data.shape

        self.file_history.add_file(file_path, file_type, rows, cols)

        self.import_status_label.configure(
            text=f"✓ {file_name}  ({rows} linhas × {cols} colunas)",
            text_color="#2D8A4E"
        )
        self.spreadsheet.load_dataframe(data)
        self.show_loading(False)

        messagebox.showinfo(
            "Sucesso",
            f"Arquivo carregado com sucesso!\n\nLinhas: {rows}\nColunas: {cols}"
        )

    def _on_spreadsheet_data_change(self, df: pd.DataFrame):
        """Chamado sempre que o spreadsheet é editado diretamente."""
        if df is not None and not df.empty:
            self.current_data = df
            r, c = df.shape
            self.import_status_label.configure(
                text=f"Tabela: {r} linhas × {c} colunas  (editado manualmente)",
                text_color="#1f538d"
            )

    def _on_file_error(self, error_msg: str):
        """Callback (na thread da UI) após erro de leitura"""
        self.show_loading(False)
        messagebox.showerror("Erro", f"Erro ao carregar arquivo:\n{error_msg}")
    
    def show_loading(self, show: bool):
        """
        Mostra/oculta indicador de carregamento
        
        Args:
            show: True para mostrar, False para ocultar
        """
        if show:
            # Desabilita botões
            self.import_excel_btn.configure(state="disabled")
            self.import_csv_btn.configure(state="disabled")
            
            # Mostra mensagem de loading
            self.import_status_label.configure(
                text="⏳ Carregando arquivo, aguarde...",
                text_color="#FFA500"
            )
            
            # Força atualização da UI
            self.update()
        else:
            # Reabilita botões
            self.import_excel_btn.configure(state="normal")
            self.import_csv_btn.configure(state="normal")
    
    def show_file_info(self, filename: str, rows: int, cols: int):
        """Atualiza o status de importação."""
        self.import_status_label.configure(
            text=f"✓ {filename}  ({rows} linhas × {cols} colunas)",
            text_color="#2D8A4E"
        )
    
    def open_tool(self, feature_id: str):
        """
        Abre uma ferramenta de análise
        
        Args:
            feature_id: ID da ferramenta
        """
        # Tools that don't need data input
        if feature_id == 'monte_carlo':
            from src.analytics.monte_carlo.monte_carlo_window import MonteCarloWindow
            MonteCarloWindow(self)
            return

        if feature_id == 'stackup':
            from src.analytics.stack_up.stack_up_window import StackUpWindow
            StackUpWindow(self)
            return

        if feature_id == 'sample_size_explorer':
            from src.analytics.sample_size_explorer.sample_size_explorer_window import SampleSizeExplorerWindow
            SampleSizeExplorerWindow(self)
            return

        if feature_id == 'data_editor':
            from src.analytics.data_editor.data_editor_window import DataEditorWindow

            def _on_editor_data(df):
                self.current_data = df
                messagebox.showinfo(
                    "Dados Carregados",
                    f"Dados do editor disponíveis para análise!\n"
                    f"{len(df)} linhas × {len(df.columns)} colunas",
                    parent=self
                )

            DataEditorWindow(
                self,
                initial_df=self.current_data,
                on_use_data=_on_editor_data
            )
            return
        
        # A tabela na tela é a fonte de verdade: sempre que tiver conteúdo,
        # usa os dados/colunas que o usuário preencheu/editou nela
        self.spreadsheet.commit_pending_edit()
        sheet_df = self.spreadsheet.get_dataframe()
        if not sheet_df.empty:
            self.current_data = sheet_df

        # Check if there's any data available (current or historical)
        has_current_data = self.current_data is not None
        has_history = len(self.file_history.get_recent_files(count=1)) > 0

        # If no current data and no history, show warning
        if not has_current_data and not has_history:
            response = messagebox.askyesno(
                "Nenhum Dado Disponível",
                "Nenhum dado disponível.\n\n"
                "Você pode:\n"
                "• Importar um arquivo Excel ou CSV\n"
                "• Colar dados do Excel com Ctrl+V na tabela\n"
                "• Digitar dados diretamente na tabela\n\n"
                "Deseja importar um arquivo agora?"
            )
            if response:
                self.import_excel()
            return
        
        # Show data selection window
        from src.ui.data_selection_window import DataSelectionWindow
        
        def on_data_selected(selected_data, selected_file_path):
            """Callback when data is selected"""
            if selected_data is None:
                return
            
            # Open the appropriate tool
            if feature_id == 'process_capability':
                from src.analytics.capability.capability_window import CapabilityWindow
                capability_window = CapabilityWindow(self, selected_data)
            elif feature_id == 'cov_ems':
                from src.analytics.cov.cov_window import CovWindow
                cov_window = CovWindow(self, selected_data)
            elif feature_id == 'hypothesis_test':
                from src.analytics.hypothesis_test.hypothesis_test_window import HypothesisTestWindow
                hypothesis_window = HypothesisTestWindow(self, selected_data)
            elif feature_id == 'distribution_test':
                from src.analytics.distribution_test.distribution_test_window import DistributionTestWindow
                DistributionTestWindow(self, selected_data)
            elif feature_id == 'normalization_test':
                from src.analytics.normality_test.normality_test_window import NormalityTestWindow
                NormalityTestWindow(self, selected_data)
            elif feature_id == 'control_charts':
                from src.analytics.control_charts.control_chart_window import ControlChartWindow
                ControlChartWindow(self, selected_data)
            elif feature_id == 'variability':
                from src.analytics.variability.variability_window import VariabilityWindow
                VariabilityWindow(self, selected_data)
            elif feature_id == 'text_analysis':
                from src.analytics.text_analysis.text_analysis_window import TextAnalysisWindow
                TextAnalysisWindow(self, selected_data)
            elif feature_id == 'simple_regression':
                from src.analytics.simple_regression.simple_regression_window import SimpleRegressionWindow
                SimpleRegressionWindow(self, selected_data)
            elif feature_id == 'multiple_regression':
                from src.analytics.multiple_regression.multiple_regression_window import MultipleRegressionWindow
                MultipleRegressionWindow(self, selected_data)
            elif feature_id == 'multivariate':
                from src.analytics.multivariate.multivariate_window import MultivariateWindow
                MultivariateWindow(self, selected_data)
            elif feature_id == 'space_filling':
                from src.analytics.space_filling.space_filling_window import SpaceFillingWindow
                SpaceFillingWindow(self, selected_data)
            elif feature_id == 'nonlinear':
                from src.analytics.nonlinear.nonlinear_window import NonlinearWindow
                NonlinearWindow(self, selected_data)
            elif feature_id == 'ccd':
                from src.analytics.ccd.ccd_window import CCDWindow
                CCDWindow(self, selected_data)
            elif feature_id == 'neural_networks':
                from src.analytics.neural_network.neural_network_window import NeuralNetworkWindow
                NeuralNetworkWindow(self, selected_data)
            elif feature_id == 'tree_models':
                from src.analytics.tree_models.tree_models_window import TreeModelsWindow
                TreeModelsWindow(self, selected_data)
            elif feature_id == 'gage_rr':
                from src.analytics.msa.gage_rr_window import GageRRWindow
                GageRRWindow(self, selected_data)
            elif feature_id == 'run_chart':
                from src.analytics.run_chart.run_chart_window import RunChartWindow
                RunChartWindow(self, selected_data)
            elif feature_id == 'pareto':
                from src.analytics.pareto.pareto_window import ParetoWindow
                ParetoWindow(self, selected_data)
            elif feature_id == 'k_means':
                from src.analytics.clustering.k_means_window import KMeansWindow
                KMeansWindow(self, selected_data)
            elif feature_id == 'gaussian_process':
                from src.analytics.gaussian_process.gaussian_process_window import GaussianProcessWindow
                GaussianProcessWindow(self, selected_data)
            elif feature_id == 'logistic_regression':
                from src.analytics.logistic_regression.logistic_regression_window import LogisticRegressionWindow
                LogisticRegressionWindow(self, selected_data)
            elif feature_id == 'mixture_design':
                from src.analytics.mixture_design.mixture_design_window import MixtureDesignWindow
                MixtureDesignWindow(self, selected_data)
            elif feature_id == 'box_cox':
                from src.analytics.box_cox.box_cox_window import BoxCoxWindow
                BoxCoxWindow(self, selected_data)
            elif feature_id == 'time_series':
                from src.analytics.time_series.time_series_window import TimeSeriesWindow
                TimeSeriesWindow(self, selected_data)
            else:
                # TODO: Implementar outras ferramentas
                messagebox.showinfo(
                    "Em Desenvolvimento",
                    f"A ferramenta '{feature_id}' será implementada em breve.\n\n"
                    f"Dados selecionados: {selected_data.shape[0]} linhas"
                )
        
        # Open data selection window (will show current or history or both)
        DataSelectionWindow(self, self.current_data, self.current_file_path, on_data_selected)

    
    def show_renew_options(self):
        """Mostra opções de renovação/upgrade de plano"""
        response = messagebox.askyesno(
            "Renovar/Alterar Plano",
            "Deseja acessar a página de renovação/upgrade de plano?\n\n"
            "Você será redirecionado para o portal do cliente onde poderá:\n"
            "• Renovar seu plano atual\n"
            "• Fazer upgrade para um plano superior\n"
            "• Gerenciar suas informações de pagamento"
        )
        
        if response:
            # URL mockada - em produção seria a URL real do portal
            mock_url = "https://prosigma.com/portal/renovacao"
            messagebox.showinfo(
                "Portal do Cliente",
                f"Abrindo o portal do cliente...\n\n"
                f"URL (mockada): {mock_url}\n\n"
                f"Após a renovação, você receberá uma nova chave de licença por email."
            )
            # webbrowser.open(mock_url)  # Descomente em produção
    
    def insert_new_license(self):
        """Permite inserir uma nova chave de licença"""
        from src.ui.renew_license_window import RenewLicenseWindow
        
        renew_window = RenewLicenseWindow(
            self.parent,
            on_success_callback=self.on_license_renewed
        )
    
    def on_license_renewed(self, new_license_data: dict):
        """
        Callback chamado quando uma nova licença é ativada
        
        Args:
            new_license_data: Dados da nova licença
        """
        self.license_data = new_license_data
        
        messagebox.showinfo(
            "Licença Atualizada",
            f"Sua licença foi atualizada com sucesso!\n\n"
            f"Novo plano: {new_license_data['plan_name']}\n"
            f"Válido até: {new_license_data['expiratedDate']}\n\n"
            f"A aplicação será reiniciada para aplicar as mudanças."
        )
        
        # Notifica o callback se existir
        if self.on_license_change:
            self.on_license_change(new_license_data)
    
    def open_support(self):
        """Abre página de suporte"""
        response = messagebox.askyesno(
            "Suporte Pro Sigma",
            "Como podemos ajudar?\n\n"
            "• Documentação online\n"
            "• Tutoriais em vídeo\n"
            "• Suporte por email\n"
            "• Chat ao vivo\n\n"
            "Deseja acessar o portal de suporte?"
        )
        
        if response:
            mock_url = "https://prosigma.com/suporte"
            messagebox.showinfo(
                "Portal de Suporte",
                f"Abrindo portal de suporte...\n\n"
                f"URL (mockada): {mock_url}\n\n"
                f"Email de suporte: suporte@prosigma.com\n"
                f"Horário: Segunda a Sexta, 9h às 18h"
            )
            # webbrowser.open(mock_url)  # Descomente em produção
    
    def show_about(self):
        """Mostra informações sobre o software"""
        messagebox.showinfo(
            "Sobre Pro Sigma",
            "Pro Sigma - Análise Estatística Six Sigma\n\n"
            "Versão: 0.1.0\n"
            "Desenvolvido para profissionais da qualidade\n\n"
            "© 2025 Pro Sigma. Todos os direitos reservados.\n\n"
            "Seu plano atual: " + self.license_data['plan_name'] + "\n"
            f"Válido até: {self.license_data['expiratedDate']}"
        )
    
    def show_history(self):
        """Mostra janela de histórico de arquivos"""
        from src.ui.history_window import HistoryWindow
        
        history_window = HistoryWindow(
            self.parent,
            on_file_selected=self.load_file_from_history
        )
    
    def load_file_from_history(self, file_path: str, file_type: str):
        """
        Carrega um arquivo do histórico
        
        Args:
            file_path: Caminho do arquivo
            file_type: Tipo do arquivo (excel ou csv)
        """
        self.load_file(file_path, file_type)
    
    def _on_configure(self, event):
        """
        Handler otimizado para eventos de configuração (resize)
        Usa debounce para evitar múltiplas reconstruções
        """
        # Se a janela já foi destruída, ignora callbacks pendentes
        if not self.winfo_exists():
            return
        
        # Ignora eventos que não são da própria janela
        if event.widget != self:
            return
        
        # Usa debounce centralizado
        self._debounced_resize()
    
    def _handle_resize(self):
        """
        Processa o redimensionamento de forma otimizada
        """
        # Se a janela foi destruída, aborta
        if not self.winfo_exists():
            return

        # Força uma única atualização após o resize
        self.update_idletasks()

    def destroy(self):
        """Cancela callbacks pendentes antes de destruir."""
        try:
            if resize_optimizer.pending_call:
                self.after_cancel(resize_optimizer.pending_call)
        except Exception:
            pass
        try:
            self.parent.config(menu="")
        except Exception:
            pass
        super().destroy()