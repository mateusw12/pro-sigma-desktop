"""
Página Inicial do Pro Sigma
Importação de dados e seleção de ferramentas
"""
import customtkinter as ctk
from tkinter import filedialog, messagebox
import pandas as pd
import threading
from pathlib import Path
from typing import Optional
from src.utils.file_history import FileHistory
from src.utils.performance_utils import resize_optimizer, optimize_frame_resize


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
        plan_expiry.pack(pady=(0, 10))
        
        # Separador
        separator1 = ctk.CTkFrame(self.sidebar, height=2, fg_color="gray30")
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
            fg_color="gray30",
            hover_color="gray20",
            height=40,
            font=ctk.CTkFont(size=12)
        )
        history_button.pack(fill="x", padx=10, pady=5)
        
        # Botão Inserir Nova Licença
        new_license_button = ctk.CTkButton(
            self.sidebar,
            text="🔑 Inserir Nova Licença",
            command=self.insert_new_license,
            fg_color="gray30",
            hover_color="gray20",
            height=40,
            font=ctk.CTkFont(size=12)
        )
        new_license_button.pack(fill="x", padx=10, pady=5)
        
        # Botão Suporte
        support_button = ctk.CTkButton(
            self.sidebar,
            text="💬 Suporte",
            command=self.open_support,
            fg_color="gray30",
            hover_color="gray20",
            height=40,
            font=ctk.CTkFont(size=12)
        )
        support_button.pack(fill="x", padx=10, pady=5)
        
        # Botão Sobre
        about_button = ctk.CTkButton(
            self.sidebar,
            text="ℹ️ Sobre",
            command=self.show_about,
            fg_color="gray30",
            hover_color="gray20",
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
        
        # ===== ÁREA DE IMPORTAÇÃO (TOPO) =====
        import_frame = ctk.CTkFrame(content_area, corner_radius=10)
        import_frame.pack(fill="x", padx=20, pady=(20, 10))
        
        # Header da área de importação
        import_header = ctk.CTkFrame(import_frame, fg_color="transparent")
        import_header.pack(fill="x", padx=20, pady=(20, 10))
        
        import_title = ctk.CTkLabel(
            import_header,
            text="📁 Importar Dados",
            font=ctk.CTkFont(size=18, weight="bold")
        )
        import_title.pack(side="left")
        
        import_desc = ctk.CTkLabel(
            import_header,
            text="Excel (.xlsx, .xls) ou CSV (.csv)",
            font=ctk.CTkFont(size=11),
            text_color="gray"
        )
        import_desc.pack(side="left", padx=(10, 0))
        
        # Container principal da importação
        import_content = ctk.CTkFrame(import_frame, fg_color="transparent")
        import_content.pack(fill="x", padx=20, pady=(0, 20))
        
        # Lado esquerdo: Botões
        button_container = ctk.CTkFrame(import_content, fg_color="transparent")
        button_container.pack(side="left", fill="x", expand=True)
        
        buttons_row = ctk.CTkFrame(button_container, fg_color="transparent")
        buttons_row.pack(anchor="w")
        
        self.import_excel_btn = ctk.CTkButton(
            buttons_row,
            text="📊 Importar Excel",
            command=self.import_excel,
            width=160,
            height=45,
            font=ctk.CTkFont(size=13, weight="bold"),
            fg_color="#2E86DE",
            hover_color="#1E5BA8",
            corner_radius=8
        )
        self.import_excel_btn.pack(side="left", padx=(0, 10))
        
        self.import_csv_btn = ctk.CTkButton(
            buttons_row,
            text="📄 Importar CSV",
            command=self.import_csv,
            width=160,
            height=45,
            font=ctk.CTkFont(size=13, weight="bold"),
            fg_color="#2E86DE",
            hover_color="#1E5BA8",
            corner_radius=8
        )
        self.import_csv_btn.pack(side="left", padx=(0, 10))

        self.open_editor_btn = ctk.CTkButton(
            buttons_row,
            text="📋 Editor de Dados",
            command=self._open_data_editor,
            width=160,
            height=45,
            font=ctk.CTkFont(size=13, weight="bold"),
            fg_color="#7B2D8B",
            hover_color="#5C2167",
            corner_radius=8
        )
        self.open_editor_btn.pack(side="left")

        # Status da importação
        self.import_status_label = ctk.CTkLabel(
            button_container,
            text="💡 Nenhum arquivo carregado ainda",
            font=ctk.CTkFont(size=12),
            text_color="gray",
            anchor="w"
        )
        self.import_status_label.pack(anchor="w", pady=(10, 0))
        
        # Lado direito: Info do arquivo (se carregado)
        self.file_info_frame = ctk.CTkFrame(import_content, corner_radius=8)
        # Inicialmente oculto, será mostrado quando um arquivo for carregado
        
        # ===== FERRAMENTAS DISPONÍVEIS =====
        tools_container = ctk.CTkFrame(content_area, fg_color="transparent")
        tools_container.pack(fill="both", expand=True, padx=20, pady=(0, 20))
        
        # Header das ferramentas
        tools_header = ctk.CTkFrame(tools_container, fg_color="transparent")
        tools_header.pack(fill="x", pady=(0, 15))
        
        tools_title = ctk.CTkLabel(
            tools_header,
            text="🔧 Ferramentas Disponíveis",
            font=ctk.CTkFont(size=20, weight="bold")
        )
        tools_title.pack(side="left")
        
        # Badge com número de ferramentas
        features_count = len(self.license_data.get('features', []))
        count_badge = ctk.CTkLabel(
            tools_header,
            text=f"{features_count}",
            font=ctk.CTkFont(size=12, weight="bold"),
            text_color="white",
            fg_color="#2E86DE",
            corner_radius=15,
            width=30,
            height=30
        )
        count_badge.pack(side="left", padx=(10, 0))
        
        # Grid de ferramentas com scroll
        self.tools_scroll = ctk.CTkScrollableFrame(
            tools_container,
            fg_color="transparent",
            scrollbar_button_color="#2E86DE",
            scrollbar_button_hover_color="#1E5BA8"
        )
        self.tools_scroll.pack(fill="both", expand=True)
        optimize_frame_resize(self.tools_scroll)
        
        # Otimiza scrolling
        self.tools_scroll._scrollbar.configure(width=10)
        
        # Cria botões das ferramentas
        self.create_tool_buttons()
    
    def create_tool_buttons(self):
        """Cria botões para as ferramentas disponíveis (otimizado)"""
        
        # Definição das ferramentas
        # in_development: True = ferramenta desabilitada (em desenvolvimento)
        tools_definition = {
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
            'descriptive_stats': {
                'title': 'Descriptive Statistics',
                'description': 'Histograms, boxplots, summary metrics',
                'plan': 'basic',
                'in_development': False
            },
            'ishikawa': {
                'title': 'Diagrama de Ishikawa',
                'description': 'Diagrama de Causa e Efeito (Espinha de Peixe)',
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
        
        # Obtém features disponíveis
        available_features = self.license_data.get('features', [])
        
        # Organiza ferramentas por categoria
        categories = {
            'Básico': [],
            'Intermediário': [],
            'Avançado': []
        }
        
        plan_to_category = {
            'basic': 'Básico',
            'intermediate': 'Intermediário',
            'pro': 'Avançado'
        }
        
        for feature_id, tool_info in tools_definition.items():
            is_available = feature_id in available_features
            if is_available:
                category = plan_to_category.get(tool_info['plan'], 'Básico')
                categories[category].append((feature_id, tool_info))
        
        # Cria seção para cada categoria que tenha ferramentas (otimizado)
        # Usa after() para criar widgets de forma não-bloqueante
        self._create_categories_async(list(categories.items()), 0)
    
    def _create_categories_async(self, categories_items, index):
        """Cria categorias de forma assíncrona para não travar a UI"""
        if index >= len(categories_items):
            return
        
        category_name, tools_list = categories_items[index]
        
        if tools_list:
            self._create_category(category_name, tools_list)
        
        # Agenda criação da próxima categoria (1ms apenas para ceder controle à UI)
        if index + 1 < len(categories_items):
            self.after(1, lambda: self._create_categories_async(categories_items, index + 1))
    
    def _create_category(self, category_name, tools_list):
        """Cria uma categoria de ferramentas"""
        # Header da categoria
        category_header = ctk.CTkFrame(self.tools_scroll, fg_color="transparent")
        category_header.pack(fill="x", pady=(15, 10), padx=5)
        
        category_label = ctk.CTkLabel(
            category_header,
            text=f"■ {category_name}",
            font=ctk.CTkFont(size=14, weight="bold"),
            text_color="#2E86DE",
            anchor="w"
        )
        category_label.pack(side="left")
        
        # Grid de ferramentas desta categoria
        row_frame = None
        col_count = 0
        max_cols = 4  # 4 botões por linha para layout mais clean
        
        # Ícones pré-definidos para evitar recriação
        icon_map = {
            'variability': '📊', 'process_capability': '📈', 'hypothesis_test': '🔬',
            'distribution_test': '📉', 'cov_ems': '📐', 'distribution_analysis': '📊',
            'analytics': '🔍', 'text_analysis': '📝', 'normalization_test': '✓',
            'control_charts': '📊', 'dashboard': '📊', 'monte_carlo': '🎲',
            'simple_regression': '📈', 'multiple_regression': '📈', 'multivariate': '🔍',
            'stackup': '📏', 'doe': '🧪', 'space_filling': '⬜', 'nonlinear': '📉',
            'ccd': '🎯', 'neural_networks': '🧠', 'decision_tree': '🌳', 
            'descriptive_stats': '📊', 'ishikawa': '🐟', 'tree_models': '🌳',
            'gage_rr': '📏', 'run_chart': '📈', 'pareto': '📊', 'k_means': '🔵',
            'gaussian_process': '📉', 'logistic_regression': '🎯', 'mixture_design': '🧪',
            'box_cox': '🔄', 'sample_size_explorer': '🔢', 'time_series': '📅',
            'data_editor': '📋'
        }
        
        for idx, (feature_id, tool_info) in enumerate(tools_list):
            # Cria nova linha a cada max_cols cards
            if col_count == 0:
                row_frame = ctk.CTkFrame(self.tools_scroll, fg_color="transparent")
                row_frame.pack(fill="x", pady=5)
            
            # Card da ferramenta (simplificado para melhor performance)
            tool_card = self._create_tool_card(row_frame, feature_id, tool_info, icon_map)
            
            col_count += 1
            if col_count >= max_cols:
                col_count = 0
    
    def _create_tool_card(self, parent, feature_id, tool_info, icon_map):
        """Cria um card de ferramenta otimizado com design clean"""
        # Botão em formato de card
        icon = icon_map.get(feature_id, '🔧')
        
        # Verificar se está em desenvolvimento
        is_in_development = tool_info.get('in_development', False)
        
        # Configuração visual baseada no status
        if is_in_development:
            # Ferramenta desabilitada
            fg_color = "gray15"
            hover_color = "gray15"
            text_color = "gray50"
            border_color = "gray25"
            button_text = f"{icon}\n\n{tool_info['title']}\n\n🚧 Em Desenvolvimento"
            command = lambda: self._show_in_development_message(tool_info['title'])
        else:
            # Ferramenta ativa
            fg_color = "gray20"
            hover_color = "#2E86DE"
            text_color = "white"
            border_color = "gray30"
            button_text = f"{icon}\n\n{tool_info['title']}"
            command = lambda: self.open_tool(feature_id)
        
        tool_button = ctk.CTkButton(
            parent,
            text=button_text,
            command=command,
            width=180,
            height=140,
            corner_radius=12,
            fg_color=fg_color,
            hover_color=hover_color,
            text_color=text_color,
            border_width=2,
            border_color=border_color,
            font=ctk.CTkFont(size=11 if is_in_development else 12, weight="bold"),
            anchor="center"
        )
        tool_button.pack(side="left", padx=8, pady=8)
        
        # Tooltip com descrição (aparece no hover)
        tooltip_text = tool_info['description']
        if is_in_development:
            tooltip_text += "\n\n⚠️ Esta ferramenta ainda está em desenvolvimento e será disponibilizada em breve."
        self._create_tooltip(tool_button, tooltip_text)
        
        return tool_button
    
    def _show_in_development_message(self, tool_name):
        """Mostra mensagem quando ferramenta em desenvolvimento é clicada"""
        messagebox.showinfo(
            "Ferramenta em Desenvolvimento",
            f"🚧 {tool_name}\n\n"
            "Esta ferramenta ainda está em desenvolvimento e será "
            "disponibilizada em uma próxima versão do Pro Sigma.\n\n"
            "Agradecemos sua compreensão!"
        )
    
    def _create_tooltip(self, widget, text):
        """Cria tooltip para o widget — criado uma vez e reutilizado"""
        tooltip = ctk.CTkLabel(
            widget,
            text=text,
            font=ctk.CTkFont(size=9),
            text_color="gray70",
            fg_color="gray10",
            corner_radius=6,
            padx=8,
            pady=4,
            wraplength=160,
        )

        def show_tooltip(event):
            tooltip.place(relx=0.5, rely=1.0, anchor="n", y=5)
            tooltip.lift()

        def hide_tooltip(event):
            tooltip.place_forget()

        widget.bind("<Enter>", show_tooltip)
        widget.bind("<Leave>", hide_tooltip)
    
    def _open_data_editor(self):
        from src.analytics.data_editor.data_editor_window import DataEditorWindow

        def _on_editor_data(df):
            self.current_data = df
            self.import_status_label.configure(
                text=f"✓ Dados do editor carregados: {len(df)} linhas × {len(df.columns)} colunas",
                text_color="#4CAF50"
            )
            self.show_file_info("Editor de Dados", len(df), len(df.columns))

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
            text=f"✓ Arquivo carregado: {file_name}",
            text_color="#4CAF50"
        )
        self.show_file_info(file_name, rows, cols)
        self.show_loading(False)

        messagebox.showinfo(
            "Sucesso",
            f"Arquivo carregado com sucesso!\n\nLinhas: {rows}\nColunas: {cols}"
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
        """
        Mostra informações detalhadas do arquivo carregado
        
        Args:
            filename: Nome do arquivo
            rows: Número de linhas
            cols: Número de colunas
        """
        # Remove info antiga se existir
        for widget in self.file_info_frame.winfo_children():
            widget.destroy()
        
        # Mostra o frame
        self.file_info_frame.pack(side="right", padx=(10, 0))
        
        # Título
        info_title = ctk.CTkLabel(
            self.file_info_frame,
            text="📊 Dados Carregados",
            font=ctk.CTkFont(size=12, weight="bold")
        )
        info_title.pack(padx=15, pady=(10, 5))
        
        # Nome do arquivo
        name_label = ctk.CTkLabel(
            self.file_info_frame,
            text=filename,
            font=ctk.CTkFont(size=11),
            wraplength=200
        )
        name_label.pack(padx=15, pady=(0, 5))
        
        # Estatísticas
        stats_frame = ctk.CTkFrame(self.file_info_frame, fg_color="transparent")
        stats_frame.pack(padx=15, pady=(0, 10))
        
        rows_label = ctk.CTkLabel(
            stats_frame,
            text=f"📋 {rows} linhas",
            font=ctk.CTkFont(size=10),
            text_color="gray"
        )
        rows_label.pack()
        
        cols_label = ctk.CTkLabel(
            stats_frame,
            text=f"📊 {cols} colunas",
            font=ctk.CTkFont(size=10),
            text_color="gray"
        )
        cols_label.pack()
    
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

        if feature_id == 'ishikawa':
            from src.analytics.ishikawa.ishikawa_window import IshikawaDiagramWindow
            IshikawaDiagramWindow(self)
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
        
        # Check if there's any data available (current or historical)
        has_current_data = self.current_data is not None
        has_history = len(self.file_history.get_recent_files(count=1)) > 0
        
        # If no current data and no history, show warning
        if not has_current_data and not has_history:
            response = messagebox.askyesno(
                "Nenhum Arquivo Disponível",
                "Você não tem nenhum arquivo carregado ou no histórico.\n\n"
                "Deseja importar um arquivo agora?"
            )
            if response:
                # Try to import a file
                self.import_excel()  # or show a choice dialog
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
            elif feature_id == 'descriptive_stats':
                from src.analytics.descriptise_stats.descriptive_stats_window import DescriptiveStatsWindow
                DescriptiveStatsWindow(self, selected_data)
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
        super().destroy()