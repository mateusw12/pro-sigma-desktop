"""
Página Inicial do Pro Sigma
Importação de dados e seleção de ferramentas
"""
import customtkinter as ctk
from tkinter import filedialog, messagebox
import pandas as pd
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
        self._is_resizing = False
        self._resize_after_id = None
        
        self.create_widgets()
        
        # Bind para otimizar redimensionamento
        self.bind('<Configure>', self._on_configure)
    
    def create_widgets(self):
        """Cria os widgets da interface"""
        
        # Container principal com menu lateral
        main_container = ctk.CTkFrame(self, fg_color="transparent")
        main_container.pack(fill="both", expand=True)
        
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
        self.import_csv_btn.pack(side="left")
        
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
        tools_header.pack(fill="x", pady=(0, 10))
        
        tools_title = ctk.CTkLabel(
            tools_header,
            text="🔧 Ferramentas Disponíveis",
            font=ctk.CTkFont(size=18, weight="bold")
        )
        tools_title.pack(side="left")
        
        # Badge com número de ferramentas
        features_count = len(self.license_data.get('features', []))
        count_badge = ctk.CTkLabel(
            tools_header,
            text=f"{features_count} ferramentas",
            font=ctk.CTkFont(size=11, weight="bold"),
            text_color="white",
            fg_color="#2E86DE",
            corner_radius=12,
            padx=12,
            pady=4
        )
        count_badge.pack(side="left", padx=(10, 0))
        
        # Grid de ferramentas (otimizado para performance)
        self.tools_scroll = ctk.CTkScrollableFrame(
            tools_container,
            fg_color="transparent",
            scrollbar_button_color="#2E86DE",
            scrollbar_button_hover_color="#1E5BA8"
        )
        self.tools_scroll.pack(fill="both", expand=True)
        
        # Otimiza scrolling para melhor performance
        self.tools_scroll._scrollbar.configure(width=12)
        
        # Cria botões das ferramentas
        self.create_tool_buttons()
    
    def create_tool_buttons(self):
        """Cria botões para as ferramentas disponíveis"""
        
        # Definição das ferramentas
        tools_definition = {
            'variability': {
                'title': 'Análise de Variabilidade',
                'description': 'Análise de variabilidade de dados',
                'plan': 'basic'
            },
            'process_capability': {
                'title': 'Process Capability',
                'description': 'Cálculo de Cp, Cpk, Pp, Ppk',
                'plan': 'basic'
            },
            'hypothesis_test': {
                'title': 'Testes de Hipótese',
                'description': 'Testes T, Z, ANOVA, Qui-quadrado',
                'plan': 'basic'
            },
            'distribution_test': {
                'title': 'Teste de Distribuição',
                'description': 'Ajuste de distribuições (Normal, Weibull, etc)',
                'plan': 'basic'
            },
            'cov_ems': {
                'title': 'COV EMS',
                'description': 'Análise de coeficiente de variação',
                'plan': 'basic'
            },
            'analytics': {
                'title': 'Analytics',
                'description': 'Análise e formatação de dados',
                'plan': 'basic'
            },
            'descriptive_stats': {
                'title': 'Descriptive Statistics',
                'description': 'Histograms, boxplots, summary metrics',
                'plan': 'basic'
            },
            'text_analysis': {
                'title': 'Text Analysis',
                'description': 'Análise textual e frequência de palavras',
                'plan': 'intermediate'
            },
            'normalization_test': {
                'title': 'Testes de Normalidade',
                'description': 'Shapiro-Wilk, Kolmogorov-Smirnov, etc',
                'plan': 'intermediate'
            },
            'control_charts': {
                'title': 'Cartas de Controle',
                'description': 'X-bar, R, S, P, NP, C, U',
                'plan': 'intermediate'
            },
            'dashboard': {
                'title': 'Dashboard',
                'description': 'Visualização de métricas',
                'plan': 'intermediate'
            },
            'monte_carlo': {
                'title': 'Monte Carlo',
                'description': 'Simulações Monte Carlo',
                'plan': 'intermediate'
            },
            'cov_ems': {
                'title': 'COV EMS',
                'description': 'Análise de coeficiente de variação',
                'plan': 'basic'
            },
            'simple_regression': {
                'title': 'Regressão Simples',
                'description': 'Regressão linear simples',
                'plan': 'pro'
            },
            'multiple_regression': {
                'title': 'Regressão Múltipla',
                'description': 'Regressão linear múltipla',
                'plan': 'pro'
            },
            'multivariate': {
                'title': 'Análise Multivariada',
                'description': 'PCA, Análise Fatorial, Cluster',
                'plan': 'pro'
            },
            'stackup': {
                'title': 'StackUp',
                'description': 'Análise de tolerâncias 2D',
                'plan': 'pro'
            },
            'doe': {
                'title': 'DOE',
                'description': 'Design of Experiments',
                'plan': 'pro'
            },
            'space_filling': {
                'title': 'Space Filling',
                'description': 'Latin Hypercube',
                'plan': 'pro'
            },
            'warranty_costs': {
                'title': 'Custos de Garantia',
                'description': 'Análise de custos de garantia',
                'plan': 'pro'
            },
            'neural_networks': {
                'title': 'Redes Neurais',
                'description': 'Análise de redes neurais',
                'plan': 'pro'
            },
            'decision_tree': {
                'title': 'Árvore de Decisão',
                'description': 'Análise de árvore de decisão',
                'plan': 'pro'
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
        
        # Cria seção para cada categoria que tenha ferramentas
        for category_name, tools_list in categories.items():
            if not tools_list:
                continue
                
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
            max_cols = 3  # 3 cards por linha
            
            for idx, (feature_id, tool_info) in enumerate(tools_list):
                # Cria nova linha a cada max_cols cards
                if col_count == 0:
                    row_frame = ctk.CTkFrame(self.tools_scroll, fg_color="transparent")
                    row_frame.pack(fill="x", pady=5)
                
                # Card da ferramenta
                tool_card = ctk.CTkFrame(row_frame, corner_radius=10, border_width=1, border_color="gray25")
                tool_card.pack(side="left", fill="both", expand=True, padx=5, pady=5)
                
                # Efeito hover (simulado com bind)
                def on_enter(event, card=tool_card):
                    card.configure(border_color="#2E86DE")
                
                def on_leave(event, card=tool_card):
                    card.configure(border_color="gray25")
                
                tool_card.bind("<Enter>", on_enter)
                tool_card.bind("<Leave>", on_leave)
            
                # Container interno do card
                card_content = ctk.CTkFrame(tool_card, fg_color="transparent")
                card_content.pack(fill="both", expand=True, padx=15, pady=12)
                
                # Ícone/Badge grande no topo
                icon_map = {
                    'variability': '📊', 'process_capability': '📈', 'hypothesis_test': '🔬',
                    'distribution_test': '📉', 'cov_ems': '📐', 'distribution_analysis': '📊',
                    'analytics': '🔍', 'text_analysis': '📝', 'normalization_test': '✓',
                    'control_charts': '📊', 'dashboard': '📊', 'monte_carlo': '🎲', 'cov_ems': '🎲',
                    'simple_regression': '📈', 'multiple_regression': '📈', 'multivariate': '🔄',
                    'stackup': '📏', 'doe': '🧪', 'space_filling': '⬜', 'warranty_costs': '💰',
                    'neural_networks': '🧠', 'decision_tree': '🌳', 'descriptive_stats': '📊'
                }
                icon = icon_map.get(feature_id, '🔧')
                
                icon_label = ctk.CTkLabel(
                    card_content,
                    text=icon,
                    font=ctk.CTkFont(size=32)
                )
                icon_label.pack(pady=(5, 8))
                
                # Título da ferramenta
                title_label = ctk.CTkLabel(
                    card_content,
                    text=tool_info['title'],
                    font=ctk.CTkFont(size=13, weight="bold"),
                    anchor="center",
                    wraplength=200
                )
                title_label.pack(pady=(0, 6))
                
                # Descrição
                desc_label = ctk.CTkLabel(
                    card_content,
                    text=tool_info['description'],
                    font=ctk.CTkFont(size=10),
                    text_color="gray60",
                    anchor="center",
                    wraplength=200,
                    height=40
                )
                desc_label.pack(pady=(0, 10))
                
                # Botão de ação compacto
                action_btn = ctk.CTkButton(
                    card_content,
                    text="Abrir →",
                    command=lambda fid=feature_id: self.open_tool(fid),
                    height=32,
                    font=ctk.CTkFont(size=11, weight="bold"),
                    fg_color="#2E86DE",
                    hover_color="#1E5BA8",
                    corner_radius=6
                )
                action_btn.pack(fill="x")
                
                col_count += 1
                if col_count >= max_cols:
                    col_count = 0
    
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
        """
        Carrega arquivo de dados
        
        Args:
            file_path: Caminho do arquivo
            file_type: Tipo do arquivo (excel ou csv)
        """
        # Mostra loading
        self.show_loading(True)
        
        # Usa after para não travar a UI
        self.after(100, lambda: self._load_file_async(file_path, file_type))
    
    def _load_file_async(self, file_path: str, file_type: str):
        """
        Carrega arquivo de forma assíncrona
        
        Args:
            file_path: Caminho do arquivo
            file_type: Tipo do arquivo (excel ou csv)
        """
        try:
            # Carrega dados
            if file_type == 'excel':
                self.current_data = pd.read_excel(file_path)
            else:
                self.current_data = pd.read_csv(file_path)
            
            self.current_file_path = file_path
            
            # Atualiza status
            file_name = Path(file_path).name
            rows, cols = self.current_data.shape
            
            # Adiciona ao histórico
            self.file_history.add_file(file_path, file_type, rows, cols)
            
            self.import_status_label.configure(
                text=f"✓ Arquivo carregado: {file_name}",
                text_color="#4CAF50"
            )
            
            # Mostra info do arquivo
            self.show_file_info(file_name, rows, cols)
            
            # Remove loading
            self.show_loading(False)
            
            messagebox.showinfo(
                "Sucesso",
                f"Arquivo carregado com sucesso!\n\n"
                f"Linhas: {rows}\n"
                f"Colunas: {cols}"
            )
            
        except Exception as e:
            # Remove loading
            self.show_loading(False)
            
            messagebox.showerror(
                "Erro",
                f"Erro ao carregar arquivo:\n{str(e)}"
            )
    
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
                from src.analytics.capability_window import CapabilityWindow
                capability_window = CapabilityWindow(self, selected_data)
            elif feature_id == 'cov_ems':
                from src.analytics.cov_window import CovEmsWindow
                cov_window = CovEmsWindow(self, selected_data)
            elif feature_id == 'hypothesis_test':
                from src.analytics.hypothesis_test.hypothesis_test_window import HypothesisTestWindow
                hypothesis_window = HypothesisTestWindow(self, selected_data)
            elif feature_id == 'descriptive_stats':
                from src.analytics.descriptive_stats_window import DescriptiveStatsWindow
                DescriptiveStatsWindow(self, selected_data)
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
        # Ignora eventos que não são da própria janela
        if event.widget != self:
            return
        
        # Cancela timer anterior se existir
        if self._resize_after_id:
            self.after_cancel(self._resize_after_id)
        
        # Agenda atualização após 150ms de inatividade
        self._resize_after_id = self.after(150, self._handle_resize)
    
    def _handle_resize(self):
        """
        Processa o redimensionamento de forma otimizada
        """
        self._is_resizing = False
        self._resize_after_id = None
        
        # Força uma única atualização após o resize
        self.update_idletasks()