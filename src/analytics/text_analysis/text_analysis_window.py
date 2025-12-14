"""
Text Analysis Window
Interface for text mining and natural language processing
"""
import customtkinter as ctk
from tkinter import messagebox, scrolledtext
from src.utils.lazy_imports import get_pandas, get_numpy, get_matplotlib_figure, get_matplotlib_backend, get_matplotlib
from src.utils.ui_components import create_minitab_style_table

from .text_analysis_utils import (
    preprocess_text,
    extract_words,
    extract_phrases,
    count_words,
    count_phrases,
    search_keyword,
    create_wordcloud,
    create_word_frequency_chart,
    create_phrase_frequency_chart,
    calculate_text_statistics,
    STOPWORDS
)

# Lazy-loaded libraries
_pd = None
_np = None
_plt = None
_Figure = None
_FigureCanvasTkAgg = None

def _ensure_libs():
    """Ensure heavy libraries are loaded"""
    global _pd, _np, _plt, _Figure, _FigureCanvasTkAgg
    if _pd is None:
        _pd = get_pandas()
        _np = get_numpy()
        _plt = get_matplotlib()
        _Figure = get_matplotlib_figure()
        _FigureCanvasTkAgg = get_matplotlib_backend()
    return _pd, _np, _plt, _Figure, _FigureCanvasTkAgg


class TextAnalysisWindow(ctk.CTkToplevel):
    def __init__(self, parent, df):
        super().__init__(parent)
        
        # Load heavy libraries (lazy)
        pd, np, plt, Figure, FigureCanvasTkAgg = _ensure_libs()
        self.pd = pd
        self.np = np
        self.plt = plt
        self.Figure = Figure
        self.FigureCanvasTkAgg = FigureCanvasTkAgg
        
        self.df = df
        self.processed_texts = []
        self.all_words = []
        self.all_phrases = []
        
        # Window configuration
        self.title("Análise de Texto")
        
        # Allow resizing
        self.resizable(True, True)
        self.minsize(1200, 800)
        
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
            text="📝 Análise de Texto",
            font=ctk.CTkFont(size=24, weight="bold")
        )
        title.pack(pady=(0, 20))
        
        # Description
        desc = ctk.CTkLabel(
            self.main_container,
            text="Análise de texto, mineração de dados e processamento de linguagem natural",
            font=ctk.CTkFont(size=12),
            text_color="gray"
        )
        desc.pack(pady=(0, 20))
        
        # Configuration Frame
        config_frame = ctk.CTkFrame(self.main_container)
        config_frame.pack(fill="x", pady=(0, 20))
        
        # Column Selection
        column_frame = ctk.CTkFrame(config_frame)
        column_frame.pack(fill="x", padx=20, pady=10)
        
        ctk.CTkLabel(
            column_frame,
            text="Coluna de Texto:",
            font=ctk.CTkFont(size=14, weight="bold")
        ).pack(side="left", padx=(10, 10))
        
        self.text_column_var = ctk.StringVar()
        text_column_menu = ctk.CTkOptionMenu(
            column_frame,
            variable=self.text_column_var,
            values=list(self.df.columns),
            width=300,
            font=ctk.CTkFont(size=12)
        )
        text_column_menu.pack(side="left", padx=10)
        
        # Language Selection
        language_frame = ctk.CTkFrame(config_frame)
        language_frame.pack(fill="x", padx=20, pady=10)
        
        ctk.CTkLabel(
            language_frame,
            text="Idioma:",
            font=ctk.CTkFont(size=14, weight="bold")
        ).pack(side="left", padx=(10, 10))
        
        self.language_var = ctk.StringVar(value="portuguese")
        
        ctk.CTkRadioButton(
            language_frame,
            text="Português",
            variable=self.language_var,
            value="portuguese",
            font=ctk.CTkFont(size=12)
        ).pack(side="left", padx=10)
        
        ctk.CTkRadioButton(
            language_frame,
            text="English",
            variable=self.language_var,
            value="english",
            font=ctk.CTkFont(size=12)
        ).pack(side="left", padx=10)
        
        ctk.CTkRadioButton(
            language_frame,
            text="Español",
            variable=self.language_var,
            value="spanish",
            font=ctk.CTkFont(size=12)
        ).pack(side="left", padx=10)
        
        # Keyword Search Frame
        keyword_frame = ctk.CTkFrame(config_frame)
        keyword_frame.pack(fill="x", padx=20, pady=10)
        
        ctk.CTkLabel(
            keyword_frame,
            text="Buscar Palavra/Frase (opcional):",
            font=ctk.CTkFont(size=14, weight="bold")
        ).pack(side="left", padx=(10, 10))
        
        self.keyword_entry = ctk.CTkEntry(
            keyword_frame,
            width=300,
            placeholder_text="Digite a palavra ou frase para buscar",
            font=ctk.CTkFont(size=12)
        )
        self.keyword_entry.pack(side="left", padx=10)
        
        self.case_sensitive_var = ctk.BooleanVar(value=False)
        ctk.CTkCheckBox(
            keyword_frame,
            text="Diferenciar maiúsculas/minúsculas",
            variable=self.case_sensitive_var,
            font=ctk.CTkFont(size=11)
        ).pack(side="left", padx=10)
        
        # Custom Stopwords Frame
        stopwords_frame = ctk.CTkFrame(config_frame)
        stopwords_frame.pack(fill="x", padx=20, pady=10)
        
        ctk.CTkLabel(
            stopwords_frame,
            text="Palavras a Ignorar (opcional):",
            font=ctk.CTkFont(size=14, weight="bold")
        ).pack(anchor="w", padx=10, pady=(10, 5))
        
        ctk.CTkLabel(
            stopwords_frame,
            text="Digite palavras separadas por vírgula (ex: teste, exemplo, abc)",
            font=ctk.CTkFont(size=10),
            text_color="gray"
        ).pack(anchor="w", padx=10, pady=(0, 5))
        
        self.custom_stopwords_entry = ctk.CTkEntry(
            stopwords_frame,
            width=600,
            placeholder_text="palavras, para, ignorar, separadas, por, vírgula",
            font=ctk.CTkFont(size=11)
        )
        self.custom_stopwords_entry.pack(padx=10, pady=(0, 10), fill="x")
        
        # Custom Ignore Characters Frame
        ignore_chars_frame = ctk.CTkFrame(config_frame)
        ignore_chars_frame.pack(fill="x", padx=20, pady=10)
        
        ctk.CTkLabel(
            ignore_chars_frame,
            text="Caracteres a Ignorar (opcional):",
            font=ctk.CTkFont(size=14, weight="bold")
        ).pack(anchor="w", padx=10, pady=(10, 5))
        
        ctk.CTkLabel(
            ignore_chars_frame,
            text="Digite caracteres que deseja remover (ex: #@!$%)",
            font=ctk.CTkFont(size=10),
            text_color="gray"
        ).pack(anchor="w", padx=10, pady=(0, 5))
        
        self.ignore_chars_entry = ctk.CTkEntry(
            ignore_chars_frame,
            width=300,
            placeholder_text="#@!$%&*()[]{}",
            font=ctk.CTkFont(size=11)
        )
        self.ignore_chars_entry.pack(padx=10, pady=(0, 10))
        
        # Options Frame
        options_frame = ctk.CTkFrame(config_frame)
        options_frame.pack(fill="x", padx=20, pady=10)
        
        ctk.CTkLabel(
            options_frame,
            text="Opções de Análise:",
            font=ctk.CTkFont(size=14, weight="bold")
        ).pack(anchor="w", padx=10, pady=(10, 5))
        
        options_inner = ctk.CTkFrame(options_frame)
        options_inner.pack(fill="x", padx=10, pady=(0, 10))
        
        self.use_stopwords_var = ctk.BooleanVar(value=True)
        ctk.CTkCheckBox(
            options_inner,
            text="Remover stopwords automáticas",
            variable=self.use_stopwords_var,
            font=ctk.CTkFont(size=12)
        ).pack(side="left", padx=10)
        
        self.show_phrases_var = ctk.BooleanVar(value=True)
        ctk.CTkCheckBox(
            options_inner,
            text="Análise de frases (bigramas)",
            variable=self.show_phrases_var,
            font=ctk.CTkFont(size=12)
        ).pack(side="left", padx=10)
        
        self.show_wordcloud_var = ctk.BooleanVar(value=True)
        ctk.CTkCheckBox(
            options_inner,
            text="Gerar nuvem de palavras",
            variable=self.show_wordcloud_var,
            font=ctk.CTkFont(size=12)
        ).pack(side="left", padx=10)
        
        # N-gram selection
        ngram_frame = ctk.CTkFrame(options_frame)
        ngram_frame.pack(fill="x", padx=10, pady=(0, 10))
        
        ctk.CTkLabel(
            ngram_frame,
            text="Tamanho das frases (n-gram):",
            font=ctk.CTkFont(size=12)
        ).pack(side="left", padx=(10, 10))
        
        self.ngram_var = ctk.IntVar(value=2)
        for i in range(2, 5):
            ctk.CTkRadioButton(
                ngram_frame,
                text=f"{i} palavras",
                variable=self.ngram_var,
                value=i,
                font=ctk.CTkFont(size=11)
            ).pack(side="left", padx=5)
        
        # Generate button
        generate_btn = ctk.CTkButton(
            config_frame,
            text="🔍 Analisar Texto",
            command=self.generate_analysis,
            font=ctk.CTkFont(size=14, weight="bold"),
            height=40
        )
        generate_btn.pack(pady=20)
        
        # Results container (initially empty)
        self.results_container = ctk.CTkFrame(self.main_container)
        self.results_container.pack(fill="both", expand=True, pady=(0, 20))
    
    def generate_analysis(self):
        """Generate text analysis"""
        try:
            # Get selected column
            text_column = self.text_column_var.get()
            
            if not text_column:
                messagebox.showerror(
                    "Erro",
                    "Por favor, selecione uma coluna de texto."
                )
                return
            
            # Get texts
            texts = self.df[text_column].tolist()
            
            # Clear previous results
            for widget in self.results_container.winfo_children():
                widget.destroy()
            
            # Get options
            language = self.language_var.get()
            use_stopwords = self.use_stopwords_var.get()
            show_phrases = self.show_phrases_var.get()
            show_wordcloud = self.show_wordcloud_var.get()
            ngram = self.ngram_var.get()
            
            # Get custom stopwords
            custom_stopwords = set()
            if self.custom_stopwords_entry.get().strip():
                words = self.custom_stopwords_entry.get().strip().split(',')
                custom_stopwords = {word.strip().lower() for word in words if word.strip()}
            
            # Get ignore characters
            ignore_chars = self.ignore_chars_entry.get().strip()
            
            # Preprocess texts
            if use_stopwords:
                self.processed_texts = [
                    preprocess_text(
                        text,
                        language=language,
                        custom_stopwords=custom_stopwords,
                        custom_ignore_chars=ignore_chars
                    )
                    for text in texts
                ]
            else:
                # Minimal preprocessing (no stopword removal)
                self.processed_texts = []
                for text in texts:
                    if self.pd.isna(text) or not isinstance(text, str):
                        self.processed_texts.append("")
                    else:
                        t = text.lower()
                        if ignore_chars:
                            for char in ignore_chars:
                                t = t.replace(char, ' ')
                        self.processed_texts.append(t)
            
            # Extract words and phrases
            self.all_words = extract_words(self.processed_texts)
            if show_phrases:
                self.all_phrases = extract_phrases(self.processed_texts, n_gram=ngram)
            
            # Show statistics
            self.show_statistics(texts)
            
            # Show keyword search results if keyword provided
            keyword = self.keyword_entry.get().strip()
            if keyword:
                self.show_keyword_search(texts, keyword)
            
            # Show word analysis
            self.show_word_analysis()
            
            # Show phrase analysis if enabled
            if show_phrases and self.all_phrases:
                self.show_phrase_analysis()
            
            # Show visualizations
            if show_wordcloud:
                self.show_wordcloud()
            
            self.show_word_frequency_chart()
            
            if show_phrases and self.all_phrases:
                self.show_phrase_frequency_chart()
            
        except Exception as e:
            messagebox.showerror(
                "Erro",
                f"Erro ao gerar análise:\n{str(e)}"
            )
    
    def show_statistics(self, texts):
        """Show text statistics"""
        stats_frame = ctk.CTkFrame(self.results_container)
        stats_frame.pack(fill="x", padx=10, pady=10)
        
        ctk.CTkLabel(
            stats_frame,
            text="📊 Estatísticas Gerais",
            font=ctk.CTkFont(size=16, weight="bold")
        ).pack(pady=(10, 5))
        
        stats_df = calculate_text_statistics(texts)
        
        headers = stats_df.columns.tolist()
        data_rows = stats_df.values.tolist()
        
        create_minitab_style_table(
            stats_frame,
            headers=headers,
            data_rows=data_rows,
            title="Estatísticas do Texto"
        )
    
    def show_keyword_search(self, texts, keyword):
        """Show keyword search results"""
        case_sensitive = self.case_sensitive_var.get()
        results = search_keyword(texts, keyword, case_sensitive)
        
        search_frame = ctk.CTkFrame(self.results_container)
        search_frame.pack(fill="x", padx=10, pady=10)
        
        ctk.CTkLabel(
            search_frame,
            text=f"🔍 Resultados da Busca: '{keyword}'",
            font=ctk.CTkFont(size=16, weight="bold")
        ).pack(pady=(10, 5))
        
        # Summary
        summary_text = f"""
Total de ocorrências: {results['total_occurrences']}
Documentos com a palavra: {results['documents_with_keyword']} de {len(texts)} ({results['documents_with_keyword']/len(texts)*100:.1f}%)
        """
        
        summary_label = ctk.CTkLabel(
            search_frame,
            text=summary_text,
            font=ctk.CTkFont(size=12),
            justify="left"
        )
        summary_label.pack(pady=10, padx=20)
        
        # Show first few contexts
        if results['contexts']:
            ctk.CTkLabel(
                search_frame,
                text="Contextos (primeiros 5):",
                font=ctk.CTkFont(size=12, weight="bold")
            ).pack(pady=(10, 5), padx=20, anchor="w")
            
            context_text = ctk.CTkTextbox(search_frame, height=150, font=ctk.CTkFont(size=10))
            context_text.pack(padx=20, pady=10, fill="x")
            
            for i, ctx in enumerate(results['contexts'][:5]):
                context_text.insert("end", f"[Doc {ctx['document_index']}] ...{ctx['context']}...\n\n")
            
            context_text.configure(state="disabled")
    
    def show_word_analysis(self):
        """Show word count table"""
        if not self.all_words:
            return
        
        word_frame = ctk.CTkFrame(self.results_container)
        word_frame.pack(fill="x", padx=10, pady=10)
        
        ctk.CTkLabel(
            word_frame,
            text="📝 Análise de Palavras",
            font=ctk.CTkFont(size=16, weight="bold")
        ).pack(pady=(10, 5))
        
        word_counts_df = count_words(self.all_words, top_n=50)
        
        headers = word_counts_df.columns.tolist()
        data_rows = word_counts_df.values.tolist()
        
        create_minitab_style_table(
            word_frame,
            headers=headers,
            data_rows=data_rows,
            title="Top 50 Palavras Mais Frequentes"
        )
    
    def show_phrase_analysis(self):
        """Show phrase count table"""
        if not self.all_phrases:
            return
        
        phrase_frame = ctk.CTkFrame(self.results_container)
        phrase_frame.pack(fill="x", padx=10, pady=10)
        
        ctk.CTkLabel(
            phrase_frame,
            text="💬 Análise de Frases",
            font=ctk.CTkFont(size=16, weight="bold")
        ).pack(pady=(10, 5))
        
        phrase_counts_df = count_phrases(self.all_phrases, top_n=30)
        
        headers = phrase_counts_df.columns.tolist()
        data_rows = phrase_counts_df.values.tolist()
        
        create_minitab_style_table(
            phrase_frame,
            headers=headers,
            data_rows=data_rows,
            title="Top 30 Frases Mais Frequentes"
        )
    
    def show_wordcloud(self):
        """Show word cloud"""
        if not self.all_words:
            return
        
        cloud_frame = ctk.CTkFrame(self.results_container)
        cloud_frame.pack(fill="both", expand=True, padx=10, pady=10)
        
        ctk.CTkLabel(
            cloud_frame,
            text="☁️ Nuvem de Palavras",
            font=ctk.CTkFont(size=16, weight="bold")
        ).pack(pady=(10, 5))
        
        fig = create_wordcloud(self.all_words, max_words=100)
        
        canvas = self.FigureCanvasTkAgg(fig, master=cloud_frame)
        canvas.draw()
        canvas.get_tk_widget().pack(fill="both", expand=True, padx=10, pady=10)
    
    def show_word_frequency_chart(self):
        """Show word frequency bar chart"""
        if not self.all_words:
            return
        
        chart_frame = ctk.CTkFrame(self.results_container)
        chart_frame.pack(fill="both", expand=True, padx=10, pady=10)
        
        ctk.CTkLabel(
            chart_frame,
            text="📊 Gráfico de Frequência de Palavras",
            font=ctk.CTkFont(size=16, weight="bold")
        ).pack(pady=(10, 5))
        
        fig = create_word_frequency_chart(self.all_words, top_n=20, orientation='horizontal')
        
        canvas = self.FigureCanvasTkAgg(fig, master=chart_frame)
        canvas.draw()
        canvas.get_tk_widget().pack(fill="both", expand=True, padx=10, pady=10)
    
    def show_phrase_frequency_chart(self):
        """Show phrase frequency bar chart"""
        if not self.all_phrases:
            return
        
        chart_frame = ctk.CTkFrame(self.results_container)
        chart_frame.pack(fill="both", expand=True, padx=10, pady=10)
        
        ctk.CTkLabel(
            chart_frame,
            text="📊 Gráfico de Frequência de Frases",
            font=ctk.CTkFont(size=16, weight="bold")
        ).pack(pady=(10, 5))
        
        fig = create_phrase_frequency_chart(self.all_phrases, top_n=15)
        
        canvas = self.FigureCanvasTkAgg(fig, master=chart_frame)
        canvas.draw()
        canvas.get_tk_widget().pack(fill="both", expand=True, padx=10, pady=10)
