"""
Ishikawa Diagram Window
Interface para criação de Diagramas de Ishikawa (Espinha de Peixe)
"""

import customtkinter as ctk
from tkinter import messagebox, filedialog
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

from src.analytics.ishikawa.ishikawa_utils import (
    create_ishikawa_diagram,
    export_diagram_to_png,
    validate_ishikawa_data
)


class IshikawaDiagramWindow(ctk.CTkToplevel):
    """Janela para criação de Diagrama de Ishikawa"""
    
    def __init__(self, parent):
        super().__init__(parent)
        self.title("Diagrama de Ishikawa (Espinha de Peixe)")
        self.geometry("1400x800")
        self.minsize(1200, 700)
        
        # Maximizar janela
        try:
            self.state("zoomed")
        except Exception:
            pass
        
        # Configurar como modal
        self.transient(parent)
        self.grab_set()
        
        self.categories = {}  # {category_name: [causes]}
        self.current_fig = None
        self.canvas_widget = None
        self.auto_update = True  # Atualização automática habilitada
        
        self._build_ui()
        self._show_initial_message()
    
    def _build_ui(self):
        """Constrói a interface"""
        main_frame = ctk.CTkFrame(self)
        main_frame.pack(fill="both", expand=True, padx=16, pady=16)
        
        # ===== PAINEL ESQUERDO (Inputs) =====
        left_container = ctk.CTkFrame(main_frame, width=400)
        left_container.pack(side="left", fill="both", expand=False, padx=(0, 8))
        left_container.pack_propagate(False)
        
        # Botões de ação (fixos no TOPO) - EMPACOTAR PRIMEIRO
        btn_frame = ctk.CTkFrame(left_container, fg_color="transparent")
        btn_frame.pack(fill="x", padx=8, pady=(8, 8), side="top")
        
        # Linha 1: Adicionar e Exemplo
        btn_row1 = ctk.CTkFrame(btn_frame, fg_color="transparent")
        btn_row1.pack(fill="x", pady=2)
        
        ctk.CTkButton(
            btn_row1,
            text="➕ Adicionar",
            command=self._add_category,
            height=32,
            width=140,
            fg_color="#2E86DE",
            hover_color="#1B5AA3"
        ).pack(side="left", padx=2, expand=True, fill="x")
        
        ctk.CTkButton(
            btn_row1,
            text="📋 Exemplo",
            command=self._create_sample_diagram,
            height=32,
            width=140,
            fg_color="#9B59B6",
            hover_color="#7D3C98"
        ).pack(side="left", padx=2, expand=True, fill="x")
        
        # Linha 2: Exportar e Limpar
        btn_row2 = ctk.CTkFrame(btn_frame, fg_color="transparent")
        btn_row2.pack(fill="x", pady=2)
        
        ctk.CTkButton(
            btn_row2,
            text="💾 Exportar",
            command=self._export_png,
            height=32,
            width=140,
            fg_color="#27AE60",
            hover_color="#1E8449"
        ).pack(side="left", padx=2, expand=True, fill="x")
        
        ctk.CTkButton(
            btn_row2,
            text="🗑️ Limpar",
            command=self._clear_all,
            height=32,
            width=140,
            fg_color="#E74C3C",
            hover_color="#C0392B"
        ).pack(side="left", padx=2, expand=True, fill="x")
        
        # Área scrollável para inputs - EMPACOTAR DEPOIS
        left_panel = ctk.CTkScrollableFrame(left_container)
        left_panel.pack(fill="both", expand=True, padx=0, pady=(8, 0))
        
        # Título
        ctk.CTkLabel(
            left_panel,
            text="Diagrama de Ishikawa",
            font=ctk.CTkFont(size=18, weight="bold")
        ).pack(pady=(8, 16))
        
        # Efeito Principal
        ctk.CTkLabel(
            left_panel,
            text="Efeito (Problema):",
            font=ctk.CTkFont(size=11, weight="bold")
        ).pack(anchor="w", padx=12, pady=(0, 4))
        
        self.effect_entry = ctk.CTkEntry(left_panel, height=35)
        self.effect_entry.pack(fill="x", padx=12, pady=(0, 12))
        self.effect_entry.insert(0, "Defeito no Produto")
        self.effect_entry.bind("<KeyRelease>", lambda e: self._schedule_auto_update())
        
        # Título do diagrama (opcional)
        ctk.CTkLabel(
            left_panel,
            text="Título (opcional):",
            font=ctk.CTkFont(size=11, weight="bold")
        ).pack(anchor="w", padx=12, pady=(0, 4))
        
        self.title_entry = ctk.CTkEntry(left_panel, height=35)
        self.title_entry.pack(fill="x", padx=12, pady=(0, 20))
        self.title_entry.insert(0, "Análise de Causa e Efeito")
        self.title_entry.bind("<KeyRelease>", lambda e: self._schedule_auto_update())
        
        # Seção de Categorias
        ctk.CTkLabel(
            left_panel,
            text="Categorias e Causas:",
            font=ctk.CTkFont(size=11, weight="bold")
        ).pack(anchor="w", padx=12, pady=(0, 4))
        
        ctk.CTkLabel(
            left_panel,
            text="Adicione até 8 categorias, cada uma com até 5 causas",
            text_color="gray",
            font=ctk.CTkFont(size=10)
        ).pack(anchor="w", padx=12, pady=(0, 12))
        
        # Container para categorias
        self.categories_container = ctk.CTkFrame(left_panel, fg_color="transparent")
        self.categories_container.pack(fill="x", padx=12, pady=(0, 12))
        
        # ===== PAINEL DIREITO (Diagrama) =====
        right_panel = ctk.CTkFrame(main_frame)
        right_panel.pack(side="right", fill="both", expand=True)
        
        # Título do painel
        ctk.CTkLabel(
            right_panel,
            text="Visualização do Diagrama",
            font=ctk.CTkFont(size=14, weight="bold")
        ).pack(anchor="w", padx=12, pady=(8, 8))
        
        # Container para o canvas do matplotlib
        self.diagram_container = ctk.CTkFrame(right_panel, fg_color="white")
        self.diagram_container.pack(fill="both", expand=True, padx=12, pady=(0, 12))
    
    def _add_category(self):
        """Adiciona uma nova categoria"""
        if len(self.categories_container.winfo_children()) >= 8:
            messagebox.showwarning("Aviso", "Máximo de 8 categorias atingido")
            return
        
        # Cria frame para a categoria
        cat_frame = ctk.CTkFrame(self.categories_container, border_width=1, border_color="gray")
        cat_frame.pack(fill="x", pady=6)
        
        # Header da categoria
        header_frame = ctk.CTkFrame(cat_frame, fg_color="transparent")
        header_frame.pack(fill="x", padx=8, pady=8)
        
        ctk.CTkLabel(
            header_frame,
            text="Categoria:",
            font=ctk.CTkFont(size=10, weight="bold")
        ).pack(side="left", padx=(0, 4))
        
        cat_entry = ctk.CTkEntry(header_frame, width=150)
        cat_entry.pack(side="left", padx=(0, 8))
        
        # Sugestões de categorias comuns (6M's)
        suggestions = ["Método", "Material", "Mão de Obra", "Máquina", "Medição", "Meio Ambiente"]
        cat_index = len(self.categories_container.winfo_children()) - 1
        if cat_index < len(suggestions):
            cat_entry.insert(0, suggestions[cat_index])
        
        # Botão remover
        remove_btn = ctk.CTkButton(
            header_frame,
            text="❌",
            width=30,
            command=lambda: self._remove_category(cat_frame),
            fg_color="#E74C3C",
            hover_color="#C0392B"
        )
        remove_btn.pack(side="right")
        
        # Container para causas
        causes_frame = ctk.CTkFrame(cat_frame, fg_color="transparent")
        causes_frame.pack(fill="x", padx=8, pady=(0, 8))
        
        ctk.CTkLabel(
            causes_frame,
            text="Causas (até 5):",
            font=ctk.CTkFont(size=10),
            text_color="gray"
        ).pack(anchor="w")
        
        # Campos para causas
        causes_entries = []
        for i in range(5):
            cause_entry = ctk.CTkEntry(causes_frame, placeholder_text=f"Causa {i+1}")
            cause_entry.pack(fill="x", pady=2)
            causes_entries.append(cause_entry)
        
        # Adiciona binds para atualização automática
        cat_entry.bind("<KeyRelease>", lambda e: self._schedule_auto_update())
        for cause_entry in causes_entries:
            cause_entry.bind("<KeyRelease>", lambda e: self._schedule_auto_update())
        
        # Armazena referências
        cat_frame.category_entry = cat_entry
        cat_frame.causes_entries = causes_entries
        
        # Atualiza diagrama automaticamente após adicionar
        if self.auto_update:
            self.after(100, self._update_diagram_silent)
    
    def _remove_category(self, cat_frame):
        """Remove uma categoria"""
        cat_frame.destroy()
        # Atualiza diagrama automaticamente após remover
        if self.auto_update:
            self.after(100, self._update_diagram_silent)
    
    def _get_categories_data(self):
        """Obtém os dados das categorias e causas"""
        categories = {}
        
        for cat_frame in self.categories_container.winfo_children():
            category_name = cat_frame.category_entry.get().strip()
            if not category_name:
                continue
            
            causes = []
            for cause_entry in cat_frame.causes_entries:
                cause_text = cause_entry.get().strip()
                if cause_text:
                    causes.append(cause_text)
            
            if causes:  # Só adiciona categoria se tiver pelo menos uma causa
                categories[category_name] = causes
        
        return categories
    
    def _update_diagram(self):
        """Atualiza o diagrama com os dados atuais"""
        effect = self.effect_entry.get().strip()
        title = self.title_entry.get().strip() or None
        categories = self._get_categories_data()
        
        # Valida dados
        is_valid, error_msg = validate_ishikawa_data(effect, categories)
        if not is_valid:
            messagebox.showerror("Erro de Validação", error_msg)
            return
        
        try:
            # Cria o diagrama
            self.current_fig = create_ishikawa_diagram(effect, categories, title)
            
            # Remove canvas anterior se existir
            if self.canvas_widget:
                self.canvas_widget.get_tk_widget().destroy()
            
            # Cria novo canvas
            self.canvas_widget = FigureCanvasTkAgg(self.current_fig, master=self.diagram_container)
            self.canvas_widget.draw()
            self.canvas_widget.get_tk_widget().pack(fill="both", expand=True)
            
            messagebox.showinfo("Sucesso", "Diagrama atualizado com sucesso!")
            
        except Exception as e:
            messagebox.showerror("Erro", f"Erro ao criar diagrama: {str(e)}")
    
    def _update_diagram_silent(self):
        """Atualiza o diagrama sem mensagens de sucesso (para atualização automática)"""
        effect = self.effect_entry.get().strip()
        title = self.title_entry.get().strip() or None
        categories = self._get_categories_data()
        
        # Valida dados silenciosamente
        is_valid, _ = validate_ishikawa_data(effect, categories)
        if not is_valid:
            return
        
        try:
            # Cria o diagrama
            self.current_fig = create_ishikawa_diagram(effect, categories, title)
            
            # Remove canvas anterior se existir
            if self.canvas_widget:
                self.canvas_widget.get_tk_widget().destroy()
            
            # Cria novo canvas
            self.canvas_widget = FigureCanvasTkAgg(self.current_fig, master=self.diagram_container)
            self.canvas_widget.draw()
            self.canvas_widget.get_tk_widget().pack(fill="both", expand=True)
            
        except Exception:
            pass  # Ignora erros na atualização automática
    
    def _schedule_auto_update(self):
        """Agenda atualização automática com debounce"""
        # Cancela timer anterior se existir
        if hasattr(self, '_update_timer'):
            self.after_cancel(self._update_timer)
        
        # Agenda nova atualização após 1 segundo de inatividade
        self._update_timer = self.after(1000, self._update_diagram_silent)
    
    def _show_initial_message(self):
        """Mostra mensagem inicial explicativa"""
        msg_frame = ctk.CTkFrame(self.diagram_container, fg_color="white")
        msg_frame.pack(expand=True)
        
        ctk.CTkLabel(
            msg_frame,
            text="🐟 Diagrama de Ishikawa",
            font=ctk.CTkFont(size=24, weight="bold"),
            text_color="#2E86DE"
        ).pack(pady=(20, 10))
        
        ctk.CTkLabel(
            msg_frame,
            text="Comece criando seu diagrama:",
            font=ctk.CTkFont(size=14),
            text_color="black"
        ).pack(pady=5)
        
        instructions = [
            "1. Defina o efeito/problema no campo acima",
            "2. Clique em '➕ Adicionar Categoria'",
            "3. Preencha as causas para cada categoria",
            "4. O diagrama atualiza automaticamente!",
            "",
            "Ou clique em '📋 Carregar Exemplo' para ver um modelo"
        ]
        
        for instruction in instructions:
            ctk.CTkLabel(
                msg_frame,
                text=instruction,
                font=ctk.CTkFont(size=11),
                text_color="gray"
            ).pack(pady=2)
    
    def _export_png(self):
        """Exporta o diagrama para PNG"""
        if not self.current_fig:
            messagebox.showwarning("Aviso", "Crie ou atualize o diagrama primeiro")
            return
        
        file_path = filedialog.asksaveasfilename(
            defaultextension=".png",
            filetypes=[("PNG Image", "*.png"), ("All Files", "*.*")],
            title="Salvar Diagrama como PNG"
        )
        
        if file_path:
            try:
                export_diagram_to_png(self.current_fig, file_path, dpi=300)
                messagebox.showinfo("Sucesso", f"Diagrama exportado para:\n{file_path}")
            except Exception as e:
                messagebox.showerror("Erro", f"Erro ao exportar: {str(e)}")
    
    def _clear_all(self):
        """Limpa todos os dados"""
        response = messagebox.askyesno(
            "Confirmar Limpeza",
            "Tem certeza que deseja limpar todos os dados?"
        )
        
        if response:
            # Limpa categorias
            for widget in self.categories_container.winfo_children():
                widget.destroy()
            
            # Remove canvas
            if self.canvas_widget:
                self.canvas_widget.get_tk_widget().destroy()
                self.canvas_widget = None
            
            self.current_fig = None
            
            # Limpa campos de texto
            self.effect_entry.delete(0, 'end')
            self.effect_entry.insert(0, "Defeito no Produto")
            self.title_entry.delete(0, 'end')
            self.title_entry.insert(0, "Análise de Causa e Efeito")
            
            # Mostra mensagem inicial novamente
            self._show_initial_message()
    
    def _create_sample_diagram(self):
        """Cria um diagrama de exemplo inicial"""
        # Adiciona categorias de exemplo (6M's)
        examples = [
            ("Método", ["Processo inadequado", "Falta de padronização"]),
            ("Material", ["Matéria-prima com defeito", "Armazenamento incorreto"]),
            ("Mão de Obra", ["Falta de treinamento", "Fadiga"]),
            ("Máquina", ["Equipamento desregulado", "Falta de manutenção"]),
            ("Medição", ["Instrumento descalibrado", "Erro de leitura"]),
            ("Meio Ambiente", ["Temperatura inadequada", "Umidade elevada"])
        ]
        
        for category, causes in examples[:4]:  # Adiciona 4 categorias de exemplo
            self._add_category()
            cat_frame = self.categories_container.winfo_children()[-1]
            cat_frame.category_entry.delete(0, 'end')
            cat_frame.category_entry.insert(0, category)
            
            for i, cause in enumerate(causes):
                if i < len(cat_frame.causes_entries):
                    cat_frame.causes_entries[i].insert(0, cause)
        
        # Atualiza o diagrama automaticamente
        self._update_diagram()
