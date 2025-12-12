"""
Janela de Histórico de Arquivos
"""
import customtkinter as ctk
from tkinter import messagebox
from pathlib import Path
from typing import Callable
from src.utils.file_history import FileHistory


class HistoryWindow(ctk.CTkToplevel):
    """Janela para visualizar e gerenciar histórico de arquivos"""
    
    def __init__(self, parent, on_file_selected: Callable):
        """
        Inicializa a janela de histórico
        
        Args:
            parent: Janela pai
            on_file_selected: Função callback quando um arquivo é selecionado
        """
        super().__init__(parent)
        
        self.on_file_selected = on_file_selected
        self.file_history = FileHistory()
        
        # Configurações da janela
        self.title("Pro Sigma - Histórico de Arquivos")
        self.geometry("900x600")
        
        # Centraliza a janela
        self.center_window()
        
        # Torna a janela modal
        self.transient(parent)
        self.grab_set()
        
        # Cria interface
        self.create_widgets()
        
    def center_window(self):
        """Centraliza a janela na tela"""
        self.update_idletasks()
        width = 900
        height = 600
        x = (self.winfo_screenwidth() // 2) - (width // 2)
        y = (self.winfo_screenheight() // 2) - (height // 2)
        self.geometry(f'{width}x{height}+{x}+{y}')
    
    def create_widgets(self):
        """Cria os widgets da interface"""
        
        # Container principal
        main_frame = ctk.CTkFrame(self, fg_color="transparent")
        main_frame.pack(expand=True, fill="both", padx=30, pady=25)
        
        # Header
        header_frame = ctk.CTkFrame(main_frame, fg_color="transparent")
        header_frame.pack(fill="x", pady=(0, 20))
        
        title_label = ctk.CTkLabel(
            header_frame,
            text="📚 Histórico de Arquivos",
            font=ctk.CTkFont(size=26, weight="bold")
        )
        title_label.pack(side="left")
        
        # Botões de ação
        button_container = ctk.CTkFrame(header_frame, fg_color="transparent")
        button_container.pack(side="right")
        
        refresh_btn = ctk.CTkButton(
            button_container,
            text="🔄 Atualizar",
            command=self.refresh_list,
            width=120,
            height=36,
            font=ctk.CTkFont(size=12),
            fg_color="gray30",
            hover_color="gray20"
        )
        refresh_btn.pack(side="left", padx=5)
        
        clean_btn = ctk.CTkButton(
            button_container,
            text="🗑️ Limpar",
            command=self.clean_missing,
            width=120,
            height=36,
            font=ctk.CTkFont(size=12),
            fg_color="gray30",
            hover_color="gray20"
        )
        clean_btn.pack(side="left", padx=5)
        
        # Tabs
        self.tabview = ctk.CTkTabview(main_frame)
        self.tabview.pack(fill="both", expand=True)
        
        # Tab: Recentes
        self.tabview.add("📁 Recentes")
        self.create_recent_tab()
        
        # Tab: Mais Usados
        self.tabview.add("⭐ Mais Usados")
        self.create_most_used_tab()
        
        # Tab: Todos
        self.tabview.add("📋 Todos")
        self.create_all_tab()
        
        # Informação no rodapé
        total_files = len(self.file_history.get_history())
        footer_label = ctk.CTkLabel(
            main_frame,
            text=f"Total de arquivos no histórico: {total_files}",
            font=ctk.CTkFont(size=11),
            text_color="gray"
        )
        footer_label.pack(pady=(15, 0))
    
    def create_recent_tab(self):
        """Cria a aba de arquivos recentes"""
        tab = self.tabview.tab("📁 Recentes")
        
        scroll = ctk.CTkScrollableFrame(tab, fg_color="transparent")
        scroll.pack(fill="both", expand=True, padx=10, pady=10)
        
        recent_files = self.file_history.get_recent_files(10)
        
        if not recent_files:
            empty_label = ctk.CTkLabel(
                scroll,
                text="Nenhum arquivo no histórico ainda.\nImporte um arquivo para começar!",
                font=ctk.CTkFont(size=13),
                text_color="gray"
            )
            empty_label.pack(pady=50)
        else:
            for file_info in recent_files:
                self.create_file_card(scroll, file_info)
    
    def create_most_used_tab(self):
        """Cria a aba de arquivos mais usados"""
        tab = self.tabview.tab("⭐ Mais Usados")
        
        scroll = ctk.CTkScrollableFrame(tab, fg_color="transparent")
        scroll.pack(fill="both", expand=True, padx=10, pady=10)
        
        most_used = self.file_history.get_most_used_files(10)
        
        if not most_used:
            empty_label = ctk.CTkLabel(
                scroll,
                text="Nenhum arquivo no histórico ainda.",
                font=ctk.CTkFont(size=13),
                text_color="gray"
            )
            empty_label.pack(pady=50)
        else:
            for file_info in most_used:
                self.create_file_card(scroll, file_info, show_usage=True)
    
    def create_all_tab(self):
        """Cria a aba com todos os arquivos"""
        tab = self.tabview.tab("📋 Todos")
        
        scroll = ctk.CTkScrollableFrame(tab, fg_color="transparent")
        scroll.pack(fill="both", expand=True, padx=10, pady=10)
        
        all_files = self.file_history.get_history()
        
        if not all_files:
            empty_label = ctk.CTkLabel(
                scroll,
                text="Nenhum arquivo no histórico ainda.",
                font=ctk.CTkFont(size=13),
                text_color="gray"
            )
            empty_label.pack(pady=50)
        else:
            for file_info in all_files:
                self.create_file_card(scroll, file_info, show_usage=True)
    
    def create_file_card(self, parent, file_info: dict, show_usage: bool = False):
        """
        Cria um card para um arquivo
        
        Args:
            parent: Widget pai
            file_info: Informações do arquivo
            show_usage: Se deve mostrar contador de uso
        """
        # Verifica se arquivo ainda existe
        file_exists = self.file_history.file_exists(file_info['file_path'])
        
        # Card
        card = ctk.CTkFrame(parent, corner_radius=8, border_width=1, border_color="gray25")
        card.pack(fill="x", pady=5, padx=5)
        
        # Container de conteúdo
        content = ctk.CTkFrame(card, fg_color="transparent")
        content.pack(fill="x", padx=15, pady=12)
        
        # Lado esquerdo: Info do arquivo
        left_side = ctk.CTkFrame(content, fg_color="transparent")
        left_side.pack(side="left", fill="x", expand=True)
        
        # Ícone + Nome
        name_frame = ctk.CTkFrame(left_side, fg_color="transparent")
        name_frame.pack(fill="x")
        
        icon = "📊" if file_info['file_type'] == 'excel' else "📄"
        if not file_exists:
            icon = "⚠️"
        
        icon_label = ctk.CTkLabel(
            name_frame,
            text=icon,
            font=ctk.CTkFont(size=16)
        )
        icon_label.pack(side="left", padx=(0, 8))
        
        name_label = ctk.CTkLabel(
            name_frame,
            text=file_info['file_name'],
            font=ctk.CTkFont(size=13, weight="bold"),
            anchor="w"
        )
        name_label.pack(side="left", fill="x", expand=True)
        
        # Info adicional
        info_text = (
            f"📋 {file_info['rows']} linhas  |  "
            f"📊 {file_info['cols']} colunas  |  "
            f"💾 {self.file_history.format_size(file_info['size_bytes'])}  |  "
            f"🕒 {self.file_history.format_date(file_info['last_accessed'])}"
        )
        
        if show_usage:
            info_text += f"  |  ⭐ {file_info.get('access_count', 1)}x usado"
        
        info_label = ctk.CTkLabel(
            left_side,
            text=info_text,
            font=ctk.CTkFont(size=10),
            text_color="gray60",
            anchor="w"
        )
        info_label.pack(fill="x", pady=(5, 0))
        
        # Caminho do arquivo
        path_label = ctk.CTkLabel(
            left_side,
            text=file_info['file_path'],
            font=ctk.CTkFont(size=9),
            text_color="gray50",
            anchor="w"
        )
        path_label.pack(fill="x", pady=(3, 0))
        
        # Lado direito: Botões
        right_side = ctk.CTkFrame(content, fg_color="transparent")
        right_side.pack(side="right", padx=(10, 0))
        
        if file_exists:
            open_btn = ctk.CTkButton(
                right_side,
                text="Abrir",
                command=lambda: self.open_file(file_info),
                width=100,
                height=32,
                font=ctk.CTkFont(size=12, weight="bold"),
                fg_color="#2E86DE",
                hover_color="#1E5BA8"
            )
            open_btn.pack(pady=2)
        else:
            missing_label = ctk.CTkLabel(
                right_side,
                text="Arquivo não\nencontrado",
                font=ctk.CTkFont(size=10),
                text_color="#FF5555"
            )
            missing_label.pack(pady=2)
        
        remove_btn = ctk.CTkButton(
            right_side,
            text="Remover",
            command=lambda: self.remove_file(file_info),
            width=100,
            height=28,
            font=ctk.CTkFont(size=11),
            fg_color="gray30",
            hover_color="gray20"
        )
        remove_btn.pack(pady=2)
    
    def open_file(self, file_info: dict):
        """
        Abre um arquivo do histórico
        
        Args:
            file_info: Informações do arquivo
        """
        self.file_history.update_access(file_info['file_path'])
        self.destroy()
        if self.on_file_selected:
            self.on_file_selected(file_info['file_path'], file_info['file_type'])
    
    def remove_file(self, file_info: dict):
        """
        Remove um arquivo do histórico
        
        Args:
            file_info: Informações do arquivo
        """
        response = messagebox.askyesno(
            "Remover do Histórico",
            f"Deseja remover '{file_info['file_name']}' do histórico?\n\n"
            "O arquivo não será deletado, apenas removido desta lista."
        )
        
        if response:
            self.file_history.remove_file(file_info['file_path'])
            self.refresh_list()
    
    def refresh_list(self):
        """Atualiza a lista de arquivos"""
        # Recarrega histórico
        self.file_history.history = self.file_history._load_history()
        
        # Recria as tabs
        self.destroy()
        self.__init__(self.master, self.on_file_selected)
    
    def clean_missing(self):
        """Remove arquivos que não existem mais"""
        response = messagebox.askyesno(
            "Limpar Arquivos Ausentes",
            "Deseja remover do histórico todos os arquivos que não existem mais?"
        )
        
        if response:
            before = len(self.file_history.get_history())
            self.file_history.clean_missing_files()
            after = len(self.file_history.get_history())
            removed = before - after
            
            messagebox.showinfo(
                "Limpeza Concluída",
                f"{removed} arquivo(s) removido(s) do histórico."
            )
            
            self.refresh_list()
