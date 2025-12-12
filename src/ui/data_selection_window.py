"""
Data Selection Window
Allows user to choose between current data or historical files
"""
import customtkinter as ctk
from tkinter import messagebox
import pandas as pd
from pathlib import Path
from typing import Optional, Callable
from src.utils.file_history import FileHistory


class DataSelectionWindow(ctk.CTkToplevel):
    def __init__(self, parent, current_data: Optional[pd.DataFrame], current_file_path: Optional[str], on_select: Callable):
        super().__init__(parent)
        
        self.current_data = current_data
        self.current_file_path = current_file_path
        self.on_select = on_select
        self.file_history = FileHistory()
        self.selected_data = None
        self.selected_file_path = None
        
        # Window configuration
        self.title("Selecionar Dados para Análise")
        self.geometry("900x700")
        
        # Allow resizing
        self.resizable(True, True)
        self.minsize(600, 400)
        
        # Center window
        self.update_idletasks()
        x = (self.winfo_screenwidth() // 2) - (450)
        y = (self.winfo_screenheight() // 2) - (350)
        self.geometry(f'900x700+{x}+{y}')
        
        # Make modal
        self.transient(parent)
        self.grab_set()
        
        self.create_widgets()
    
    def create_widgets(self):
        # Title
        title = ctk.CTkLabel(
            self,
            text="📂 Selecionar Dados para Análise",
            font=ctk.CTkFont(size=20, weight="bold")
        )
        title.pack(pady=20)
        
        # Info label
        info_label = ctk.CTkLabel(
            self,
            text="Escolha qual arquivo deseja usar para a análise:",
            font=ctk.CTkFont(size=12),
            text_color="gray"
        )
        info_label.pack(pady=(0, 10))
        
        # Current file section (if available)
        if self.current_data is not None:
            current_frame = ctk.CTkFrame(self)
            current_frame.pack(fill="x", padx=20, pady=(0, 10))
            
            ctk.CTkLabel(
                current_frame,
                text="📊 Arquivo Atual",
                font=ctk.CTkFont(size=16, weight="bold")
            ).pack(pady=10)
            
            # File info
            if self.current_file_path:
                file_name = Path(self.current_file_path).name
            else:
                file_name = "Arquivo em memória"
            
            info_text = f"{file_name}\n{self.current_data.shape[0]} linhas × {self.current_data.shape[1]} colunas"
            
            ctk.CTkLabel(
                current_frame,
                text=info_text,
                font=ctk.CTkFont(size=12),
                text_color="gray"
            ).pack(pady=(0, 10))
            
            use_current_btn = ctk.CTkButton(
                current_frame,
                text="✓ Usar Arquivo Atual",
                command=self.use_current_data,
                height=40,
                font=ctk.CTkFont(size=14, weight="bold"),
                fg_color="#2E86DE"
            )
            use_current_btn.pack(pady=(0, 10), padx=20, fill="x")
        
        # Separator if there's current data
        if self.current_data is not None:
            separator_label = ctk.CTkLabel(
                self,
                text="─── ou ───",
                font=ctk.CTkFont(size=12),
                text_color="gray50"
            )
            separator_label.pack(pady=10)
        
        # History section
        history_frame = ctk.CTkFrame(self)
        history_frame.pack(fill="both", expand=True, padx=20, pady=(10, 20))
        
        ctk.CTkLabel(
            history_frame,
            text="📜 Arquivos Salvos no Histórico" if self.current_data is not None else "📜 Selecione um Arquivo do Histórico",
            font=ctk.CTkFont(size=16, weight="bold")
        ).pack(pady=10)
        
        # Scrollable frame for history
        self.history_scroll = ctk.CTkScrollableFrame(history_frame)
        self.history_scroll.pack(fill="both", expand=True, padx=10, pady=(0, 10))
        
        # Load history
        self.load_history()
        
        # Cancel button
        cancel_btn = ctk.CTkButton(
            self,
            text="Cancelar",
            command=self.destroy,
            height=35,
            fg_color="#95A5A6"
        )
        cancel_btn.pack(pady=(0, 20))
    
    def load_history(self):
        """Load and display file history"""
        recent_files = self.file_history.get_recent_files(count=10)
        
        if not recent_files:
            no_history_frame = ctk.CTkFrame(self.history_scroll, fg_color="transparent")
            no_history_frame.pack(pady=30, fill="x")
            
            no_history_label = ctk.CTkLabel(
                no_history_frame,
                text="📭 Nenhum arquivo no histórico",
                font=ctk.CTkFont(size=14),
                text_color="gray"
            )
            no_history_label.pack()
            
            hint_label = ctk.CTkLabel(
                no_history_frame,
                text="Importe um arquivo primeiro para começar a usar as ferramentas",
                font=ctk.CTkFont(size=11),
                text_color="gray60"
            )
            hint_label.pack(pady=(5, 0))
            return
        
        for file_data in recent_files:
            self.create_history_card(file_data)
    
    def create_history_card(self, file_data: dict):
        """Create a card for each historical file"""
        card = ctk.CTkFrame(self.history_scroll, fg_color="#2b2b2b")
        card.pack(fill="x", pady=5, padx=5)
        
        # File info
        file_path = file_data['file_path']
        file_name = Path(file_path).name
        
        # Check if file exists
        file_exists = Path(file_path).exists()
        
        # Left side - file info
        info_frame = ctk.CTkFrame(card, fg_color="transparent")
        info_frame.pack(side="left", fill="both", expand=True, padx=10, pady=10)
        
        # File name
        name_label = ctk.CTkLabel(
            info_frame,
            text=file_name,
            font=ctk.CTkFont(size=13, weight="bold"),
            anchor="w"
        )
        name_label.pack(anchor="w")
        
        # File details
        if file_exists:
            details_text = f"{file_data.get('rows', '?')} linhas × {file_data.get('cols', '?')} colunas"
            if 'last_accessed' in file_data:
                last_accessed = self.file_history.format_date(file_data['last_accessed'])
                details_text += f" • {last_accessed}"
            
            details_label = ctk.CTkLabel(
                info_frame,
                text=details_text,
                font=ctk.CTkFont(size=10),
                text_color="gray60",
                anchor="w"
            )
            details_label.pack(anchor="w")
        else:
            missing_label = ctk.CTkLabel(
                info_frame,
                text="⚠️ Arquivo não encontrado",
                font=ctk.CTkFont(size=10),
                text_color="#E74C3C",
                anchor="w"
            )
            missing_label.pack(anchor="w")
        
        # Right side - action button
        if file_exists:
            select_btn = ctk.CTkButton(
                card,
                text="Selecionar →",
                command=lambda fp=file_path: self.use_history_file(fp),
                width=120,
                height=35,
                fg_color="#27AE60"
            )
            select_btn.pack(side="right", padx=10, pady=10)
        else:
            # Disabled button for missing files
            disabled_btn = ctk.CTkButton(
                card,
                text="Não disponível",
                width=120,
                height=35,
                fg_color="#7F8C8D",
                state="disabled"
            )
            disabled_btn.pack(side="right", padx=10, pady=10)
    
    def use_current_data(self):
        """Use the currently loaded data"""
        self.selected_data = self.current_data
        self.selected_file_path = self.current_file_path
        self.destroy()
        self.on_select(self.selected_data, self.selected_file_path)
    
    def use_history_file(self, file_path: str):
        """Load and use a file from history"""
        try:
            # Determine file type
            file_ext = Path(file_path).suffix.lower()
            
            if file_ext in ['.xlsx', '.xls']:
                df = pd.read_excel(file_path)
            elif file_ext == '.csv':
                df = pd.read_csv(file_path)
            else:
                messagebox.showerror("Erro", f"Tipo de arquivo não suportado: {file_ext}")
                return
            
            # Update access count
            self.file_history.update_access(file_path)
            
            self.selected_data = df
            self.selected_file_path = file_path
            self.destroy()
            self.on_select(self.selected_data, self.selected_file_path)
            
        except Exception as e:
            messagebox.showerror("Erro ao Carregar", f"Não foi possível carregar o arquivo:\n{str(e)}")
