"""
Tela de Login/Autenticação do Pro Sigma
"""
import customtkinter as ctk
from tkinter import messagebox
from src.core.license_manager import LicenseManager


class LoginWindow(ctk.CTkToplevel):
    """Janela de login e ativação de licença"""
    
    def __init__(self, parent, on_success_callback):
        """
        Inicializa a janela de login
        
        Args:
            parent: Janela pai
            on_success_callback: Função a ser chamada após login bem-sucedido
        """
        super().__init__(parent)
        
        self.on_success_callback = on_success_callback
        self.license_manager = LicenseManager()
        
        # Configurações da janela
        self.title("Pro Sigma - Ativação")
        self.geometry("500x400")
        self.resizable(False, False)
        
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
        width = self.winfo_width()
        height = self.winfo_height()
        x = (self.winfo_screenwidth() // 2) - (width // 2)
        y = (self.winfo_screenheight() // 2) - (height // 2)
        self.geometry(f'{width}x{height}+{x}+{y}')
    
    def create_widgets(self):
        """Cria os widgets da interface"""
        
        # Container principal
        main_frame = ctk.CTkFrame(self, fg_color="transparent")
        main_frame.pack(expand=True, fill="both", padx=40, pady=40)
        
        # Logo/Título
        title_label = ctk.CTkLabel(
            main_frame,
            text="Pro Sigma",
            font=ctk.CTkFont(size=32, weight="bold")
        )
        title_label.pack(pady=(0, 10))
        
        subtitle_label = ctk.CTkLabel(
            main_frame,
            text="Software de Análise Estatística Six Sigma",
            font=ctk.CTkFont(size=12)
        )
        subtitle_label.pack(pady=(0, 40))
        
        # Descrição
        info_label = ctk.CTkLabel(
            main_frame,
            text="Para começar a usar o Pro Sigma,\ninsira sua chave de licença:",
            font=ctk.CTkFont(size=13)
        )
        info_label.pack(pady=(0, 20))
        
        # Campo de entrada da licença
        self.license_entry = ctk.CTkEntry(
            main_frame,
            placeholder_text="Cole sua chave de licença aqui",
            width=400,
            height=40,
            font=ctk.CTkFont(size=12)
        )
        self.license_entry.pack(pady=(0, 10))
        
        # Mensagem de status
        self.status_label = ctk.CTkLabel(
            main_frame,
            text="",
            font=ctk.CTkFont(size=11),
            text_color="gray"
        )
        self.status_label.pack(pady=(0, 20))
        
        # Botão de ativação
        self.activate_button = ctk.CTkButton(
            main_frame,
            text="Ativar Licença",
            command=self.activate_license,
            width=200,
            height=40,
            font=ctk.CTkFont(size=14, weight="bold")
        )
        self.activate_button.pack(pady=(0, 10))
        
        # Link para obter licença (futuro)
        help_label = ctk.CTkLabel(
            main_frame,
            text="Não tem uma licença? Entre em contato conosco",
            font=ctk.CTkFont(size=10),
            text_color="gray"
        )
        help_label.pack(pady=(10, 0))
        
        # Bind Enter key
        self.license_entry.bind('<Return>', lambda e: self.activate_license())
        
        # Foca no campo de entrada
        self.license_entry.focus()
    
    def activate_license(self):
        """Valida e ativa a licença inserida"""
        license_key = self.license_entry.get().strip()
        
        if not license_key:
            self.show_status("Por favor, insira uma chave de licença", "error")
            return
        
        # Desabilita botão durante validação
        self.activate_button.configure(state="disabled", text="Validando...")
        self.update()
        
        try:
            # Valida e salva a licença
            license_data = self.license_manager.save_license(license_key)
            
            # Mostra mensagem de sucesso
            messagebox.showinfo(
                "Sucesso!",
                f"Licença ativada com sucesso!\n\n"
                f"Plano: {license_data['plan_name']}\n"
                f"Válida até: {license_data['expiratedDate']}"
            )
            
            # Fecha a janela e chama callback
            self.destroy()
            if self.on_success_callback:
                self.on_success_callback(license_data)
                
        except ValueError as e:
            # Mostra erro
            error_msg = str(e).replace("Erro ao validar licença: ", "")
            self.show_status(error_msg, "error")
            self.activate_button.configure(state="normal", text="Ativar Licença")
            
        except Exception as e:
            self.show_status(f"Erro inesperado: {str(e)}", "error")
            self.activate_button.configure(state="normal", text="Ativar Licença")
    
    def show_status(self, message: str, status_type: str = "info"):
        """
        Mostra mensagem de status
        
        Args:
            message: Mensagem a ser exibida
            status_type: Tipo da mensagem (info, error, success)
        """
        colors = {
            "info": "gray",
            "error": "#FF5555",
            "success": "#55FF55"
        }
        
        self.status_label.configure(
            text=message,
            text_color=colors.get(status_type, "gray")
        )
