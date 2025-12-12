"""
Janela para Renovação/Inserção de Nova Licença
"""
import customtkinter as ctk
from tkinter import messagebox
from src.core.license_manager import LicenseManager


class RenewLicenseWindow(ctk.CTkToplevel):
    """Janela para inserir nova licença (renovação/upgrade)"""
    
    def __init__(self, parent, on_success_callback):
        """
        Inicializa a janela de renovação
        
        Args:
            parent: Janela pai
            on_success_callback: Função a ser chamada após renovação bem-sucedida
        """
        super().__init__(parent)
        
        self.on_success_callback = on_success_callback
        self.license_manager = LicenseManager()
        
        # Configurações da janela
        self.title("Pro Sigma - Renovar/Alterar Licença")
        self.geometry("600x550")
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
        main_frame.pack(expand=True, fill="both", padx=40, pady=35)
        
        # Ícone/Título
        title_label = ctk.CTkLabel(
            main_frame,
            text="🔄 Renovar/Alterar Licença",
            font=ctk.CTkFont(size=28, weight="bold")
        )
        title_label.pack(pady=(0, 15))
        
        # Informação da licença atual
        current_license = self.license_manager.load_license()
        if current_license:
            current_info = ctk.CTkFrame(main_frame)
            current_info.pack(fill="x", pady=(0, 20))
            
            current_label = ctk.CTkLabel(
                current_info,
                text="Licença Atual",
                font=ctk.CTkFont(size=12, weight="bold")
            )
            current_label.pack(pady=(10, 5))
            
            plan_label = ctk.CTkLabel(
                current_info,
                text=f"Plano: {current_license['plan_name']}",
                font=ctk.CTkFont(size=11)
            )
            plan_label.pack()
            
            expiry_label = ctk.CTkLabel(
                current_info,
                text=f"Válido até: {current_license['expiratedDate']}",
                font=ctk.CTkFont(size=11),
                text_color="gray"
            )
            expiry_label.pack(pady=(0, 10))
        
        # Separador
        separator = ctk.CTkFrame(main_frame, height=2, fg_color="gray30")
        separator.pack(fill="x", pady=10)
        
        # Descrição
        info_label = ctk.CTkLabel(
            main_frame,
            text="Insira sua nova chave de licença abaixo:\n"
                 "Se você renovou ou fez upgrade, cole a nova chave recebida por email.",
            font=ctk.CTkFont(size=12),
            text_color="gray"
        )
        info_label.pack(pady=(10, 20))
        
        # Campo de entrada da nova licença
        license_label = ctk.CTkLabel(
            main_frame,
            text="Nova Chave de Licença:",
            font=ctk.CTkFont(size=12, weight="bold"),
            anchor="w"
        )
        license_label.pack(anchor="w", pady=(0, 5))
        
        self.license_entry = ctk.CTkTextbox(
            main_frame,
            height=90,
            font=ctk.CTkFont(size=11),
            wrap="word"
        )
        self.license_entry.pack(fill="x", pady=(0, 15))
        
        # Mensagem de status
        self.status_label = ctk.CTkLabel(
            main_frame,
            text="",
            font=ctk.CTkFont(size=11),
            text_color="gray"
        )
        self.status_label.pack(pady=(0, 15))
        
        # Frame para botões
        button_frame = ctk.CTkFrame(main_frame, fg_color="transparent")
        button_frame.pack(fill="x", pady=(15, 0))
        
        # Botão de cancelar
        self.cancel_button = ctk.CTkButton(
            button_frame,
            text="Cancelar",
            command=self.destroy,
            width=220,
            height=50,
            font=ctk.CTkFont(size=14),
            fg_color="gray30",
            hover_color="gray20",
            corner_radius=8
        )
        self.cancel_button.pack(side="left", padx=8, expand=True, fill="x")
        
        # Botão de ativar
        self.activate_button = ctk.CTkButton(
            button_frame,
            text="Ativar Nova Licença",
            command=self.activate_new_license,
            width=220,
            height=50,
            font=ctk.CTkFont(size=14, weight="bold"),
            fg_color="#2E86DE",
            hover_color="#1E5BA8",
            corner_radius=8
        )
        self.activate_button.pack(side="left", padx=8, expand=True, fill="x")
        
        # Dica
        tip_label = ctk.CTkLabel(
            main_frame,
            text="💡 Dica: A chave deve ser copiada exatamente como recebida,\nsem espaços extras no início ou fim.",
            font=ctk.CTkFont(size=10),
            text_color="gray50"
        )
        tip_label.pack(pady=(20, 0))
        
        # Foca no campo de entrada
        self.license_entry.focus()
    
    def activate_new_license(self):
        """Valida e ativa a nova licença"""
        license_key = self.license_entry.get("1.0", "end-1c").strip()
        
        if not license_key:
            self.show_status("Por favor, insira uma chave de licença", "error")
            return
        
        # Desabilita botão durante validação
        self.activate_button.configure(state="disabled", text="Validando...")
        self.cancel_button.configure(state="disabled")
        self.update()
        
        try:
            # Valida e salva a nova licença
            new_license_data = self.license_manager.save_license(license_key)
            
            # Compara com licença anterior
            old_license = self.license_manager.load_license()
            
            upgrade_msg = ""
            if old_license:
                old_plan = old_license.get('plan', '')
                new_plan = new_license_data.get('plan', '')
                
                if new_plan != old_plan:
                    plan_hierarchy = {'basic': 1, 'intermediate': 2, 'pro': 3}
                    if plan_hierarchy.get(new_plan, 0) > plan_hierarchy.get(old_plan, 0):
                        upgrade_msg = "\n\n🎉 Parabéns! Você fez upgrade de plano!"
                    else:
                        upgrade_msg = "\n\n✓ Plano alterado com sucesso!"
            
            # Mostra mensagem de sucesso
            messagebox.showinfo(
                "Sucesso!",
                f"Nova licença ativada com sucesso!{upgrade_msg}\n\n"
                f"Plano: {new_license_data['plan_name']}\n"
                f"Válida até: {new_license_data['expiratedDate']}"
            )
            
            # Fecha a janela e chama callback
            self.destroy()
            if self.on_success_callback:
                self.on_success_callback(new_license_data)
                
        except ValueError as e:
            # Mostra erro
            error_msg = str(e).replace("Erro ao validar licença: ", "")
            self.show_status(error_msg, "error")
            self.activate_button.configure(state="normal", text="Ativar Nova Licença")
            self.cancel_button.configure(state="normal")
            
        except Exception as e:
            self.show_status(f"Erro inesperado: {str(e)}", "error")
            self.activate_button.configure(state="normal", text="Ativar Nova Licença")
            self.cancel_button.configure(state="normal")
    
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
