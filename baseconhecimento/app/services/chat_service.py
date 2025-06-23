from app.models.message import Message
from app.services.ai_service import AIService

class ChatService:
    def __init__(self):
        self.ai_service = AIService()
    
    def add_message(self, content: str, role: str) -> Message:
        message = Message(content=content, role=role)
        self.conversation_history.append(message)
        return message

    def process_user_message(self, sector: str, content: str, sector_name: str,):
        """Processa a mensagem do usuário e retorna a resposta do assistente"""
        # Aqui você pode adicionar lógica para registrar a conversa em um banco de dados
        # Por exemplo, salvar a mensagem do usuário antes de processar
        
        try:
            # Gerar resposta usando o serviço de IA
            ai_response = self.ai_service.generate_response(sector, content, sector_name)
            
            # Aqui você pode adicionar lógica para salvar a resposta no histórico
            
            return ai_response
        except Exception as e:
            print(f"Erro ao processar mensagem: {str(e)}")
            return f"Desculpe, ocorreu um erro ao processar sua mensagem. Detalhes: {str(e)}"