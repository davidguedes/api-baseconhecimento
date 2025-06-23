import datetime
import os
from flask import Blueprint, request, jsonify, send_file
from app.services.chat_service import ChatService

chat_blueprint = Blueprint('chat', __name__)
chat_service = ChatService()

@chat_blueprint.route('/api/chat/message', methods=['POST'])
def send_message():
    data = request.json
    
    if not data or 'message' not in data or 'sector' not in data:
        return jsonify({'error': 'Mensagem ou setor não informados'}), 400
    
    # Processar a mensagem
    print(f"Processando mensagem para o setor: {data['sector']}")
    response = chat_service.process_user_message(data['sector'], data['message'], data['sector_name'])
    
    # Verificar formato da resposta
    if isinstance(response, dict) and "text" in response:
        
        return {
            'content': response["text"].content if hasattr(response["text"], "content") else str(response["text"]),
            'role': 'maquina',
            'timestamp': datetime.datetime.now().isoformat()
        }
    else:
        # Para compatibilidade com o formato anterior
        return jsonify({
            'content': response,
            'images': [],
            'role': 'maquina',
            'timestamp': datetime.datetime.now().isoformat()
        })

@chat_blueprint.route('/api/chat/history', methods=['GET'])
def get_history():
    history = [
        {
            'content': msg.content,
            'role': msg.role,
            'timestamp': msg.timestamp.isoformat()
        }
        for msg in chat_service.conversation_history
    ]

    return jsonify(history)

# Rota para listar setores disponíveis
@chat_blueprint.route('/api/chat/sectors', methods=['GET'])
def list_sectors():
    try:
        # Listar diretórios na pasta de persistência do ChromaDB
        persist_dir = chat_service.ai_service.persist_directory
        sectors = [d for d in os.listdir(persist_dir) if os.path.isdir(os.path.join(persist_dir, d))]

        return jsonify({
            'status': 'success',
            'sectors': sectors
        }), 200
    except Exception as e:
        return jsonify({
            'status': 'error',
            'message': f'Erro ao listar setores: {str(e)}'
        }), 500