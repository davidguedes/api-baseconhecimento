import os
from venv import logger
from flask import Blueprint, request, jsonify
from app.services.document_service import DocumentService
from werkzeug.utils import secure_filename
from threading import Thread

document_blueprint = Blueprint('document', __name__)
document_service = DocumentService()

# Configuração para uploads
ALLOWED_EXTENSIONS = {'pdf', 'png', 'jpg', 'jpeg'}

def allowed_file(filename):
    return '.' in filename and \
        filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# Endpoint para upload de PDF
@document_blueprint.route('/api/documents/upload-pdf', methods=['POST'])
def upload_pdf():
    if 'file' not in request.files:
        return jsonify({'error': 'Nenhum arquivo enviado'}), 400
    
    file = request.files['file']
    sector = request.form.get('sector')
    
    # Novo campo para receber o setor
    if not sector:
        return jsonify({'error': 'Setor não informado'}), 400

    if file.filename == '':
        return jsonify({'error': 'Nenhum arquivo selecionado'}), 400
    
    filename = secure_filename(file.filename)
    file_content = file.read()

    def async_process():
        try:
            typefile = filename.rsplit('.', 1)[1].lower()
            if(typefile == 'pdf'):
                result = document_service.process_pdf(file_content, filename, sector)

            elif(typefile == 'xls' or typefile == 'xlsx'):
                result = document_service.process_excel(file_content, filename, sector)

            elif(typefile == 'doc' or typefile == 'docx'):
                result = document_service.process_word(file_content, filename, sector)

            logger.info(f"Processamento finalizado: {result}")
        except Exception as e:
            logger.error(f"Erro no processamento assíncrono: {str(e)}")

    # Inicia o processamento em background
    Thread(target=async_process).start()

    # Responde imediatamente ao usuário
    return jsonify({
        'status': 'accepted',
        'message': f'O arquivo {filename} foi recebido e será processado para o setor {sector}.'
    }), 202

@document_blueprint.route('/api/documents/<sector>', methods=['GET'])
def list_documents(sector):
    if not sector:
        return jsonify({'error': 'Setor não informado'}), 400
    
    return document_service.list_documents(sector=sector)

@document_blueprint.route('/api/documents/<sector>/<document_id>', methods=['DELETE'])
def delete_document(sector, document_id):
        if not sector:
            return jsonify({'error': 'Setor não informado'}), 400
            
        if not document_id:
            return jsonify({'error': 'ID do documento não informado'}), 400
            
        return document_service.delete_document_by_id(sector, document_id)
    
@document_blueprint.route('/api/documents/<sector>/source/<source>', methods=['DELETE'])
def delete_document_by_source(sector, source):
    if not sector:
        return jsonify({'error': 'Setor não informado'}), 400
        
    if not source:
        return jsonify({'error': 'Nome do arquivo não informado'}), 400
        
    return document_service.delete_document_by_source(sector, source)

@document_blueprint.route('/api/documents/images', methods=['GET'])
def list_sector_images():
    """Lista todas as imagens disponíveis para um setor"""
    sector = request.args.get('sector')
    
    if not sector:
        return jsonify({'error': 'Setor não informado'}), 400
    
    try:
        # Diretório de imagens do setor
        image_dir = os.path.join("images", sector)
        
        if not os.path.exists(image_dir):
            return jsonify({
                'status': 'success',
                'images': [],
                'message': 'Nenhuma imagem encontrada para este setor'
            }), 200
        
        # Listar arquivos de imagem
        image_files = []
        for root, _, files in os.walk(image_dir):
            for file in files:
                if file.lower().endswith(('.png', '.jpg', '.jpeg')):
                    full_path = os.path.join(root, file)
                    rel_path = os.path.relpath(full_path, "images")
                    image_files.append({
                        'name': file,
                        'path': rel_path,
                        'url': f'/api/images/{rel_path}'
                    })
        
        return jsonify({
            'status': 'success',
            'images': image_files,
            'count': len(image_files)
        }), 200
    
    except Exception as e:
        return jsonify({
            'status': 'error',
            'message': f'Erro ao listar imagens: {str(e)}'
        }), 500