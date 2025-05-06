
"""
Sistema de Base de Conhecimento Corporativa

Este módulo implementa um sistema para processamento e armazenamento
de documentos por setor, com suporte para análise multimodal (texto, imagens, tabelas).
"""

import os
import tempfile
import logging

from flask import jsonify
from pathlib import Path
from typing import List, Tuple
from config.settings import Config
from typing import List, Dict, Optional
from datetime import datetime

from langchain_ollama import OllamaEmbeddings
from langchain_ollama import OllamaLLM
from langchain_chroma import Chroma
from langchain.schema.document import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter

from docling.datamodel.base_models import InputFormat
from docling.datamodel.pipeline_options import PdfPipelineOptions
from docling.document_converter import DocumentConverter, PdfFormatOption

# Configuração de logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class DocumentRepository:
    """
    Classe responsável por gerenciar o acesso ao banco de dados vetorial.
    Implementa o padrão de repositório para abstrair o acesso ao ChromaDB.
    """
    def __init__(self, persist_directory=Config.CHROMA_DIRECTORY):
        self.persist_directory = persist_directory
        self.embeddings = OllamaEmbeddings(model=Config.MODEL_NAME)
    
    def get_vectorstore(self, sector: str) -> Chroma:
        """Retorna uma instância do ChromaDB para o setor especificado"""
        sector_persist_dir = os.path.join(self.persist_directory, sector)
        os.makedirs(sector_persist_dir, exist_ok=True)
        
        return Chroma(
            collection_name=f'sector_{sector}',
            persist_directory=sector_persist_dir,
            embedding_function=self.embeddings
        )
    
    def add_documents(self, sector: str, documents: List[Document]) -> List[str]:
        """Adiciona documentos ao ChromaDB e retorna os IDs"""
        vectorstore = self.get_vectorstore(sector)
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=Config.CHUNK_SIZE,
            chunk_overlap=Config.CHUNK_OVERLAP
        )
        
        split_docs = text_splitter.split_documents(documents)
        
        # Processa em lotes para evitar sobrecarga de memória
        ids = []
        batch_size = Config.EMBEDDING_BATCH_SIZE
        for i in range(0, len(split_docs), batch_size):
            batch = split_docs[i:i + batch_size]
            batch_ids = vectorstore.add_documents(batch)
            ids.extend(batch_ids)
        
        return ids
    
    def get_documents_by_source(self, sector: str, source: str) -> Tuple[List[Dict], List[str]]:
        """Recupera todos os documentos com a origem especificada"""
        vectorstore = self.get_vectorstore(sector)
        collection = vectorstore._collection
        
        # Obter todos os metadados e IDs
        result = collection.get()
        metadatas = result["metadatas"]
        ids = result["ids"]
        
        matching_docs = []
        matching_ids = []
        
        for i, metadata in enumerate(metadatas):
            if metadata and metadata.get("source") == source:
                doc_info = {
                    "id": ids[i],
                    "source": metadata.get("source", "Unknown"),
                    "sector": metadata.get("sector", sector),
                    "type": metadata.get("type", "Unknown"),
                    "upload_date": metadata.get("upload_date", "Unknown"),
                    "file_type": metadata.get("format", "Unknown")
                }
                matching_docs.append(doc_info)
                matching_ids.append(ids[i])
                
        return matching_docs, matching_ids
    
    def delete_documents_by_ids(self, sector: str, doc_ids: List[str]) -> None:
        """Remove documentos pelos IDs"""
        vectorstore = self.get_vectorstore(sector)
        vectorstore.delete(doc_ids)
    
    def delete_documents_by_source(self, sector: str, source: str) -> int:
        """Remove todos os documentos com a origem especificada e retorna o número de documentos removidos"""
        _, doc_ids = self.get_documents_by_source(sector, source)
        
        if doc_ids:
            self.delete_documents_by_ids(sector, doc_ids)
            
        return len(doc_ids)
    
    def list_unique_sources(self, sector: str) -> List[Dict]:
        """Lista fontes únicas (arquivos) no setor, agrupando por nome de arquivo"""
        vectorstore = self.get_vectorstore(sector)
        collection = vectorstore._collection
        
        # Obter todos os metadados
        result = collection.get()
        metadatas = result["metadatas"]
        
        # Usar um dicionário para armazenar informações únicas por fonte
        unique_sources = {}
        
        for metadata in metadatas:
            if not metadata:
                continue
                
            source = metadata.get("source")
            if not source:
                continue
                
            # Se a fonte já existe, apenas atualiza a contagem
            if source in unique_sources:
                unique_sources[source]["chunk_count"] += 1
                
                # Registrar diferentes tipos de conteúdo
                content_type = metadata.get("type", "unknown")
                if content_type not in unique_sources[source]["content_types"]:
                    unique_sources[source]["content_types"].append(content_type)
            else:
                # Criar nova entrada para esta fonte
                unique_sources[source] = {
                    "source": source,
                    "sector": metadata.get("sector", sector),
                    "format": metadata.get("format", "Unknown"),
                    "upload_date": metadata.get("upload_date", "Unknown"),
                    "chunk_count": 1,
                    "content_types": [metadata.get("type", "unknown")]
                }
        
        # Converter o dicionário para uma lista
        return list(unique_sources.values())

class ImageProcessor:
    """Classe responsável pelo processamento de imagens"""
    def __init__(self, vision_model_name=Config.VISION_MODEL_NAME):
        self.vision_model = OllamaLLM(
            model=vision_model_name,
            num_gpu=1,  # Limitar uso de GPU
            num_thread=4  # Limitar threads de CPU
        )
        
    def process_image(self, image_path: str, source: str, sector: str, page_no: int) -> Optional[Document]:
        """Processa uma imagem e retorna um documento com a descrição da imagem"""
        try:
            image_llm = self.vision_model.bind(images=[image_path])
            
            prompt = """
            Descreva o que você vê na imagem em português. 
            Faça de forma simples, fácil de entender e com uma linguagem muito clara. 
            Seja o mais descritivo possível, sem perder ou pular nenhum detalhe.
            """
            
            response = image_llm.invoke(prompt)
            
            logger.info(f"Resultado da descrição: {response}")

            return Document(
                page_content=response,
                metadata={
                    "source": source,
                    "sector": sector,
                    "type": "image-content",
                    "page": page_no,
                    "image_path": image_path,
                    "format": "image/png",
                    "upload_date": datetime.now().isoformat()
                }
            )
        except Exception as e:
            logger.error(f"Erro ao processar imagem: {str(e)}")
            return None

class DocumentService:
    def __init__(
            self, 
            persist_directory=Config.CHROMA_DIRECTORY,
            vision_model_name=Config.VISION_MODEL_NAME
        ):
        
        self.embeddings = OllamaEmbeddings(model=Config.MODEL_NAME)

        self.repository = DocumentRepository(persist_directory)

        self.image_processor = ImageProcessor(vision_model_name)

    def _save_temp_file(self, file_content, file_name):
        temp_dir = tempfile.gettempdir()
        temp_path = os.path.join(temp_dir, file_name)
        
        with open(temp_path, 'wb') as f:
            f.write(file_content)

        return temp_path
    
    def process_pdf(self, pdf_file, file_name: str, sector: str):
        """
        Processa um arquivo PDF e adiciona seu conteúdo ao banco de dados vetorial.
        
        Inclui um ID de grupo para facilitar a exclusão e listagem por arquivo.
        """
        # Salva o arquivo temporariamente
        temp_path = self._save_temp_file(pdf_file, file_name)

        try:
            # Usar docling para processamento do PDF
            input_doc_path = Path(temp_path)
            output_dir = Path(f"./images/{sector}")
            output_dir.mkdir(parents=True, exist_ok=True)
            
            # Configuração do docling
            pipeline_options = PdfPipelineOptions()
            pipeline_options.images_scale=2.0
            pipeline_options.generate_page_images=True
            pipeline_options.generate_picture_images=True
            
            doc_converter = DocumentConverter(
                format_options={
                    InputFormat.PDF: PdfFormatOption(pipeline_options=pipeline_options)
                }
            )
            
            # Converter o documento
            conv_res = doc_converter.convert(input_doc_path)
            data = conv_res.document.export_to_dict()
            text = conv_res.document.export_to_text()
            
            # Gerar um timestamp de upload para agrupar documentos
            upload_date = datetime.now().isoformat()

            # Preparar documentos para vetorização
            documents = []
            
            # Adicionar conteúdo de texto
            text_doc = Document(
                page_content=text,
                metadata={
                    "source": file_name,
                    "sector": sector,
                    "type": "text-content",
                    "format": "pdf",
                    "upload_date": upload_date
                }
            )
            documents.append(text_doc)
            
            # Processar imagens com o modelo de visão
            pics = data.get("pictures", [])
            page_nums = []

            for pic in pics:
                if "prov" in pic and pic["prov"] and "page_no" in pic["prov"][0]:
                    page_nums.append(pic["prov"][0]["page_no"])
            
            page_nums = sorted(list(set(page_nums)))
            doc_filename = conv_res.input.file.stem
            image_docs_count = 0
            #print(f"Processando {len(page_nums)} páginas com imagens no documento {doc_filename}")
            
            for page_no, page in conv_res.document.pages.items():
                page_no = page.page_no
                if page_no in page_nums:
                    logger.info(f"Processando página {page_no}")
                    
                    # Salvar imagem da página
                    image_path = f"{output_dir}/{doc_filename}-page-{page_no}.png"
                    with open(image_path, "wb") as f:
                        page.image.pil_image.save(f, format="PNG")
                    
                    # Processar imagem
                    image_doc = self.image_processor.process_image(
                        image_path, file_name, sector, page_no
                    )
                    
                    if image_doc:
                        documents.append(image_doc)
                        image_docs_count += 1
                        logger.info(f"Imagem da página {page_no} processada com sucesso")

            # Extrair tabelas e elementos estruturados
            tables = data.get("tables", [])
            table_docs_count = 0

            for i, table in enumerate(tables):
                table_content = str(table.get("content", ""))
                table_doc = Document(
                    page_content=table_content,
                    metadata={
                        "source": file_name,
                        "sector": sector,
                        "type": "table-content",
                        "table_index": i,
                        "format": "text/table",
                        "upload_date": upload_date
                    }
                )
                documents.append(table_doc)
                table_docs_count += 1

            # Adicionar documentos ao repositório
            self.repository.add_documents(sector, documents)
            
            # Preparar estatísticas para resposta
            stats = {
                "text_docs": 1,  # Temos sempre um documento de texto principal
                "image_docs": image_docs_count,
                "table_docs": table_docs_count,
            }
            
            return {
                "status": "success",
                "message": f"PDF {file_name} processado com sucesso para o setor {sector}",
                "document_count": len(documents),
                "stats": stats,
                "sector": sector
            }
            
        except Exception as e:
            logger.error(f"Erro ao processar PDF: {str(e)}")
            return {
                "status": "error",
                "message": f"Erro ao processar PDF: {str(e)}"
            }
        finally:
            # Limpa o arquivo temporário
            if os.path.exists(temp_path):
                os.remove(temp_path)
    
    def list_documents(self, sector: str):
        """Lista documentos únicos do setor (agrupados por arquivo)"""
        try:
            unique_docs = self.repository.list_unique_sources(sector)
            
            return jsonify(unique_docs), 200

            """return jsonify({
                "status": "success", 
                "count": len(unique_docs),
                "documents": unique_docs
            }), 200"""
            
        except Exception as e:
            logger.error(f"Erro ao listar documentos: {str(e)}")
            return jsonify({
                "status": "error",
                "message": f"Erro ao listar documentos: {str(e)}"
            }), 500

    def delete_document_by_id(self, sector: str, document_id: str):
        """Remove um único documento pelo ID"""
        try:
            self.repository.delete_documents_by_ids(sector, [document_id])
            
            return jsonify({
                "status": "success",
                "message": f"Documento {document_id} removido com sucesso"
            }), 200
            
        except Exception as e:
            logger.error(f"Erro ao remover documento: {str(e)}")
            return jsonify({
                "status": "error",
                "message": f"Erro ao remover documento: {str(e)}"
            }), 500
    
    def delete_document_by_source(self, sector: str, source: str):
        """Remove todos os documentos relacionados a um arquivo específico"""
        try:
            count = self.repository.delete_documents_by_source(sector, source)
            
            return jsonify({
                "status": "success",
                "message": f"Documento {source} e todos os seus {count} fragmentos foram removidos com sucesso"
            }), 200
            
        except Exception as e:
            logger.error(f"Erro ao remover documento: {str(e)}")
            return jsonify({
                "status": "error",
                "message": f"Erro ao remover documento: {str(e)}"
            }), 500