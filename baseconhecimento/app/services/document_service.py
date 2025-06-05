
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
from docling.backend.pypdfium2_backend import PyPdfiumDocumentBackend

import cv2
import numpy as np
from PIL import Image, ImageEnhance, ImageFilter
import re

from spellchecker import SpellChecker
import language_tool_python
from docling.datamodel.pipeline_options import PdfPipelineOptions, PaginatedPipelineOptions
from docling.document_converter import DocumentConverter, PdfFormatOption, WordFormatOption, ExcelFormatOption
from docling.backend.pypdfium2_backend import PyPdfiumDocumentBackend

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

        print('Documentos: ', documents)
        
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
            Não retorne junto a descrição saudações ou cumprimentos.
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

        # Inicializar corretor ortográfico para português
        self.spell_checker = SpellChecker(language='pt')
        
        self.language_tool = None  # Inicializar apenas quando necessário

    def _save_temp_file(self, file_content, file_name):
        temp_dir = tempfile.gettempdir()
        temp_path = os.path.join(temp_dir, file_name)
        
        with open(temp_path, 'wb') as f:
            f.write(file_content)

        return temp_path
    
    def process_pdf(self, pdf_file, file_name: str, sector: str):
        """
        Processa um arquivo PDF com melhorias na extração de texto e qualidade do OCR.
        
        Inclui pré-processamento de imagens, múltiplas tentativas de OCR e 
        pós-processamento de texto para melhor qualidade.
        """
        
        # Salva o arquivo temporariamente
        temp_path = self._save_temp_file(pdf_file, file_name)

        try:
            # Usar docling para processamento do PDF com configurações otimizadas
            input_doc_path = Path(temp_path)
            output_dir = Path(f"./images/{sector}")
            output_dir.mkdir(parents=True, exist_ok=True)
            
            # Configuração otimizada do docling para melhor OCR
            pipeline_options = PdfPipelineOptions()
            pipeline_options.ocr_options.lang = ["pt"]  # Português brasileiro e português
            pipeline_options.images_scale = 3.0  # Aumentar escala para melhor OCR
            pipeline_options.generate_page_images = True
            pipeline_options.generate_picture_images = True
            
            doc_converter = DocumentConverter(
                format_options={
                    InputFormat.PDF: PdfFormatOption(
                        pipeline_options=pipeline_options,
                        backend=PyPdfiumDocumentBackend,
                    )
                }
            )
            
            # Converter o documento
            conv_res = doc_converter.convert(input_doc_path)
            data = conv_res.document.export_to_dict()
            raw_text = conv_res.document.export_to_text()

            # Pós-processamento do texto extraído
            processed_text = self._post_process_text(raw_text)

            # Normalizar e limpar o texto português
            #text = self._normalize_portuguese_text(text)
            #text = self._clean_extracted_text(text)
            
            # Gerar um timestamp de upload para agrupar documentos
            upload_date = datetime.now().isoformat()

            # Preparar documentos para vetorização
            documents = []
            
            # Adicionar conteúdo de texto processado
            if processed_text and processed_text.strip():
                text_doc = Document(
                    page_content=processed_text,
                    metadata={
                        "source": file_name,
                        "sector": sector,
                        "type": "text-content",
                        "format": "pdf",
                        "upload_date": upload_date,
                        "processing_quality": "enhanced"
                    }
                )
                documents.append(text_doc)
                print(f"Texto processado extraído: {len(processed_text)} caracteres")

            # Processar imagens com melhorias
            pics = data.get("pictures", [])
            page_nums = []

            for pic in pics:
                if "prov" in pic and pic["prov"] and "page_no" in pic["prov"][0]:
                    page_nums.append(pic["prov"][0]["page_no"])
            
            page_nums = sorted(list(set(page_nums)))
            doc_filename = conv_res.input.file.stem
            image_docs_count = 0
            
            for page_no, page in conv_res.document.pages.items():
                page_no = page.page_no
                if page_no in page_nums:
                    logger.info(f"Processando página {page_no} com melhorias de imagem")
                    
                    # Melhorar qualidade da imagem antes de salvar
                    enhanced_image = self._enhance_image_for_ocr(page.image.pil_image)
                    
                    # Salvar imagem melhorada
                    image_path = f"{output_dir}/{doc_filename}-page-{page_no}.png"
                    enhanced_image.save(image_path, format="PNG", dpi=(300, 300))
                    
                    # Processar imagem com OCR adicional se necessário
                    image_doc = self._process_image_with_fallback_ocr(
                        image_path, enhanced_image, file_name, sector, page_no
                    )
                    
                    if image_doc:
                        documents.append(image_doc)
                        image_docs_count += 1
                        logger.info(f"Imagem da página {page_no} processada com sucesso")

            # Extrair e processar tabelas
            tables = data.get("tables", [])
            table_docs_count = 0

            for i, table in enumerate(tables):
                table_content = str(table.get("content", ""))
                if not table_content.strip():
                    continue

                # Processar conteúdo da tabela
                processed_table_content = self._post_process_text(table_content)

                table_doc = Document(
                    page_content=processed_table_content,
                    metadata={
                        "source": file_name,
                        "sector": sector,
                        "type": "table-content",
                        "table_index": i,
                        "format": "text/table",
                        "upload_date": upload_date,
                        "processing_quality": "enhanced"
                    }
                )
                documents.append(table_doc)
                table_docs_count += 1

            # Adicionar documentos ao repositório
            self.repository.add_documents(sector, documents)
            
            print('texto processado: ', processed_text)

            # Preparar estatísticas para resposta
            stats = {
                "text_docs": 1 if processed_text and processed_text.strip() else 0,
                "image_docs": image_docs_count,
                "table_docs": table_docs_count,
            }
            
            print('chegou no estagio final.')
            
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

    def _enhance_image_for_ocr(self, pil_image):
        """
        Melhora a qualidade da imagem para melhor reconhecimento de texto.
        """
        # Converter PIL para OpenCV
        img_array = np.array(pil_image)
        if len(img_array.shape) == 3:
            img = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
        else:
            img = img_array
        
        # Redimensionar se muito pequena
        height, width = img.shape[:2]
        if height < 600 or width < 600:
            scale_factor = max(800/height, 800/width)
            new_width = int(width * scale_factor)
            new_height = int(height * scale_factor)
            img = cv2.resize(img, (new_width, new_height), interpolation=cv2.INTER_CUBIC)
        
        # Converter para escala de cinza
        if len(img.shape) == 3:
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        else:
            gray = img
        
        # Aplicar filtro de denoising
        denoised = cv2.fastNlMeansDenoising(gray)
        
        # Melhorar contraste usando CLAHE
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        enhanced = clahe.apply(denoised)
        
        # Aplicar binarização adaptativa
        binary = cv2.adaptiveThreshold(
            enhanced, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2
        )
        
        # Converter de volta para PIL
        enhanced_pil = Image.fromarray(binary)
        
        # Aplicar sharpening
        enhanced_pil = enhanced_pil.filter(ImageFilter.SHARPEN)
        
        return enhanced_pil

    def _process_image_with_fallback_ocr(self, image_path, enhanced_image, file_name, sector, page_no):
        """
        Processa imagem com múltiplas tentativas de OCR para melhor qualidade.
        """
        try:
            # Primeiro tentar com o processador de imagem padrão
            image_doc = self.image_processor.process_image(
                image_path, file_name, sector, page_no
            )
            
            # Se o resultado for muito curto ou com muitos erros, tentar OCR direto
            if not image_doc or len(image_doc.page_content) < 50:
                import pytesseract
                
                # Configurações otimizadas do Tesseract para português
                custom_config = r'--oem 3 --psm 6 -l por+pt'
                
                # OCR na imagem melhorada
                ocr_text = pytesseract.image_to_string(enhanced_image, config=custom_config)
                
                if ocr_text and len(ocr_text.strip()) > 20:
                    processed_ocr_text = self._post_process_text(ocr_text)
                    
                    image_doc = Document(
                        page_content=processed_ocr_text,
                        metadata={
                            "source": file_name,
                            "sector": sector,
                            "type": "image-content",
                            "page_number": page_no,
                            "format": "ocr-enhanced",
                            "upload_date": datetime.now().isoformat(),
                            "processing_method": "fallback_ocr"
                        }
                    )
            
            return image_doc
        except Exception as e:
            logger.warning(f"Erro no processamento de imagem da página {page_no}: {str(e)}")
            return None

    def _post_process_text(self, text, spelling_correction='languagetool'):
        """
        Pós-processa o texto extraído para corrigir erros comuns de OCR.
        """
        if not text:
            return ""
        
        # Remover caracteres estranhos comuns em OCR
        text = re.sub(r'[^\w\s\-.,;:!?()[\]{}\"\'@#$%&*+=<>/\\|`~]', '', text)
        
        # Corrigir espaçamentos múltiplos
        text = re.sub(r'\s+', ' ', text)
        
        # Corrigir quebras de linha desnecessárias
        text = re.sub(r'\n\s*\n', '\n\n', text)
        text = re.sub(r'(?<=[a-z])\n(?=[a-z])', ' ', text)
        
        # Dicionário de correções comuns em português (OCR)
        common_corrections = {
            r'\bl\b': 'I',  # l minúsculo sozinho geralmente é I
            r'\bO\b': '0',  # O maiúsculo sozinho em contexto numérico
            r'\b0\b': 'O',  # 0 em contexto de palavras
            r'rn': 'm',     # rn é frequentemente m mal reconhecido
            r'vv': 'w',     # vv é frequentemente w
            r'\bao\b': 'do', # ao pode ser do
            r'\bcla\b': 'da', # cla pode ser da
            r'§': 'S',      # símbolo de seção confundido com S
        }
        
        for pattern, replacement in common_corrections.items():
            text = re.sub(pattern, replacement, text, flags=re.IGNORECASE)
        
        # 5. Aplicar correção ortográfica se solicitada
        if spelling_correction != 'none':
            text = self._correct_with_context(text, method=spelling_correction)

        # Remover linhas muito curtas que provavelmente são ruído
        lines = text.split('\n')
        filtered_lines = [line.strip() for line in lines if len(line.strip()) > 2]
        
        return '\n'.join(filtered_lines).strip()

    def _initialize_language_tool(self):
        """Inicializa o LanguageTool apenas quando necessário"""
        print('aqui inicializando')
        if self.language_tool is None:
            self.language_tool = language_tool_python.LanguageTool('pt-BR')

    def _correct_with_context(self, text, method='languagetool'):
        print('O motodo é: ', method)

        """Correção com diferentes níveis de contexto"""
        if method == 'simple':
            return self._correct_spelling_simple(text)
        
        elif method == 'languagetool':
            try:
                #self._initialize_language_tool()
                language_tool = language_tool_python.LanguageTool('pt-BR')
                #matches = self.language_tool.check(text)
                matches = language_tool.check(text)
                print('Matches: ', matches)
                correcao = language_tool_python.utils.correct(text, matches)
                print('A correcao: ', correcao)
                return correcao
            except Exception as e:
                print('O erro: ', e)
                return self._correct_spelling_simple(text)
        
        return text
    
    def _correct_spelling_simple(self, text):
        """Correção ortográfica simples palavra por palavra"""
        words = re.findall(r'\b\w+\b', text)
        corrections = {}
        
        spell_checker = SpellChecker(language='pt')
        
        print('passa aqui')

        for word in words:
            word_lower = word.lower()
            if word_lower not in spell_checker and len(word_lower) > 2:
                # Busca correção apenas para palavras não conhecidas
                candidates = spell_checker.candidates(word_lower)
                if candidates:
                    # Pega a primeira sugestão (mais provável)
                    best_correction = list(candidates)[0]
                    corrections[word] = best_correction
        
        # Aplica as correções mantendo a capitalização original
        corrected_text = text
        for original, correction in corrections.items():
            if original[0].isupper():
                correction = correction.capitalize()
            if original.isupper() and len(original) > 1:
                correction = correction.upper()
            
            corrected_text = re.sub(r'\b' + re.escape(original) + r'\b', 
                correction, corrected_text)
        
        return corrected_text

    def _save_temp_file(self, pdf_file, file_name: str):
        """
        Salva arquivo temporário para processamento.
        """
        import tempfile
        temp_dir = tempfile.gettempdir()
        temp_path = os.path.join(temp_dir, f"temp_{file_name}")
        
        with open(temp_path, "wb") as f:
            if hasattr(pdf_file, 'read'):
                f.write(pdf_file.read())
            else:
                f.write(pdf_file)
        
        return temp_path

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
        
    def _normalize_portuguese_text(self, text):
        """Normaliza o texto em português para corrigir problemas comuns de encoding"""
        # Correções comuns para problemas de encoding em português
        replacements = {
            'Ã£': 'ã', 'Ã¡': 'á', 'Ã©': 'é', 'Ãª': 'ê', 'Ã§': 'ç',
            'Ã³': 'ó', 'Ãµ': 'õ', 'Ã­': 'í', 'Ãº': 'ú',
            'Ã‰': 'É', 'Ãƒ': 'Ã', 'Ã‡': 'Ç', 'Ã"': 'Ó', 'Ãš': 'Ú',
            'Ø': 'é', 'ª' : 'ã', 'Æ': 'á', '€': 'e'
        }
        
        for old, new in replacements.items():
            text = text.replace(old, new)
        
        # Normalização Unicode
        import unicodedata
        text = unicodedata.normalize('NFC', text)
        
        return text

    def _clean_extracted_text(self, text):
        """Limpa e corrige problemas comuns após extração"""
        import re
        
        # Remove caracteres de controle
        text = re.sub(r'[\x00-\x1F\x7F]', '', text)
        
        # Corrige espaçamentos duplos
        text = re.sub(r' +', ' ', text)
        
        # Corrige quebras de linha excessivas
        text = re.sub(r'\n{3,}', '\n\n', text)
        
        # Corrige problemas comuns com hífens em português
        text = re.sub(r'(?<=\w)- (?=\w)', '', text)  # Remove hífens desnecessários
        
        return text