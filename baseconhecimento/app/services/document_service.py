import os
import tempfile

from flask import jsonify
from pathlib import Path
from typing import List
from config.settings import Config
from typing import List, Dict

from langchain_ollama import OllamaEmbeddings
from langchain_ollama import ChatOllama
from langchain_ollama import OllamaLLM
from langchain_chroma import Chroma
from langchain.schema.document import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter

from docling.datamodel.base_models import InputFormat
from docling.datamodel.pipeline_options import PdfPipelineOptions
from docling.document_converter import DocumentConverter, PdfFormatOption

class DocumentService:
    def __init__(self, persist_directory=Config.CHROMA_DIRECTORY):
        self.persist_directory = persist_directory
        self.embeddings = OllamaEmbeddings(model=Config.MODEL_NAME)
        #self.ensure_persist_directory()
        #self.vectorstore = self._initialize_vectorstore()

        self.vectorstore = Chroma(
            collection_name='multi_modal_rag',
            persist_directory=self.persist_directory,
            embedding_function=self.embeddings
        )
        
        self.llm = ChatOllama(
            model=Config.MODEL_NAME,
            temperature=0.1
        )

        self.vision_model = OllamaLLM(
            model=Config.VISION_MODEL_NAME
            #num_gpu=1,  # Limitar uso de GPU
            #num_thread=4  # Limitar threads de CPU
        )
    
    def ensure_persist_directory(self):
        if not os.path.exists(self.persist_directory):
            os.makedirs(self.persist_directory)

        # Cria diretório para imagens se não existir
        self.images_directory = os.path.join(self.persist_directory, 'images')
        if not os.path.exists(self.images_directory):
            os.makedirs(self.images_directory)

    def _initialize_vectorstore_old(self):
        if os.path.exists(self.persist_directory):
            return Chroma(
                persist_directory=self.persist_directory,
                embedding_function=self.embeddings
            )
        return None

    def _save_temp_file(self, file_content, file_name):
        temp_dir = tempfile.gettempdir()
        temp_path = os.path.join(temp_dir, file_name)
        
        with open(temp_path, 'wb') as f:
            f.write(file_content)

        return temp_path
    
    def process_pdf(self, pdf_file, file_name: str, sector: str):
        print('o setor ', sector)
        # Salva o arquivo temporariamente
        temp_path = self._save_temp_file(pdf_file, file_name)
        
        print('temp_path: ', temp_path)
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
            
            # Preparar documentos para vetorização
            documents = []
            
            # Adicionar conteúdo de texto
            text_doc = Document(
                page_content=text,
                metadata={
                    "source": file_name,
                    "sector": sector,
                    "type": "text-content",
                    "format": "pdf"
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
            
            #print(f"Processando {len(page_nums)} páginas com imagens no documento {doc_filename}")
            
            for page_no, page in conv_res.document.pages.items():
                page_no = page.page_no
                if page_no in page_nums:
                    print(f"Processando página {page_no}")
                    
                    # Salvar imagem da página
                    image_path = f"{output_dir}/{doc_filename}-page-{page_no}.png"
                    with open(image_path, "wb") as f:
                        page.image.pil_image.save(f, format="PNG")
                    
                    # Usar o modelo de visão para obter descrição da imagem
                    try:
                        # Adaptar para usar o modelo Ollama com visão
                        image_llm = self.vision_model.bind(images=[image_path])
                        
                        # Prompt para descrição de imagem
                        prompt_old = """
                        Descreva em português em detalhes o que você vê nesta imagem extraída de um documento PDF. 
                        Inclua todos os elementos visuais importantes, textos visíveis, gráficos, tabelas e qualquer outro conteúdo relevante. 
                        Seja objetivo, claro e detalhado, priorizando informações que possam ser úteis no contexto profissional.
                        """
                        prompt = """
                        Descreva o que você vê na imagem em português. Faça de forma simples, fácil de entender e com uma linguagem muito clara. Seja o mais descritivo possível, sem perder ou pular nenhum detalhe.
                        """
                        
                        # Obter descrição da imagem
                        response = image_llm.invoke(prompt)
                        
                        #print('descrição: ', response)

                        # Criar documento com a descrição da imagem
                        image_doc = Document(
                            page_content=response,
                            metadata={
                                "source": file_name,
                                "sector": sector,
                                "type": "image-content",
                                "page": page_no,
                                "image_path": image_path,
                                "format": "image/png"
                            }
                        )
                        documents.append(image_doc)
                        print(f"Imagem da página {page_no} processada com sucesso")
                    except Exception as img_error:
                        print(f"Erro ao processar imagem da página {page_no}: {str(img_error)}")

            # Extrair tabelas e elementos estruturados
            tables = data.get("tables", [])
            for i, table in enumerate(tables):
                table_content = str(table.get("content", ""))
                table_doc = Document(
                    page_content=table_content,
                    metadata={
                        "source": file_name,
                        "sector": sector,
                        "type": "table-content",
                        "table_index": i,
                        "format": "text/table"
                    }
                )
                documents.append(table_doc)

            # Configurar o vector store por setor
            sector_persist_dir = os.path.join(self.persist_directory, sector)
            os.makedirs(sector_persist_dir, exist_ok=True)
            
            vectorstore = Chroma(
                collection_name=f'sector_{sector}',
                persist_directory=sector_persist_dir,
                embedding_function=self.embeddings
            )
            
            #print(f"Carregando collection: sector_{sector}")
            #print(f"Diretório: {os.path.join(self.persist_directory, sector)}")
            #print(f"Documentos presentes: {len(vectorstore.get()['documents'])}")

            st=RecursiveCharacterTextSplitter(chunk_size=2000,chunk_overlap=50)
            sd=st.split_documents(documents)
            
            batch_size = 10
            for i in range(0, len(sd), batch_size):
                batch = sd[i:i + batch_size]
                vectorstore.add_documents(batch)

            # Adicionar documentos ao vectorstore
            #vectorstore.add_documents(sd)
            
            # Preparar estatísticas para resposta
            stats = {
                "text_docs": sum(1 for doc in documents if doc.metadata.get("type") == "text-content"),
                "image_docs": sum(1 for doc in documents if "image" in doc.metadata.get("type", "")),
                "table_docs": sum(1 for doc in documents if doc.metadata.get("type") == "table-content"),
            }
            
            return {
                "status": "success",
                "message": f"PDF {file_name} processado com sucesso para o setor {sector}",
                "document_count": len(documents),
                "stats": stats,
                "sector": sector
            }
            
        except Exception as e:
            print(f"Erro ao processar PDF: {str(e)}")
            return {
                "status": "error",
                "message": f"Erro ao processar PDF: {str(e)}"
            }
        finally:
            # Limpa o arquivo temporário
            if os.path.exists(temp_path):
                os.remove(temp_path)
    
    def list_documents_atual(self, sector: str) -> List[Dict]:
        print('vem aqui ó')
        vectorstore = Chroma(
                collection_name=sector,
                persist_directory=self.persist_directory,
                embedding_function=self.embeddings
            )
        
        print('o vectorstore', vectorstore)

        if not vectorstore:
            return []
        
        # Obtém todos os documentos do Chroma
        collection = vectorstore._collection
        documents = collection.get()
        
        print('os documentos: ', documents)

        # Agrupa documentos por fonte (arquivo original)
        doc_dict = {}
        for i, doc in enumerate(documents['documents']):
            source = documents['metadatas'][i].get('source', 'Unknown')
            if source not in doc_dict:
                doc_dict[source] = {
                    'id': documents['ids'][i],
                    'filename': source,
                    'chunks': 1,
                    'created_at': documents['metadatas'][i].get('created_at', 'Unknown')
                }
            else:
                doc_dict[source]['chunks'] += 1

        print(doc_dict.values())
        return list(doc_dict.values())

    def delete_document_atual(self, sector: str, filename: str) -> Dict:
        try:
            vectorstore = Chroma(
                collection_name=sector,
                persist_directory=self.persist_directory,
                embedding_function=self.embeddings
            )

            if not vectorstore:
                return {"status": "error", "message": "Nenhum documento encontrado"}

            # Obtém IDs dos documentos com o filename especificado
            collection = vectorstore._collection
            documents = collection.get()
            
            # Lista para armazenar IDs a serem deletados e caminhos de imagens
            ids_to_delete = []
            image_paths = []
            
            # Itera sobre documentos e coleta IDs e caminhos de imagens
            for doc_id, metadata in zip(documents['ids'], documents['metadatas']):
                if metadata.get('source') == filename:
                    ids_to_delete.append(doc_id)
                    
                    # Se for uma imagem, adiciona o caminho para remoção
                    if metadata.get('content_type') == 'image' and metadata.get('image_path'):
                        image_paths.append(metadata['image_path'])

            if not ids_to_delete:
                return {"status": "error", "message": f"Documento {filename} não encontrado"}

            # Remove os documentos
            collection.delete(ids_to_delete)
            vectorstore.persist()

            # Remove os arquivos de imagem
            removed_images = 0
            for image_path in image_paths:
                try:
                    if os.path.exists(image_path):
                        os.remove(image_path)
                        removed_images += 1
                except Exception as img_e:
                    print(f"Erro ao remover imagem {image_path}: {str(img_e)}")

            return {
                "status": "success",
                "message": f"Documento {filename} removido com sucesso. {removed_images} imagens removidas.",
                "removed_documents": len(ids_to_delete),
                "removed_images": removed_images
            }
        except Exception as e:
            return {
                "status": "error",
                "message": f"Erro ao remover documento: {str(e)}"
            }
        
    def currentDate(self):
        from datetime import date

        dia = date.today()
        dia = dia.strftime('%d/%m/%y')
        return dia
    
    def list_documents(self, sector: str) -> List[Dict]:
        try:
            # Implementação depende de como você está armazenando os metadados dos documentos
            # Exemplo usando Chroma diretamente:
            vectorstore = Chroma(
                collection_name=f'sector_{sector}',
                persist_directory=os.path.join(self.persist_directory, sector),
                embedding_function=self.embeddings
            )
            
            # Obter os metadados de todos os documentos
            collection = vectorstore._collection
            metadatas = collection.get()["metadatas"]
            ids = collection.get()["ids"]
            
            documents_info = []
            for i, metadata in enumerate(metadatas):
                if metadata:  # Alguns metadados podem ser None
                    doc_info = {
                        "id": ids[i],
                        "source": metadata.get("source", "Unknown"),
                        "sector": metadata.get("sector", sector),
                        "type": metadata.get("type", "Unknown")
                    }
                    documents_info.append(doc_info)
            
            return jsonify(documents_info), 200
            
        except Exception as e:
            return jsonify({
                "status": "error",
                "message": f"Erro ao listar documentos: {str(e)}"
            }), 500
    
    def delete_document(self, sector: str, document_id: str) -> Dict:
        try:
            vectorstore = Chroma(
                collection_name=f'sector_{sector}',
                persist_directory=os.path.join(self.persist_directory, sector),
                embedding_function=self.embeddings
            )
            
            # Remover documento pelo ID
            vectorstore.delete([document_id])
            
            return jsonify({
                "status": "success",
                "message": f"Documento {document_id} removido com sucesso"
            }), 200
            
        except Exception as e:
            return jsonify({
                "status": "error",
                "message": f"Erro ao remover documento: {str(e)}"
            }), 500