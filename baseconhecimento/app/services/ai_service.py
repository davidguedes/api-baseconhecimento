import os
import traceback

from langchain_chroma import Chroma
from langchain_ollama import ChatOllama, OllamaEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain.prompts import PromptTemplate
from config.settings import Config
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

os.environ["LANGCHAIN_API_KEY"] = Config.LANGCHAIN_API_KEY

class AIService:
    def __init__(self, persist_directory=Config.CHROMA_DIRECTORY):
        # Inicializa o modelo LLM
        self.llm = ChatOllama(
            model=Config.MODEL_NAME,
            temperature=Config.TEMP
        )

        # Inicializa embeddings e vector store
        self.embeddings = OllamaEmbeddings(model=Config.MODEL_NAME)
        self.persist_directory = persist_directory

        self.system_prompt = "Você é um assitente prestativo e está respondendo perguntas gerais."

        self.token_s, self.token_e = "<|begin_of_text|><|start_header_id|>system<|end_header_id|>","<|eot_id|><|start_header_id|>user<|end_header_id|>"

        self.template_rag = """
            <|begin_of_text|>
            <|start_header_id|>system<|end_header_id|>
            Você é um assistente virtual prestativo e está respondendo perguntas sobre o departamento de tecnologia.
            Use os seguintes pedaços de contexto recuperado para responder à pergunta.
            Se você não sabe a resposta, apenas diga que não sabe. 
            Mantenha a resposta concisa mas informativa.
            Responda em português.
            <|eot_id|>
            <|start_header_id|>user<|end_header_id|>
            Pergunta: {pergunta}
            Contexto: {contexto}
            <|eot_id|>
            <|start_header_id|>assistant<|end_header_id|>
        """

        self.template_rag_sector = """
            <|begin_of_text|>
            <|start_header_id|>system<|end_header_id|>
            Você é um assistente virtual especializado para o departamento de {setor}.
            Sua função é fornecer informações úteis baseadas apenas nos documentos da base de conhecimento.

            INSTRUÇÕES:
            1. Use SOMENTE as informações fornecidas no contexto para responder às perguntas.
            2. Se o contexto não contiver informações, admita que não sabe a resposta em vez de inventar.
            3. Cite as fontes dos documentos que você utilizou na sua resposta, mencionando nomes dos arquivos e páginas quando disponíveis.
            4. Mantenha suas respostas concisas, objetivas e bem estruturadas.
            5. Responda sempre em português.

            NUNCA invente informações que não estejam presentes no contexto fornecido.
            <|eot_id|>

            <|start_header_id|>user<|end_header_id|>
            Pergunta: {pergunta}

            Contexto:
            {contexto}
            <|eot_id|>

            <|start_header_id|>assistant<|end_header_id|>
        """

        self.prompt_rag = PromptTemplate.from_template(self.template_rag)

    def get_retriever(self, sector):
        """Obtém o retriever específico para o setor"""
        try:
            # Caminho completo para o diretório de persistência do setor
            sector_persist_dir = os.path.join(self.persist_directory, sector)

            # Verificar se o diretório existe
            if not os.path.exists(sector_persist_dir):
                print(f"Diretório para o setor {sector} não encontrado: {sector_persist_dir}")
                return None
            
            # Carregar o vector store do setor específico
            vectorstore = Chroma(
                collection_name=f'sector_{sector}',
                persist_directory=sector_persist_dir,
                embedding_function=self.embeddings
            )

            # Verificar se há documentos no vectorstore
            collection_count = vectorstore._collection.count()
            if collection_count == 0:
                print(f"Nenhum documento encontrado para o setor {sector}")
                return None
            
            print(f"Vectorstore para setor {sector} carregado com sucesso. Documentos: {collection_count}")

            # Configurar retriever
            retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 3})
            #vectorstore.as_retriever(
                #search_type="mmr",  # Maximal Marginal Relevance para melhor diversidade nos resultados
                #search_kwargs={
                #    "k": 5,  # Aumentando para 5 documentos (ajuste conforme necessário)
                #    "fetch_k": 20,  # Buscar mais documentos antes de filtrar
                #    "lambda_mult": 0.7,  # Balanceamento entre relevância e diversidade
                #    "filter": None  # Sem filtros por enquanto, mas útil para diagnóstico
                #}
            #)

            return retriever
        except Exception as e:
            print(f"Erro ao carregar retriever para o setor {sector}: {str(e)}")
            # Retornar um retriever vazio ou fallback
            return None

    def generate_response(self, user_sector: str, user_message: str, sector_name: str) -> str:
        """Gera uma resposta para a mensagem do usuário"""
        return self.question(user_sector, sector_name, user_message)

    def question(self, sector: str, sector_name: str, message: str):
        """Processa a pergunta usando RAG e retorna a resposta"""
        # Obter retriever para o setor
        self.retriever = self.get_retriever(sector)
        
        if not self.retriever:
            return {
                "text": "Não foi possível acessar a base de conhecimento para este setor.",
                "images": []
            }
        
        prompt_rag = PromptTemplate(
            input_variables=["setor", "contexto", "pergunta"],
            template=self.template_rag_sector,
        )

        try:
            # Recuperar documentos relevantes
            docs = self.retriever.get_relevant_documents(message)

            print(f"Documentos recuperados: {len(docs)}")

            # Se não encontrou documentos relevantes
            if not docs:
                return {
                    "text": "Não encontrei informações relevantes para sua pergunta na base de conhecimento deste setor."
                }
            
            print('Docs relevantes: ', docs)

            # Formatar documentos em contexto
            context_parts = []
            for doc in docs:
                source = doc.metadata.get("source", "Desconhecido")
                doc_type = doc.metadata.get("type", "texto")
                page = doc.metadata.get("page", "")
                page_info = f" (Página: {page})" if page else ""
                
                # Formatar o contexto com informações de origem
                context_parts.append(f"--- Documento: {source}{page_info} (Tipo: {doc_type}) ---\n{doc.page_content}")
            
            context = "\n\n".join(context_parts)
            
            print(f"Tamanho do contexto: {len(context)} caracteres")

            # Gerar resposta usando o LLM
            formatted_prompt = prompt_rag.format(setor=sector_name, contexto=context, pergunta=message)
            print('O prompt: ', formatted_prompt)
            response_text = self.llm.invoke(formatted_prompt)
            
            # Retornar texto
            return {
                "text": response_text
            }
        except Exception as e:
            print(f"Erro ao processar pergunta: {str(e)}")
            traceback.print_exc()
            return {
                "text": f"Desculpe, ocorreu um erro ao processar sua pergunta: {str(e)}"
            }