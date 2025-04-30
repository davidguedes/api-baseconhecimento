import os
import base64
from io import BytesIO
import time
from PIL import Image
from langchain_ollama import OllamaLLM
import logging

def setup_logging():
    """
    Configura o logging para registrar informações e erros.
    """
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s: %(message)s',
        handlers=[
            logging.FileHandler('image_extraction.log', encoding='utf-8'),
            logging.StreamHandler()
        ]
    )
    return logging.getLogger(__name__)

def convert_to_base64(pil_image, format="PNG"):
    """
    Converte uma imagem PIL para base64 de forma eficiente.
    """
    buffered = BytesIO()
    pil_image.save(buffered, format=format)
    return base64.b64encode(buffered.getvalue()).decode("utf-8")

def extract_image_content(image_path, llm):
    """
    Extrai conteúdo de uma imagem com tratamento de erros e logging.
    
    Args:
        image_path (str): Caminho completo da imagem
        llm (OllamaLLM): Modelo de linguagem
        logger (logging.Logger): Objeto de logging
    
    Returns:
        dict: Informações extraídas da imagem
    """
    # Carregar a imagem
    with Image.open(image_path) as pil_image:
        # Verificar se a imagem não está vazia
        if pil_image.getbbox() is None:
            print(f"Imagem vazia: {image_path}")
            return None
        
        # Converter para base64
        image_b64 = convert_to_base64(pil_image)
        
        # Prompt otimizado para extração direta
        prompt = (
            "Observe a imagem e extraia APENAS o conteúdo textual e/ou númerico bruto, sem nenhuma informação extra ou comentário adicional. Não adicione descrições como 'The image contains the text', 'O conteúdo textual desta imagem é' ou coisas do tipo. Considere que as imagens provenientes são de origem brasileira."
        )
        
        # Configurar LLM com imagem
        llm_with_image_context = llm.bind(images=[image_b64])
        
        # Medir tempo de resposta
        inicio = time.time()
        res = llm_with_image_context.invoke(prompt).strip()
        fim = time.time()
        
        # Registrar log de sucesso
        print(f"Processado com sucesso: {os.path.basename(image_path)}")
        
        return {
            'filename': os.path.basename(image_path),
            'filepath': image_path,
            'content': res,
            'response_time': fim - inicio
        }

def process_png_images(base_dir, output_file):
    """
    Processa imagens PNG em um diretório, com logging detalhado.
    
    Args:
        base_dir (str): Diretório base para busca de imagens
        output_file (str): Arquivo de saída para resultados
    """
    # Configurar logging
    #logger = setup_logging()
    
    # Configurar o LLM
    llm = OllamaLLM(model="llama3.2-vision")
    
    # Preparar arquivo de saída

    with open(output_file, 'w', encoding='utf-8') as f:
        # Contador de imagens processadas
        total_processadas = 0
        total_sucesso = 0
        
        # Percorrer todas as subpastas
        for root, _, files in os.walk(base_dir):
            for file in files:
                # Verificar extensões de imagem (adicionar mais se necessário)
                if file.lower().endswith(('.png', '.jpg', '.jpeg', '.webp')):
                    total_processadas += 1
                    image_path = os.path.join(root, file)
                    
                    # Extrair conteúdo da imagem
                    resultado = extract_image_content(image_path, llm)
                    
                    if resultado:
                        total_sucesso += 1
                        # Formato de saída compacto
                        f.write(f"{resultado['filepath']}||{resultado['content']}\n")
                        
        # Resumo do processamento
        print(f"Total de imagens processadas: {total_processadas}")
        print(f"Imagens processadas com sucesso: {total_sucesso}")

# Uso do script
if __name__ == "__main__":
    base_directory = "C:/Users/David/Desktop/test"
    output_file_path = "resultados_extracao.txt"

process_png_images(base_directory, output_file_path)