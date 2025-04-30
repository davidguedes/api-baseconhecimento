import os
import base64
from io import BytesIO
import time
from PIL import Image
from langchain_ollama import OllamaLLM

def convert_to_base64(pil_image, format="PNG"):
    """
    Converte uma imagem PIL para base64.
    
    Args:
        pil_image (PIL.Image): Imagem a ser convertida
        format (str): Formato da imagem (padrão: PNG)
    
    Returns:
        str: Representação da imagem em base64
    """
    buffered = BytesIO()
    pil_image.save(buffered, format=format)
    img_str = base64.b64encode(buffered.getvalue()).decode("utf-8")
    return img_str

def extract_image_content(image_path, llm):
    """
    Extrai conteúdo de uma imagem usando o modelo LLM.
    
    Args:
        image_path (str): Caminho da imagem
        llm (OllamaLLM): Modelo de linguagem para extração
    
    Returns:
        str: Conteúdo extraído da imagem
    """
    # Carregar a imagem
    pil_image = Image.open(image_path)
    
    # Converter para base64
    image_b64 = convert_to_base64(pil_image)
    
    # Prompt de extração
    prompt = "Observe a imagem e extraia APENAS o conteúdo textual ou númerico bruto, sem nenhuma informação a mais ou comentário adicional. Não adicione informações acerca de contexto. Não adicione descrições como 'The image contains the text', 'The image shows a white background' ou coisas do tipo. O conteúdo das imagens possuem textos em portugues e/ou números."
    
    # Configurar LLM com imagem
    llm_with_image_context = llm.bind(images=[image_b64])
    
    # Calcular tempo de resposta
    inicio = time.time()
    res = llm_with_image_context.invoke(prompt)
    fim = time.time()
    
    return {
        'content': res,
        'response_time': fim - inicio,
        'filename': os.path.basename(image_path)
    }

def process_png_images(base_dir, output_file):
    """
    Percorre pastas e processa imagens PNG.
    
    Args:
        base_dir (str): Diretório base para busca de imagens
        output_file (str): Arquivo de saída para resultados
    """
    # Configurar o LLM
    llm = OllamaLLM(model="llama3.2-vision")
    
    # Abrir arquivo de saída
    with open(output_file, 'w', encoding='utf-8') as f:
        # Percorrer todas as subpastas
        for root, dirs, files in os.walk(base_dir):
            for file in files:
                if file.lower().endswith('.png'):
                    image_path = os.path.join(root, file)
                    
                    try:
                        # Extrair conteúdo da imagem
                        resultado = extract_image_content(image_path, llm)
                        
                        # Escrever resultados no arquivo
                        f.write(f"Arquivo: {resultado['filename']}\n")
                        f.write(f"Tempo de Resposta: {resultado['response_time']:.2f} segundos\n")
                        f.write("Conteúdo Extraído:\n")
                        f.write(resultado['content'] + "\n\n")
                        
                        print(f"Processado: {image_path}")
                    
                    except Exception as e:
                        print(f"Erro ao processar {image_path}: {e}")

# Uso do script
base_directory = "C:/Users/David/Desktop/test"
output_file_path = "resultados_extracao.txt"

process_png_images(base_directory, output_file_path)