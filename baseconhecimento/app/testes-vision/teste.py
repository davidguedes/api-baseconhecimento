import base64
from io import BytesIO
import time
from PIL import Image
from IPython.display import HTML, display
from langchain_ollama import OllamaLLM

def convert_to_base64(pil_image, format="WEBP"):
    buffered = BytesIO()
    pil_image.save(buffered, format=format)
    img_str = base64.b64encode(buffered.getvalue()).decode("utf-8")
    return img_str

# Carregar a imagem WebP
file_path = "etiquetas/etiqueta_28.webp"
pil_image = Image.open(file_path)

# Converter para base64, usando PNG
image_b64 = convert_to_base64(pil_image, format="PNG")

# Visualizar a imagem (opcional)
display(HTML(f'<img src="data:image/png;base64,{image_b64}" />'))

# Configurar o LLM
llm = OllamaLLM(model="llama3.2-vision")

#. Em caso de etiqueta vermelha, o retorno terá dois valores sendo eles o valor sem a promoção aplicada e o valor promocional.

# Prompt melhorado
#prompt = "Extraia o preço e o nome completo (o nome pode apresentar infomações a mais sobre o produto porém sera tratado como parte do nome do produto) do produto dessa etiqueta de preço e nada mais. Caso a informação seja extraida, a organize em 'Nome do produto' e 'Preço'."

prompt = """
Observe atentamente a etiqueta do produto e extraia APENAS as seguintes informações em formato JSON:
{
  "titulo_produto": "TEXTO COMPLETO DA LINHA SUPERIOR DA ETIQUETA",
  "preco": "VALOR NUMÉRICO VISÍVEL NA ETIQUETA"
}

INSTRUÇÕES IMPORTANTES:
1. O título do produto é TODO o texto na parte superior da etiqueta, incluindo TODAS as palavras, números, pontos, barras e abreviações.
2. Mesmo que o texto contenha termos como "MEDIO", "PREMIUM", ou abreviações como "PREM.", "DIVA" ou contenha pontos/barras, capture o texto EXATAMENTE como aparece.
3. Se houver múltiplas linhas na área superior, capture TODAS como parte do título.
4. NÃO separe ou omita nenhuma parte do texto do título, não importa qual seja.
5. Para o preço, extraia apenas o valor monetário principal.
6. Retorne APENAS as infomações pedidas sem explicações adicionais.

Exemplo: Se na parte superior estiver escrito "MAÇA FUJI PREM.TIPO 1", o título deve ser exatamente "MAÇA FUJI PREM.TIPO 1" e não apenas "MAÇA FUJI".
"""

#prompt = "Considere a etiqueta em questão de um único produto e extraia o label completo como sendo o título do produto e seu preço e nada mais. Considere como label o texto completo presente na parte superior da etiqueta (o nome pode possuir pontos ou barras mas ainda deve ser considerado como nome)."

# Calcular tempo de resposta
inicio = time.time()
llm_with_image_context = llm.bind(images=[image_b64])
res = llm_with_image_context.invoke(prompt)
fim = time.time()

tempo_resposta = fim - inicio
print(f'Tempo de resposta: {tempo_resposta:.2f} segundos')
print('Resultado:')
print(res)