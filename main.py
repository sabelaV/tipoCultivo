import streamlit as st
import requests
import json
from PIL import Image
import io
import base64
from langchain import PromptTemplate

# URL de la API de Groq (deberías reemplazarla con la URL correcta)
GROQ_API_URL = "https://api.groq.com/v1/llama-3.2-90B-vision-preview"

# Función para cargar el archivo JSON de cultivos
def cargar_cultivos_json():
    with open("cultivos.json", "r") as file:
        return json.load(file)
    
# Función para convertir imagen a base64
def convertir_imagen_a_base64(image_data):
    image_base64 = base64.b64encode(image_data).decode('utf-8')
    return image_base64

# Función para enviar la imagen y el prompt a la API de Groq
def enviar_a_groq(api_key, image_data, prompt):
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }
    # Codificar los datos de la imagen a Base64
    image_base64 = convertir_imagen_a_base64(image_data)
    payload = {
        "image": image_base64,  # Enviar la imagen como una cadena codificada en Base64
        "prompt": prompt
    }
    response = requests.post(GROQ_API_URL, headers=headers, json=payload)
    return response.json()

# Configuración de la aplicación Streamlit
st.title("Análisis de Cultivos con LLM")
st.write("Esta aplicación utiliza un modelo de lenguaje de Groq para identificar el cultivo presente en una imagen.")
st.header("Instrucciones")

# Pedir la API key de Groq
groq_api_key = st.text_input("Ingrese su API key de Groq:", type="password")

# Cargar el archivo JSON de cultivos
cultivos = cargar_cultivos_json()

# Subir la imagen
uploaded_image = st.file_uploader("Subir imagen", type=["jpg", "jpeg", "png"])

# Definir el template del prompt para que el usuario indique el rol del LLM
template = """
Quiero que adoptes el rol de un {rol} experto. 
Por favor, analiza la imagen que te proporciono y determina el tipo de cultivo que aparece en ella.
"""
prompt_template = PromptTemplate(
    input_variables=["rol"],
    template=template,
)

# Pedir al usuario que defina el rol del LLM
rol_usuario = st.text_input("Define el rol del LLM (ej. 'experto ingeniero agrónomo')", value="experto ingeniero agrónomo")

# Botón para procesar la imagen
if st.button("Submit"):
    if groq_api_key and uploaded_image:
        # Convertir la imagen a bytes
        image = Image.open(uploaded_image)
        image_bytes = io.BytesIO()
        image.save(image_bytes, format='JPEG')
        image_data = image_bytes.getvalue()

         # Generar el prompt utilizando el template y el rol proporcionado por el usuario
        prompt = prompt_template.format(rol=rol_usuario)

        # Enviar la imagen y el prompt a la API de Groq
        response = enviar_a_groq(groq_api_key, image_data, prompt)

        # Obtener el código del cultivo del resultado de la API
        codigo_cultivo = response.get("codigo_cultivo", "Código no encontrado")

        # Buscar la descripción del cultivo en el archivo JSON
        descripcion_cultivo = cultivos.get(codigo_cultivo, "Descripción no encontrada")

        # Mostrar el resultado
        st.write(f"Código del cultivo: {codigo_cultivo}")
        st.write(f"Descripción del cultivo: {descripcion_cultivo}")
    else:
        st.error("Por favor, ingrese su API key de Groq y suba una imagen.")