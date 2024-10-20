import streamlit as st
import requests
import json
from PIL import Image
import io
from langchain import PromptTemplate

# URL de la API de Groq (deberías reemplazarla con la URL correcta)
GROQ_API_URL = "https://api.groq.com/v1/llama-3.2-90B-vision-preview"

# Función para cargar el archivo JSON de cultivos
def cargar_cultivos_json():
    with open("cultivos.json", "r") as file:
        return json.load(file)

# Función para enviar la imagen y el prompt a la API de Groq
def enviar_a_groq(api_key, image_data, prompt):
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }
    payload = {
        "image": image_data,
        "prompt": prompt
    }
    response = requests.post(GROQ_API_URL, headers=headers, json=payload)
    return response.json()

# Configuración de la aplicación Streamlit
st.title("Análisis de Cultivos con LLM")
st.write("Esta aplicación utiliza un modelo de lenguaje de Groq para identificar el cultivo presente en una imagen.")
st.header("Instrucciones")

# Pedir la API key de Groq
groq_api_key = st.text_input("Ingrese su API key de Groq:")

# Cargar el archivo JSON de cultivos
cultivos = cargar_cultivos_json()

# Subir la imagen
uploaded_image = st.file_uploader("Subir imagen", type=["jpg", "jpeg", "png"])

# Definir el template del prompt
template = "¿Cuál es el cultivo presente en la fotografía? {tipo_cultivo}"
prompt_template = PromptTemplate(
    input_variables=["tipo_cultivo"],
    template=template,
)

# Pedir el tipo de cultivo
tipo_cultivo = st.text_input("Tipo de cultivo", value="")

# Botón para procesar la imagen
if st.button("Submit"):
    if groq_api_key and uploaded_image:
        # Convertir la imagen a bytes
        image = Image.open(uploaded_image)
        image_bytes = io.BytesIO()
        image.save(image_bytes, format='JPEG')
        image_data = image_bytes.getvalue()

        # Generar el prompt utilizando el template
        prompt = prompt_template.format(tipo_cultivo=tipo_cultivo)

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

# Ejecutar la aplicación
if __name__ == "__main__":
    st.run()