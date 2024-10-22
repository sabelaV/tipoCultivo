import streamlit as st
import json
from PIL import Image
import base64
from io import BytesIO
from langchain_core.prompts import PromptTemplate
from langchain.schema import HumanMessage
from langchain_core.output_parsers import StrOutputParser
from groq import Groq
import re
import requests
import torch
from transformers import BlipProcessor, BlipForConditionalGeneration
import logging
import pdb

# Configurar logging
logging.basicConfig(level=logging.DEBUG)

def convert_to_base64(pil_image, max_size=(900, 900)):
    """
    Convert PIL images to Base64 encoded strings and resize to reduce size
    """
    logging.debug("Converting image to base64")
    pil_image.thumbnail(max_size)
    
    # Convertir a modo RGB si la imagen está en modo RGBA
    if pil_image.mode == 'RGBA':
        pil_image = pil_image.convert('RGB')
    
    buffered = BytesIO()
    pil_image.save(buffered, format="JPEG", quality=85)
    img_str = base64.b64encode(buffered.getvalue()).decode("utf-8")
    return img_str

def upload_image_to_imgur(image_b64, client_id):
    """
    Upload an image to Imgur and return the URL
    """
    logging.debug("Uploading image to Imgur")
    headers = {"Authorization": f"Client-ID {client_id}"}
    data = {"image": image_b64, "type": "base64"}
    response = requests.post("https://api.imgur.com/3/image", headers=headers, data=data)
    if response.status_code == 200:
        return response.json()["data"]["link"]
    else:
        st.error("Error al subir la imagen a Imgur")
        return None

# Define the Imgur client ID as a constant
IMGUR_CLIENT_ID = "bb01075a0d68448"

# Input for the API key
groq_api_key = st.text_input("Introduce tu API key de Groq", type="password")

# Initialize the Groq client with the API key
client = Groq(api_key=groq_api_key)

# Load the JSON file of crops and search for the crop type
def cargar_cultivos_json(crop_name):
    def buscar_cultivo(crop_name, data):
        # Normalizar el nombre del cultivo
        crop_name_normalized = crop_name.lower()

        for cultivo in data:
            if isinstance(cultivo, list) and len(cultivo) >= 2:
                codigo, nombre = cultivo
                nombre_normalized = nombre.lower()
                
                # Comprobar si el nombre comienza con crop_name
                if nombre_normalized.startswith(crop_name_normalized):
                    # Asegurarse de que hay un espacio después del nombre buscado o que es una coincidencia exacta
                    if len(nombre) == len(crop_name) or nombre[len(crop_name)] == ' ':
                        return codigo, nombre
        
        return None, None
    
    # Eliminar comillas al principio y al final si están presentes
    crop_name = crop_name.strip("'\"")

    with open('productos.json', 'r', encoding='utf-8') as file:
        data = json.load(file).get('rows', [])  # Asegurarse de obtener la lista de cultivos
        
        # Intentar buscar el nombre del cultivo tal cual
        codigo, nombre = buscar_cultivo(crop_name, data)
        if codigo is not None and nombre is not None:
            return codigo, nombre
        
        # Si no se encuentra, intentar buscar sin la 's' o 'es' final
        crop_name_singular = crop_name.rstrip('es').rstrip('s')
        
        codigo, nombre = buscar_cultivo(crop_name_singular, data)
        if codigo is not None and nombre is not None:
            return codigo, nombre
        
        # Manejar las variaciones de género
        # En este caso, simplemente agregamos las versiones masculinas y femeninas comunes
        posibles_variantes = [crop_name_singular]  # Iniciamos con la versión singular
        if crop_name_singular.endswith('a'):  # Si termina en 'a', puede ser femenino
            posibles_variantes.append(crop_name_singular[:-1] + 'o')  # Cambia 'a' por 'o'
        elif crop_name_singular.endswith('o'):  # Si termina en 'o', puede ser masculino
            posibles_variantes.append(crop_name_singular[:-1] + 'a')  # Cambia 'o' por 'a'

        # Intentar buscar cada variante
        for variante in posibles_variantes:
            codigo, nombre = buscar_cultivo(variante, data)
            if codigo is not None and nombre is not None:
                return codigo, nombre
        
        return None, None

# Function to extract crop name from the response
def extract_crop_name(response_text):
    logging.debug("Extracting crop name from response")
    match = re.search(r'tipo de cultivo: ([^\n]+)', response_text, re.IGNORECASE)
    if match:
        return match.group(1).strip()
    return None

# Initialize the BLIP model and processor
processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")

# Function to process image and generate description
def describe_image(image_path):
    logging.debug("Describing image")
    image = Image.open(image_path)
    inputs = processor(images=image, return_tensors="pt")

    # Prepare a context prompt to focus on crops
    context_prompt = "Describe the image focusing on identifying types of crops."
    inputs['input_ids'] = processor.tokenizer(context_prompt, return_tensors="pt").input_ids

    descriptions = []
    for _ in range(3):  # Generate 5 descriptions
        with torch.no_grad():
            generated_ids = model.generate(
                inputs['pixel_values'],
                max_length=400,
                do_sample=True,
                temperature=0.1,
                top_k=50,
                top_p=0.95,
                num_return_sequences=1
            )

        description = processor.decode(generated_ids[0], skip_special_tokens=True)
        descriptions.append(description)
    
    # Combine the descriptions into a single detailed description
    detailed_description = " ".join(descriptions)
    return detailed_description

# Streamlit app setup
st.title("Analizador de Imágenes de Cultivos Agrícolas")
st.subheader("Aplicación de LangChain con Servicio en la Nube")
st.markdown("Esta aplicación utiliza un modelo de lenguaje, ejecutado en hardware proporcionado por Groq, para analizar imágenes de cultivos agrícolas y determinar el tipo de cultivo principal en la imagen")
st.markdown("**Autor:** SabelaV")
st.markdown("""
### Instrucciones:
1. Introduce tu API key de Groq en el campo correspondiente. Si no tienes una, puedes obtenerla [aquí](https://console.groq.com/login).
2. Sube una imagen de un cultivo agrícola utilizando el botón "Browse Files".
3. Haz clic en el botón "Procesar imagen".
4. Espera mientras la imagen se procesa y se genera una descripción detallada.
5. La aplicación mostrará el tipo de cultivo principal detectado en la imagen, junto con un razonamiento.
""")

# File uploader for the image
uploaded_file = st.file_uploader("Sube una imagen", type=["jpg", "jpeg", "png"])

# Button to process the image
if st.button("Procesar imagen"):
    if uploaded_file and groq_api_key:
        with st.spinner('Procesando la imagen y generando la descripción...'):
            # Convert the uploaded file to a PIL image
            pil_image = Image.open(uploaded_file)
            
            # Load and convert the image to base64
            image_b64 = convert_to_base64(pil_image)

            # Upload the image to Imgur
            image_url = upload_image_to_imgur(image_b64, IMGUR_CLIENT_ID)
            if not image_url:
                st.error("Error al subir la imagen a Imgur")
                st.stop()

            # Generate a detailed description of the image using BLIP
            detailed_description = describe_image(uploaded_file)
            #st.write(f"Descripción generada de la imagen: {detailed_description}")

            # Create the prompt
            prompt_text = f"""
            Eres un asistente experimentado en el análisis de imágenes de cultivos agrícolas que categoriza con precisión las fotografías ajustándose a unas listas de valores dadas.

            Analiza la siguiente descripción de la imagen y determina el tipo de cultivo principal que aparece en la fotografía devolviendo el siguiente formato:
            "tipo de cultivo:" el nombre de ese tipo de cultivo principal detectado en la fotografía EN ESPAÑOL.
            \n
            "razonamiento:" una explicación detallada del por qué elegiste este cultivo.

            Por favor, responde en español de España (español peninsular). Utiliza términos como "patata" en lugar de "papa" y "aguacate" en lugar de "palta", y sigue las normas gramaticales y expresiones típicas de esta variante del idioma.

            Descripción de la imagen: {detailed_description}

            Esta tarea es crítica para el éxito de nuestro negocio, por lo tanto proporciona un análisis exhaustivo de cada fotografía.

            Tu categorización precisa es muy apreciada y contribuye a la eficacia de nuestras operaciones.
            """

            # Create the completion request
            #Solicitud a la API de Groq
            completion = client.chat.completions.create(
                model="llama-3.2-90b-vision-preview",
                messages=[
                    {
                        "role": "user", "content": prompt_text
                    }
                ],
                temperature=0.5,
                max_tokens=1024,
                top_p=1,
                stream=False,
                stop=None,
            )

            # Process the response
            response = completion.choices[0].message
            
            # Extract the crop name from the response
            crop_name = extract_crop_name(response.content)
            logging.debug(f"Response content: {response.content}")
            logging.debug(f"Crop name extracted: {crop_name}")

            # Extract the reasoning from the response
            response_lower = response.content.lower()
            razonamiento = response_lower.split("razonamiento:")[1].strip() if "razonamiento:" in response_lower else "No se proporcionó razonamiento."
            if crop_name:
                codigo, descripcion = cargar_cultivos_json(crop_name)
                logging.debug(f"Crop data: {codigo}, {descripcion}")
                if codigo and descripcion:
                    result = {
                        "codigo": codigo,
                        "descripcion": descripcion,
                        "razonamiento": razonamiento
                    }
                else:
                    result = {
                        "codigo": -1,
                        "descripcion": "No se encontró el cultivo en el archivo JSON.",
                        "razonamiento": razonamiento
                    }
            else:
                result = {
                    "codigo": -1,
                    "descripcion": "sin determinar",
                    "razonamiento": razonamiento
                }
            st.json(result)
            st.write("A continuación, se muestra la respuesta sin formatear, obtenida del modelo:")
            st.write(response.content)
    else:
        st.error("Por favor, sube una imagen y proporciona tu API key.")