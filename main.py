import streamlit as st
import json
from PIL import Image
from groq import Groq
import re
import requests
import logging
from io import BytesIO
import base64

# Configurar logging
logging.basicConfig(level=logging.DEBUG)

def convert_to_base64(pil_image, max_size=(1024, 1024)):
    """
    Convert PIL images to Base64 encoded strings and resize to reduce size
    """
    logging.debug("Converting image to base64")
    pil_image.thumbnail(max_size)
    
    # Convertir a modo RGB si la imagen está en modo RGBA
    if pil_image.mode == 'RGBA':
        pil_image = pil_image.convert('RGB')
    
    buffered = BytesIO()
    pil_image.save(buffered, format="JPEG", quality=150)
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
                
                # Comprobar si el nombre contiene crop_name
                if crop_name_normalized in nombre_normalized:
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
        if crop_name.lower().endswith('es'):
            crop_name_singular = crop_name[:-2]
        elif crop_name.lower().endswith('s'):
            crop_name_singular = crop_name[:-1]
        else:
            crop_name_singular = crop_name
        
        codigo, nombre = buscar_cultivo(crop_name_singular, data)
        if codigo is not None and nombre is not None:
            return codigo, nombre
        
        # Manejar las variaciones de género
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

           # Create the prompt
            prompt_text = f"""
            Rol: Eres un experto en análisis agrícola encargado de clasificar imágenes de cultivos con el más alto nivel de precisión para garantizar que nuestras decisiones comerciales se basen en datos confiables y precisos.

            Tarea: Analiza la siguiente imagen: {uploaded_file}. Tu objetivo es identificar el cultivo principal presente en la imagen utilizando un enfoque metódico y detallado.

            Instrucciones:

            1. Observación inicial: Realiza un análisis visual general, observando aspectos como la forma y el color de las plantas, la disposición de los tallos y hojas, la textura y cualquier otro patrón visual relevante.
            2. Características del entorno: Examina el paisaje circundante, incluyendo el tipo de suelo, las condiciones climáticas visibles y cualquier elemento que pueda influir en el tipo de cultivo.
            3. Comparación: Relaciona las características observadas con las propiedades conocidas de diferentes cultivos, haciendo comparaciones para estrechar las posibles opciones.
            4. Identificación del cultivo: Indica cuál es el cultivo más probable presente en la imagen.

            **Formato de respuesta (por favor, sigue estrictamente este formato):**

            "tipo de cultivo:" [nombre del cultivo en español, en minúsculas].
            "razonamiento:" [explicación detallada de por qué elegiste este cultivo, incluyendo características específicas observadas en la imagen y cómo se alinean con el cultivo identificado].

            **Nota**: Responde en español peninsular, utilizando términos como "patata" en lugar de "papa" y "aguacate" en lugar de "palta". Sigue las normas gramaticales y expresiones típicas de esta variante del idioma.

            Gracias por tu análisis detallado y preciso, que es esencial para nuestras operaciones.
            """




            # Create the completion request
            # Solicitud a la API de Groq
            completion = client.chat.completions.create(
                model="llama3-groq-8b-8192-tool-use-preview",
                messages=[
                    {
                        "role": "user", "content": prompt_text
                    }
                ],
                temperature=0.3,
                max_tokens=1500,
                top_p=0.9,
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
             # Verificar el resultado antes de mostrarlo
            if result:
                st.json(result)
            else:
                st.error("Error al generar el resultado.")
            st.write("A continuación, se muestra la respuesta sin formatear, obtenida del modelo:")
            st.write(response.content)
    else:
        st.error("Por favor, sube una imagen y proporciona tu API key.")