import pandas as pd
import tempfile
from dotenv import load_dotenv
from huggingface_hub import login
from transformers import AutoTokenizer, AutoModelForCausalLM, TextStreamer, BitsAndBytesConfig
import torch
import os

SYSTEM_PROMPT = """Rol: Generador de datos sintéticos adaptable a múltiples dominios, basado en archivos CSV de entrada.

Objetivo: A partir de un archivo CSV con encabezados, generar 20 filas de datos sintéticos que sigan la estructura y el contexto implícito en las columnas. Si el usuario proporciona una descripción adicional, esta debe usarse para enriquecer la generación.

Entradas:
- Archivo CSV con encabezados válidos (obligatorio).
- Descripción textual del contexto o tipo de datos (opcional).

Comportamiento:
- Si se proporciona una descripción, debe usarse para guiar la generación de los datos sintéticos (ej. "clientes de una tienda online en Europa").
- Si no se proporciona descripción, los datos deben generarse solo en función de los nombres de las columnas y su tipo de contenido.
- Los datos generados deben ser coherentes, variados y realistas según lo esperable para cada tipo de campo.

Manejo de errores:
- Si el archivo no es un CSV válido, devolver: "❌ Error: El archivo proporcionado no es un CSV válido o no tiene encabezados claros."
- Si no se puede inferir la estructura, devolver: "❌ Error: No se pudieron detectar columnas válidas en el archivo."

Formato de salida:
- **Vista previa**: Una tabla csv con las primeras 5–10 filas de los datos sintéticos generados.
- **Archivo descargable**: Un CSV con exactamente 20 filas, que el usuario puede descargar.
- La tabla csv debe tener un título de nivel 1 con la descripción del usuario si está presente, o "Datos sintéticos generados" si no hay descripción.

Ejemplo de interacción:

**Usuario:**
Archivo CSV con columnas: `nombre, edad, ciudad`
Descripción: *(vacía)*

**Respuesta esperada:**

nombre,edad,ciudad
Juan,25,Madrid
Ana,30,Barcelona
Luis,22,Valencia

**Nota:** Asegúrate de que la salida sea un CSV válido, sin formato Markdown ni tablas. La salida debe ser directamente descargable como un archivo CSV.
"""

def escribe(archivo_csv=None, descripcion="", modelo_id=None):
    """
    Genera 20 filas de datos sintéticos basados en un archivo CSV de entrada y una descripción opcional.
    Retorna un CSV como texto plano sin formato adicional.
    """
    load_dotenv()

    model_name = modelo_id or os.getenv("MODEL")
    api_key = os.getenv("API_KEY")

    if not api_key:
        return "❌ Error: No se encontró la API Key. Asegúrate de que esté configurada en tus variables de entorno (.env)."
    
    login(api_key, add_to_git_credential=True)

    descripcion = descripcion.strip()
    tiene_archivo = archivo_csv is not None

    if tiene_archivo:
        try:
            if hasattr(archivo_csv, 'name'):
                df = pd.read_csv(archivo_csv.name)
            else:
                df = pd.read_csv(archivo_csv)
        except Exception as e:
            return f"❌ Error: El archivo proporcionado no es un CSV válido o no pudo leerse. Detalles: {e}"

        if df.empty or df.columns.empty:
            return "❌ Error: No se pudieron detectar columnas válidas en el archivo."

        columnas = list(df.columns)
        ejemplo = df.head(3).to_csv(index=False)

        prompt_usuario = f"""
Archivo CSV con columnas: {', '.join(columnas)}
Ejemplo de contenido:
{ejemplo}

Descripción: {descripcion if descripcion else '(no proporcionada)'}

Por favor, genera exactamente 20 filas de datos sintéticos coherentes con la estructura y el contexto, 
y devuelve la salida en formato CSV separado por comas, sin ningún formato Markdown ni tabla.
"""
    elif descripcion:
        prompt_usuario = f"""
Descripción del usuario: {descripcion}

Objetivo: Generar 20 filas de datos sintéticos coherentes con la descripción anterior.

Por favor, devuelve la salida en formato CSV separado por comas, sin formato Markdown ni tablas.
"""
    else:
        return "❌ Error: Debes proporcionar un archivo CSV o una descripción para generar los datos."

    quant_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_quant_type="nf4"
    )

    try:
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        tokenizer.pad_token = tokenizer.eos_token
        streamer = TextStreamer(tokenizer)

        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            device_map="auto",
            quantization_config=quant_config,
            low_cpu_mem_usage=True
        )
    except Exception as e:
        return f"❌ Error al cargar el modelo o el tokenizer: {e}"

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": prompt_usuario}
    ]

    inputs = tokenizer.apply_chat_template(messages, return_tensors="pt").to("cuda")

    outputs = model.generate(inputs, max_new_tokens=2000, pad_token_id=tokenizer.eos_token_id)

    generated_tokens = outputs[0][inputs.shape[1]:]
    respuesta = tokenizer.decode(generated_tokens, skip_special_tokens=True).strip()

    respuesta = respuesta.replace("<|endoftext|>", "") \
                         .replace("</s>", "") \
                         .replace("<|eot_id|>", "") \
                         .replace("<|start_header_id|>", "") \
                         .replace("assistant\n", "") \
                         .replace("</think>", "") \
                         .strip()

    if respuesta.startswith("❌ Error:") or not ',' in respuesta.split('\n')[0]:
        return respuesta
    else:
        return respuesta
