# generador_datos_sinteticos

## Instrucciones

1- Clona el repositorio
2- Entra y, desde su raíz, ejecuta `pip install -r requirements.txt` desde cmd (te interesa tener un entorno virtual creado primero)
3- Crea un `.env` y escribe lo siguiente:
* API_KEY= (tienes que crearte una en huggingFace, en la foto de tu perfil > access tokens, es posible que tengas que pedir permiso para usar el modelo).
* MODEL="Qwen/Qwen3-4B" (Por si acaso ve a models en HuggingFace y revisa si hay que pedir permiso).
NOTA: no dejes espacioes en las variables de entorno, el = debe ir pegado a la palabra anterior y a la siguiente.
4- Ejecuta en CMD desde la raiz del proyecto: `python app.py` (tendrás que tener instalado python en tu equipo)


