import gradio as gr
from ia import escribe

def escribe_gradio(descripcion, archivo):
    resultado = escribe(archivo_csv=archivo, descripcion=descripcion)
    return resultado

app = gr.Interface(
    fn=escribe_gradio,
    inputs=[
        gr.Textbox(label="Descripción (opcional)", lines=3, placeholder="Ej: Clientes de una tienda online en Europa"),
        gr.File(label="Archivo CSV (opcional)")
    ],
    outputs=[
        gr.Textbox(label="CSV generado o mensaje", lines=20)
    ],
    title="Generador de Datos Sintéticos",
    description="Sube un archivo CSV con encabezados o escribe una descripción. La IA generará 20 filas de datos sintéticos realistas."
)

app.launch(share=True)
