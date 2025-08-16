import glob
import gradio as gr
import subprocess
import os
import shutil

# Directorios de trabajo
OUTPUT_DIR = "output_faces"
if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)

def process_video(video_path, style, video_step, det_min_score, det_min_size, hash_thr):
    """
    Función que procesa el video y ejecuta la herramienta videotofaces.
    """

    #test
    print(f"Argumentos recibidos:")
    print(f"  video_path: {video_path}")
    print(f"  style: {style}")
    print(f"  video_step: {video_step}")
    print(f"  det_min_score: {det_min_score}")
    print(f"  det_min_size: {det_min_size}")
    print(f"  hash_thr: {hash_thr}")
    #end test

    # Define la ruta de salida para los rostros.
    # Usaremos una subcarpeta dentro del directorio de salida.
    face_output_dir = os.path.join(OUTPUT_DIR, "extracted_faces")
    if os.path.exists(face_output_dir):
        shutil.rmtree(face_output_dir)
    os.makedirs(face_output_dir)

    # Convertir los parámetros a tipos de datos correctos para el comando
    try:
        det_min_score = float(det_min_score)
        det_min_size = int(det_min_size)
        video_step = float(video_step)
        hash_thr = int(hash_thr)
    except (ValueError, TypeError):
        return "Error: Los parámetros deben ser números válidos."

    face_files = glob.glob(os.path.join(face_output_dir, "faces", "*.jpg"))
    n_samples = len(face_files)

    if n_samples > 2:
        max_clusters = n_samples - 1
        clusters_range = f"2-{max_clusters}"
    else:
        clusters_range = "2-3"  # evita error si hay muy pocas imágenes

    print(f"  clusters_range: {clusters_range}") #test

    command = [
        "python3",
        "-m",
        "videotofaces",
        "-i", video_path,
        "-o", face_output_dir,
        "-s", style,
        "--det-min-score", str(det_min_score),
        "--det-min-size", str(det_min_size),
        "--video-step", str(video_step),
        "--hash-thr", str(hash_thr),
        "--clusters", clusters_range
    ]

    # Ejecuta el comando en el sistema.
    try:
        subprocess.run(command, check=True)
    except subprocess.CalledProcessError as e:
        return f"Error al procesar el video: {e}"

    # Comprime la carpeta de salida para la descarga.
    zip_path = os.path.join(OUTPUT_DIR, "rostros_extraidos.zip")
    shutil.make_archive(zip_path[:-4], "zip", face_output_dir)

    return zip_path

css = """
.prevista_video video {
    width: 100% !important;
    height: auto !important;
    max-height: 500px;
    object-fit: contain;
    border-radius: 8px;
    background: #000;
}

.box {
    border: 1px solid #ddd;
    border-radius: 8px;
    padding: 10px;
    margin-bottom: 10px;
}
"""

# Creación de la interfaz de Gradio.
with gr.Blocks(css=css) as demo:
    with gr.Row():
        with gr.Column(scale=3):
            gr.Markdown("# Extractor de Rostros de Videos")
        with gr.Column(scale=7):
            gr.Markdown("Sube un video para extraer y agrupar los rostros de alta calidad.")
    
    with gr.Row():
        with gr.Column(scale=1):
            with gr.Group():
                gr.Markdown("### Video")
                video_input = gr.Video(label="Selecciona un video", elem_classes=["prevista_video"])
        
        with gr.Column(scale=1):
            with gr.Group():
                gr.Markdown("### Opciones de Procesado")
                with gr.Row():
                    style = gr.Radio(["live", "anime"],
                                     label="Estilo [style]",
                                     value="live",
                                     info="Selecciona el tipo de rostros que la herramienta debe detectar. 'live' para rostros de personas reales y 'anime' para rostros de dibujos animados o manga.")
                with gr.Row():
                    video_step = gr.Slider(minimum=0.0,
                                           maximum=10.0,
                                           step=0.1,
                                           value=0.5,
                                           label="Intervalo de Captura (seg) [video_step]",
                                           info="Define la frecuencia con la que se extraen fotogramas del video para detectar rostros. Un valor muy bajo incrementa la precisión pero también el tiempo de procesamiento.")
                with gr.Row():
                    det_min_score = gr.Slider(minimum=0.2,
                                              maximum=1.0,
                                              step=0.05,
                                              value=0.8,
                                              label="Puntaje Mínimo de Calidad [det_min_score]",
                                              info="Define el umbral de confianza para detectar rostros. Solo se procesarán los rostros con un puntaje igual o superior a este valor. Un valor más alto significa mayor certeza pero menos resultados.")
                with gr.Row():
                    det_min_size = gr.Slider(minimum=25,
                                             maximum=2000,
                                             step=25,
                                             value=50,
                                             label="Tamaño Mínimo Rostro (px) [det_min_size]",
                                             info="Define el tamaño mínimo en px que debe tener un rostro para ser detectado. Un valor muy bajo filtra rostros pequeños pero de mala calidad, un valor muy alto exigirá solo imagenes con muy alta resolución.")
                with gr.Row():
                    hash_thr = gr.Slider(minimum=-1,
                                         maximum=20,
                                         step=1,
                                         value=8,
                                         label="Umbral de Duplicados (Hash) [hash_thr]",
                                         info="Define la tolerancia para considerar dos rostros como duplicados. Un valor de -1 deshabilita el filtro. Un valor más alto permite más variación entre rostros similares.")
        
        with gr.Column(scale=1):
            with gr.Group():
                gr.Markdown("### Descarga de Resultados")
                output_file = gr.File(label="Descargar Rostros Agrupados")
                
    with gr.Group():
        gr.Markdown("### Salida de la Aplicación")
        output_log = gr.Textbox(label="Mensajes de la Consola", lines=5)

    process_button = gr.Button("Procesar Video")

    # Lógica de la interfaz
    process_button.click(
        fn=process_video,
        inputs=[
            video_input,
            style,
            video_step,
            det_min_score,
            det_min_size,
            hash_thr
        ],
        outputs=[output_file]
    )

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860)