from fastapi import FastAPI, Request
from fastapi.responses import StreamingResponse, HTMLResponse
from fastapi.templating import Jinja2Templates
import pose_analyzer
import database
import generar_reporte
import wahoo_reader
import os

app = FastAPI(title="Sistema Biomecánico de Pedaleo")

# Inicializar Base de Datos (Asegúrate de que este archivo exista y esté configurado)
database.crear_tablas()
wahoo_reader.iniciar_lector_potencia()

# Configurar templates
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
RUTA_TEMPLATES = os.path.join(BASE_DIR, "templates")
templates = Jinja2Templates(directory=RUTA_TEMPLATES)

# Configuración global inicial (valores por defecto)
configuracion_usuario = {"femur": 40.0, "tibia": 37.0, "pie": 20.0, "torso": 55.0}

@app.post("/api/config")
async def save_config(data: dict):
    global configuracion_usuario
    # 1. Actualizamos la configuración local en el servidor
    configuracion_usuario.update(data)
    
    # 2. ¡CRÍTICO! Sincronizamos las medidas en tiempo real con el pose_analyzer.
    # Esto inyecta las medidas anatómicas directamente al modelo de visión
    # para que la "Ley de los Cosenos" valide las distancias correctas.
    pose_analyzer.configuracion_usuario.update(configuracion_usuario)
    
    return {"status": "Configuración actualizada y sincronizada con el VLM"}

@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    # Renderiza la interfaz del Dashboard
    return templates.TemplateResponse("index.html", {"request": request})

@app.get("/video_feed")
async def video_feed():
    """Endpoint único para el streaming del esqueleto."""
    # Como ya actualizamos la variable global directamente dentro de pose_analyzer 
    # desde /api/config, la función generar_frames() tomará los datos automáticamente 
    # sin necesidad de pasarle argumentos.
    return StreamingResponse(pose_analyzer.generar_frames(), 
                             media_type="multipart/x-mixed-replace; boundary=frame")

@app.get("/api/estado")
async def estado_actual():
    """API consultada por el frontend (JS) cada 200ms para actualizar alertas visuales."""
    # Leemos las variables globales que el pose_analyzer actualiza frame a frame
    return {
        "alerta": getattr(pose_analyzer, 'alerta_activa', False),
        "mensaje": getattr(pose_analyzer, 'mensaje_alerta', "Iniciando sistema..."),
        "perfil": getattr(pose_analyzer, 'perfil_actual', "N/A"),
        "angulo_rodilla": round(getattr(pose_analyzer, 'ultimo_angulo_rodilla', 0.0), 1)
    }

@app.get("/api/historico")
async def obtener_historico():
    """Retorna las métricas guardadas para graficar el rendimiento."""
    return database.obtener_ultimas_metricas()

@app.get("/api/detener")
async def detener_y_reportar():
    """Detiene el analisis y devuelve el reporte final."""
    pose_analyzer.estado_sistema = "Dentenido"
    reporte = generar_reporte.analizar_entrenamiento()

    return reporte
