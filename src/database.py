import sqlite3
import datetime
import os

# ==============================================================================
# 1. CONFIGURACIÓN DE RUTAS ABSOLUTAS DENTRO DE LA CARPETA 'src'
# ==============================================================================
# BASE_DIR se sitúa exactamente en la ruta física de la carpeta 'src/'
BASE_DIR = os.path.dirname(os.path.abspath(__file__)) 

# CARPETA_DATOS define la ruta para 'src/datos_guardados'
CARPETA_DATOS = os.path.join(BASE_DIR, "datos_guardados")

# DB_PATH define la ruta final para 'src/datos_guardados/entrenamientos.db'
DB_PATH = os.path.join(CARPETA_DATOS, "entrenamientos.db")

def get_connection():
    """Establece la conexión con el motor SQLite.
    Fuerza la creación de la subcarpeta 'datos_guardados' si no existiera físicamente.
    check_same_thread=False permite la concurrencia asíncrona desde FastAPI."""
    if not os.path.exists(CARPETA_DATOS):
        os.makedirs(CARPETA_DATOS)
        print(f"📁 Creada subcarpeta de persistencia en: {CARPETA_DATOS}")
        
    return sqlite3.connect(DB_PATH, check_same_thread=False)

# ==============================================================================
# 2. OPERACIONES DDL (ESTRUCTURA DE TABLAS)
# ==============================================================================
def crear_tablas():
    """Inicializa la base de datos recreando la estructura desde cero.
    El uso de DROP TABLE garantiza la depuración de esquemas antiguos incompletos
    durante el arranque del servidor."""
    conn = get_connection()
    cursor = conn.cursor()
    
    # ⚠️ Limpieza de desarrollo: Elimina la tabla vieja si tenía columnas incompletas
    cursor.execute('DROP TABLE IF EXISTS metricas')
    
    # Creación del esquema definitivo con soporte biomecánico completo
    cursor.execute('''
        CREATE TABLE metricas (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp DATETIME,
            angulo_rodilla REAL,
            angulo_cadera REAL,
            angulo_tobillo REAL,
            alerta INTEGER,
            potencia REAL
        )
    ''')
    conn.commit()
    conn.close()
    print(f"✅ Base de datos SQLite inicializada correctamente en: {DB_PATH}")

# ==============================================================================
# 3. OPERACIONES DML (MANIPULACIÓN DE DATOS)
# ==============================================================================
def insertar_metrica(angulo_rodilla, angulo_cadera, angulo_tobillo, alerta, potencia):
    """Inserta las métricas cinemáticas calculadas por pose_analyzer.py.
    Registra marcas de tiempo con precisión de segundos."""
    conn = get_connection()
    cursor = conn.cursor()
    
    # Capturamos la hora local exacta del análisis del frame
    ahora = datetime.datetime.now().strftime("%H:%M:%S")
    
    cursor.execute('''
        INSERT INTO metricas (timestamp, angulo_rodilla, angulo_cadera, angulo_tobillo, alerta, potencia)
        VALUES (?, ?, ?, ?, ?, ?)
    ''', (ahora, angulo_rodilla, angulo_cadera, angulo_tobillo, alerta, potencia))
    
    conn.commit()
    conn.close()

def init_db():
    crear_tablas()

def obtener_metricas(limite=30):
    """Extrae las muestras más recientes de forma ordenada.
    Invierte la lista devuelta por SQLite para que el frontend dibuje las curvas
    del gráfico en orden cronológico correcto (de izquierda a derecha)."""
    conn = get_connection()
    cursor = conn.cursor()
    
    cursor.execute('''
        SELECT timestamp, angulo_rodilla, alerta 
        FROM metricas 
        ORDER BY id DESC 
        LIMIT ?
    ''', (limite,))
    
    filas = cursor.fetchall()
    conn.close()
    
    # Invertir el lote para mantener la coherencia del flujo de tiempo en el eje X
    datos_cronologicos = reversed(filas)
    
    # Formateo estructurado para la serialización JSON que consume la web
    resultado = {
        "etiquetas_tiempo": [],
        "angulos_rodilla": [],
        "alertas": []
    }
    
    for fila in datos_cronologicos:
        resultado["etiquetas_tiempo"].append(fila[0])
        resultado["angulos_rodilla"].append(fila[1])
        resultado["alertas"].append(fila[2])
        
    return resultado

def limpiar_historial():
    """Vactía la tabla 'metricas'. Útil para reinicios limpios o nuevas sesiones."""
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute('DELETE FROM metricas')
    conn.commit()
    conn.close()
    print("🗑️ Historial de métricas de la sesión purgado.")