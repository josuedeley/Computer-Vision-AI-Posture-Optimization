import cv2
import numpy as np
import math
import onnxruntime as ort
import database
import wahoo_reader
import os

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH = os.path.join(BASE_DIR, "models", "best.onnx")

# --- MAPEO DE KEYPOINTS ---
INDICES_IZQ = {"rodilla": 0, "tobillo": 2, "pie": 4, "cadera": 6, "hombro": 8}
INDICES_DER = {"rodilla": 1, "tobillo": 3, "pie": 5, "cadera": 7, "hombro": 9}

configuracion_usuario = {"femur": 45.0, "tibia": 40.0, "pie": 20.0, "torso": 50.0}

estado_sistema = "CALIBRANDO" 
perfil_actual = "DETECTANDO..."
diccionario_activo = None
alerta_activa = False
mensaje_alerta = "Iniciando sistema..."
ultimo_angulo_rodilla = 0.0

frames_analizados = 0
conf_acumulada_izq = 0.0
conf_acumulada_der = 0.0

# --- CARGA DEL MODELO ONNX PURO (SIN ULTRALYTICS) ---
print("Cargando motor de IA ONNX Runtime...")
try:
    ort_session = ort.InferenceSession("models/best.onnx", providers=['CUDAExecutionProvider', 'CPUExecutionProvider'])
except Exception as e:
    print("Iniciando en CPU...")
    ort_session = ort.InferenceSession("models/best.onnx", providers=['CPUExecutionProvider'])

def calcular_angulo(p1, p2, p3):
    x1, y1 = p1
    x2, y2 = p2
    x3, y3 = p3

    angulo = math.degrees(math.atan2(y3 - y2, x3 - x2) - math.atan2(y1 - y2, x1 - x2))
    if angulo < 0:
        angulo += 360.0

    if angulo > 180.0:
        angulo = 360.0 - angulo

    return int(angulo)

def validar_coherencia(p_cadera, p_rodilla, p_tobillo, config):
    dist_femur = np.linalg.norm(p_cadera - p_rodilla)
    if abs(dist_femur - config["femur"]) > (config["femur"] * 0.3): 
        return False
    return True

def obtener_camara_usb():
    print("Iniciando camara ...")
    cap = cv2.VideoCapture(0, cv2.CAP_V4L2)
    if not cap.isOpened():
        print("No se detecto la camara")
    return cap

def preprocess_frame(img):
    """Prepara el frame 640x480 para el modelo 640x640 de forma forzada."""
    img_resized = cv2.resize(img, (640, 640))
    img_input = img_resized.astype(np.float32) / 255.0
    img_input = np.transpose(img_input, (2, 0, 1)) # HWC a CHW
    return np.expand_dims(img_input, axis=0) # Añadir dimensión de Batch

def postprocess_onnx(out_tensor):
    """Decodifica la salida matemática cruda del modelo YOLOv8-Pose."""
    preds = out_tensor[0].T # Transponer a shape (8400, 35)
    
    # Filtrar solo cajas con más de 50% de confianza de ser una persona (índice 4)
    valid_preds = preds[preds[:, 4] > 0.5]
    if len(valid_preds) == 0: return None
        
    # Optimización: En entorno de laboratorio, solo nos importa la caja con mayor confianza
    best_pred = valid_preds[np.argmax(valid_preds[:, 4])]
    
    # Extraer los 10 keypoints detectados (Índices del 5 al 34)
    num_kpts = (len(best_pred) - 5) // 3
    kpts_raw = best_pred[5:5 + num_kpts * 3]
    kpts = kpts_raw.reshape(num_kpts, 3) # Formato [X, Y, Confianza]
    
    # Restar el borde que agregamos antes para que las coordenadas encajen en el video real
    kpts[:, 1] -= 80.0
    return kpts

def procesar_pose(keypoints):
    hombro_izq = keypoints[hombro_izq_idx]
    hombro_der = keypoints[hombro_der_idx]

    if hombro_izq.x < hombro_der.x:
        perfil = "derecho"
        puntos_a_usar = [rodilla_der, cadera_der, tobillo_der, pie_der, cadera_der]
    else:
       perfil = "izquierdo"
       puntos_a_usar = [rodilla_izq, cadera_izq, tobillo_izq, pie_izq, cadera_izq]


    dibujar_esqueleto(puntos_a_usar)
    calcular_biomecanica(puntos_a_usar)

def generar_frames():
    global estado_sistema, perfil_actual, diccionario_activo, frames_analizados
    global conf_acumulada_izq, conf_acumulada_der, alerta_activa, mensaje_alerta, ultimo_angulo_rodilla

    cap = obtener_camara_usb()
    input_name = ort_session.get_inputs()[0].name

    while True:
        success, frame = cap.read()
        if not success: continue
            
        # 1. Inferencia Pura en ONNX
        img_input = preprocess_frame(frame)
        out = ort_session.run(None, {input_name: img_input})[0]
        kpts = postprocess_onnx(out)
        
        if kpts is not None:
            # 2. FASE DE CALIBRACIÓN
            if estado_sistema == "CALIBRANDO":
                conf_izq = sum([kpts[idx][2] for idx in INDICES_IZQ.values() if idx < len(kpts)])
                conf_der = sum([kpts[idx][2] for idx in INDICES_DER.values() if idx < len(kpts)])
                
                conf_acumulada_izq += conf_izq
                conf_acumulada_der += conf_der
                frames_analizados += 1
                
                cv2.putText(frame, f"CALIBRANDO... {int((frames_analizados/30)*100)}%", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)

                if frames_analizados >= 30:
                    diccionario_activo = INDICES_DER if conf_acumulada_der > conf_acumulada_izq else INDICES_IZQ
                    perfil_actual = "DERECHO" if conf_acumulada_der > conf_acumulada_izq else "IZQUIERDO"
                    estado_sistema = "ANALIZANDO"
                    mensaje_alerta = "SISTEMA LISTO"

            # 3. FASE DE ANÁLISIS CLÍNICO
            elif estado_sistema == "ANALIZANDO":
                try:
                    frames_analizados += 1
                    
                    # Extraer los 5 puntos anatómicos necesarios para el plano sagital completo
                    pt_cadera = kpts[diccionario_activo["cadera"]]
                    pt_rodilla = kpts[diccionario_activo["rodilla"]]
                    pt_tobillo = kpts[diccionario_activo["tobillo"]]
                    pt_pie = kpts[diccionario_activo["pie"]]
                    pt_hombro = kpts[diccionario_activo["hombro"]]
                    
                    # Control de confianza base para la pierna
                    if pt_cadera[2] > 0.4 and pt_rodilla[2] > 0.4 and pt_tobillo[2] > 0.4:
                        p_c = pt_cadera[:2]
                        p_r = pt_rodilla[:2]
                        p_t = pt_tobillo[:2]
                        
                        # 1. Cálculo de Ángulos en Paralelo
                        ang_rodilla = calcular_angulo(p_c, p_r, p_t)
                        ultimo_angulo_rodilla = ang_rodilla
                        
                        ang_cadera = 0.0
                        if pt_hombro[2] > 0.4:
                            ang_cadera = calcular_angulo(pt_hombro[:2], p_c, p_r)
                            
                        ang_tobillo = 0.0
                        if pt_pie[2] > 0.4:
                            ang_tobillo = calcular_angulo(p_r, p_t, pt_pie[:2])

                        # 2. Evaluación de Alertas (Basado en Rodilla)
                        alerta_activa = False
                        if ang_rodilla > 40:
                            mensaje_alerta = "Sillín Bajo (> 40°)"
                            alerta_activa = True
                        elif ang_rodilla < 30:
                            mensaje_alerta = "Sillín Alto (< 30°)"
                            alerta_activa = True
                        else:
                            mensaje_alerta = "RANGO ÓPTIMO"
                        
                        watts_actuales = wahoo_reader.potencia_actual

                        # 3. Guardado síncrono en SQLite
                        if frames_analizados % 6 == 0:
                            estado_int = 1 if alerta_activa else 0
                            database.insertar_metrica(
                                round(ang_rodilla, 2), 
                                round(ang_cadera, 2), 
                                round(ang_tobillo, 2), 
                                estado_int,
                                watts_actuales
                            )
                        
                        # 4. Dibujado de Esqueleto Biomecánico Extendido
                        color = (0, 0, 255) if alerta_activa else (0, 255, 0)
                        cv2.line(frame, tuple(p_c.astype(int)), tuple(p_r.astype(int)), color, 4)
                        cv2.line(frame, tuple(p_r.astype(int)), tuple(p_t.astype(int)), color, 4)
                        
                        if pt_hombro[2] > 0.4:
                            cv2.line(frame, tuple(pt_hombro[:2].astype(int)), tuple(p_c.astype(int)), (255, 255, 0), 3)
                        if pt_pie[2] > 0.4:
                            cv2.line(frame, tuple(p_t.astype(int)), tuple(pt_pie[:2].astype(int)), (255, 255, 0), 3)
                        
                        # Círculos en articulaciones
                        cv2.circle(frame, tuple(p_r.astype(int)), 6, (255, 255, 255), -1)
                        
                        # Texto flotante del ángulo de rodilla principal
                        cv2.putText(frame, f"{int(ang_rodilla)} deg", (int(p_r[0]) + 15, int(p_r[1])), 
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
                                    
                except IndexError:
                    pass

        # 4. Transmitir por Web
        encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), 40]
        ret, buffer = cv2.imencode('.jpg', frame, encode_param)
    
        if not ret:
            continue

        frame_bytes = buffer.tobytes()
        yield (b'--frame\r\n' b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
