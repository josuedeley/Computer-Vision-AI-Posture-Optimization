import cv2
import numpy as np
import math
from ultralytics import YOLO

def calcular_angulo(p1, p2, p3):
    """Calcula el ángulo en el punto p2 dadas las coordenadas de p1, p2 y p3."""
    x1, y1 = p1
    x2, y2 = p2
    x3, y3 = p3

    # Calcular el ángulo usando arcotangente
    angulo = math.degrees(math.atan2(y3 - y2, x3 - x2) - math.atan2(y1 - y2, x1 - x2))
    
    # Asegurar que el ángulo sea positivo
    angulo = abs(angulo)
    if angulo > 180:
        angulo = 360 - angulo
        
    return angulo

def main():
    # 1. Cargar tu modelo entrenado personalizado
    model = YOLO("models/best.onnx", task="pose")

    # 2. Iniciar la captura de la WebCam
    # El número '0' suele ser la cámara web integrada. 
    # Si conectas una cámara USB externa, intenta cambiarlo a '1' o '2' si el 0 no la detecta.
    cap = cv2.VideoCapture(0)

    # Verificar si la cámara se abrió correctamente
    if not cap.isOpened():
        print("Error: No se pudo abrir la cámara.")
        return

    print("Iniciando análisis en tiempo real... Presiona la tecla 'q' para salir.")

    # 3. Bucle infinito para procesar el video frame por frame
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error al capturar el frame de la cámara.")
            break

        # Ejecutar la predicción en el frame actual (verbose=False oculta el spam en la terminal)
        results = model.predict(frame, conf=0.5, verbose=False)

        for r in results:
            # Esta línea mágica dibuja el esqueleto que entrenaste automáticamente
            img_anotada = r.plot() 
            
            if r.keypoints is not None and len(r.keypoints) > 0:
                kpts = r.keypoints.xy[0].cpu().numpy()

                # Asegurarnos de que el array tenga la longitud esperada (por si no detecta a nadie)
                if len(kpts) >= 7: 
                    # Coordenadas lado Izquierdo (ajusta los índices si tu Roboflow fue diferente)
                    # 0: Ankle, 1: Knee, 2: Hip, 6: Shoulder
                    p_tobillo_izq = kpts[0]
                    p_rodilla_izq = kpts[1]
                    p_cadera_izq = kpts[2]
                    p_hombro_izq = kpts[6]

                    # Calcular y mostrar Ángulo de la Rodilla Izquierda
                    if all(p_cadera_izq) and all(p_rodilla_izq) and all(p_tobillo_izq):
                        angulo_rodilla = calcular_angulo(p_cadera_izq, p_rodilla_izq, p_tobillo_izq)
                        cv2.putText(img_anotada, f"Rodilla: {int(angulo_rodilla)} deg", 
                                    (10, 50), # Posición fija en la esquina superior izquierda
                                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 3)

                    # Calcular y mostrar Ángulo de la Cadera Izquierda (Torso)
                    if all(p_hombro_izq) and all(p_cadera_izq) and all(p_rodilla_izq):
                        angulo_cadera = calcular_angulo(p_hombro_izq, p_cadera_izq, p_rodilla_izq)
                        cv2.putText(img_anotada, f"Cadera: {int(angulo_cadera)} deg", 
                                    (10, 100), # Posición fija debajo de la rodilla
                                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 3)

        # 4. Mostrar el frame con los dibujos en una ventana
        cv2.imshow("Analisis Biomecanico - Tiempo Real", img_anotada)

        # 5. Condición de salida: Si el usuario presiona la tecla 'q', se rompe el bucle
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Limpiar todo al cerrar
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()