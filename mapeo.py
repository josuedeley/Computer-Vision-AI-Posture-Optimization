import cv2
from ultralytics import YOLO

def mapear_keypoints():
    # 1. Cargamos el modelo local
    model = YOLO("models/best.pt")

    # 2. INTENTA USAR UNA FOTO LO MÁS PARECIDA A TU DATASET DE ENTRENAMIENTO
    img_path = "data/yo.jpg"  # Cambia esto por una imagen tuya en la bici de perfil
    
    # 3. Inferencia forzada: conf=0.05 obliga al modelo a mostrar detecciones de baja confianza
    results = model(img_path, conf=0.001)

    for r in results:
        img_anotada = r.orig_img.copy()
        
        # Validar si al menos detectó la caja (el cuerpo) y si tiene keypoints
        if r.keypoints is not None and len(r.keypoints) > 0:
            kpts = r.keypoints.xy[0].cpu().numpy()
            
            for i, (x, y) in enumerate(kpts):
                if x > 0 and y > 0: 
                    centro = (int(x), int(y))
                    cv2.circle(img_anotada, centro, 6, (0, 0, 255), -1)
                    cv2.putText(img_anotada, f"Idx: {i}", (int(x) + 10, int(y) - 10), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
                    
        else:
            print("El modelo detectó algo, pero no encontró keypoints (puntos esqueléticos).")
                    
        cv2.imshow("Ingenieria Inversa - Indices YOLO", img_anotada)
        print("Presiona cualquier tecla en la ventana de la imagen para cerrarla.")
        cv2.waitKey(0)
        cv2.destroyAllWindows()

if __name__ == "__main__":
    mapear_keypoints()