import sqlite3
import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import io
import base64

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CARPETA_DATOS = os.path.join(BASE_DIR, "datos_guardados")
DB_PATH = os.path.join(CARPETA_DATOS, "entrenamientos.db")

def analizar_entrenamiento():
    if not os.path.exists(DB_PATH):
        return {"error": "No se encontraron registros de entrenamiento."}

    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("SELECT timestamp, angulo_rodilla, angulo_cadera, angulo_tobillo, alerta, potencia FROM metricas ORDER BY id ASC")
    datos = cursor.fetchall()
    conn.close()

    if len(datos) == 0:
        return {"error": "La base de datos está vacía."}

    tiempos = [f[0] for f in datos]
    rodillas = [f[1] for f in datos]
    caderas = [f[2] for f in datos]
    tobillos = [f[3] for f in datos]
    alertas = [f[4] for f in datos]
    potencias = [f[5] for f in datos]
    
    total_samples = len(rodillas)
    
    # --- MODELADO CINÉTICO DE POTENCIA PERDIDA ---
    watts_optimos = [potencias[i] for i in range(total_samples) if alertas[i] == 0]
    watts_alerta = [potencias[i] for i in range(total_samples) if alertas[i] == 1]
    
    prom_watts_optimos = np.mean(watts_optimos) if watts_optimos else 0.0
    prom_watts_alerta = np.mean(watts_alerta) if watts_alerta else 0.0
    watts_perdidos = max(0.0, prom_watts_optimos - prom_watts_alerta)

    # --- REGLAS EXPERTAS DE FEEDBACK MULTI-ARTICULAR ---
    frames_optimos = sum(1 for r in rodillas if 30 <= r <= 40)
    pct_optimo = (frames_optimos / total_samples) * 100
    
    if pct_optimo >= 85:
        fb_rodilla = "✅ Técnica de rodilla excelente. Sillín configurado a la altura correcta."
    elif sum(1 for r in rodillas if r < 30) > sum(1 for r in rodillas if r > 40):
        fb_rodilla = "⚠️ Rodilla hiper-extendida. Sobrecarga en la cadena posterior. 👉 RECOMENDACIÓN: Baja el sillín 3-5 mm."
    else:
        fb_rodilla = "⚠️ Exceso de flexión patelar. Alta fuerza de compresión en rótula. 👉 RECOMENDACIÓN: Sube el sillín 3-5 mm."

    cad_validas = [f for f in caderas if f > 0]
    prom_cadera = np.mean(cad_validas) if cad_validas else 0.0

    if prom_cadera < 60:
        fb_cadera = "⚠️ Inclinación del torso excesiva. Puede causar fatiga lumbar." 
    else: 
        fb_cadera = "✅ Angulación lumbo-pélvica dentro de rangos ergonómicos estables."

    tobillos_validos = [t for t in tobillos if t > 0]
    rango_tobillo = np.max(tobillos_validos) - np.min(tobillos_validos) if tobillos_validos else 0
    fb_tobillo = "⚠️ Inestabilidad en el tobillo (taloneo). Compensación de altura con el pie." if rango_tobillo > 25 else "✅ Rango dinámico del tobillo controlado. Transmisión limpia de fuerza."

    # Inyección del cálculo cinético de potencia al feedback de la IA
    fb_potencia = f"📊 <strong>Análisis de Eficiencia Energética:</strong> Al romper la postura ergonómica (Alerta Roja), tu potencia promedio cayó de {prom_watts_optimos:.1f} W a {prom_watts_alerta:.1f} W. La pérdida de alineación geométrica muscular te hace <strong>disipar aproximadamente {watts_perdidos:.1f} Watts</strong>." if watts_perdidos > 0 else "✅ <strong>Análisis de Eficiencia Energética:</strong> No se registran caídas mecánicas significativas en la transferencia de fuerza."

    feedback_unificado = f"• {fb_rodilla}<br><br>• {fb_cadera}<br><br>• {fb_tobillo}<br><br>• {fb_potencia}"

    # --- GENERACIÓN DEL LIENZO DE CUATRO GRÁFICAS (4-SUBPLOTS STACKED) ---
    fig, (ax1, ax2, ax3, ax4) = plt.subplots(4, 1, figsize=(11, 12), sharex=True)
    
    # Subplot 1: Rodilla
    ax1.plot(tiempos, rodillas, color='#2c3e50', linewidth=2, label='Rodilla')
    ax1.axhspan(30, 40, color='#27ae60', alpha=0.25, label='Rango Saludable')
    ax1.set_title("Kinematic-Kinetic Fusion of Pedalling", fontweight='bold', fontsize=14)
    ax1.set_ylabel("Knee (°)")
    ax1.legend(loc='upper right')
    ax1.grid(True, alpha=0.3)

    # Subplot 2: Cadera
    ax2.plot(tiempos, caderas, color='#2980b9', linewidth=2, label='Cadera')
    ax2.set_ylabel("Hip (°)")
    ax2.legend(loc='upper right')
    ax2.grid(True, alpha=0.3)

    # Subplot 3: Tobillo
    ax3.plot(tiempos, tobillos, color='#8e44ad', linewidth=2, label='Tobillo')
    ax3.set_ylabel("Ankle (°)")
    ax3.legend(loc='upper right')
    ax3.grid(True, alpha=0.3)

    # Subplot 4: Potencia (KICKR Watts)
    ax4.plot(tiempos, potencias, color='#e67e22', linewidth=2, label='Potencia (KICKR)')
    # Colorear el trasfondo del gráfico de potencia según el estado de la postura en cada instante
    for i in range(1, total_samples):
        color_zona = '#c0392b' if alertas[i] == 1 else '#27ae60'
        ax4.axvspan(tiempos[i-1], tiempos[i], color=color_zona, alpha=0.15)
        
    ax4.set_ylabel("Power (Watts)")
    ax4.set_xlabel("Session timeline")
    ax4.legend(loc='upper right')
    ax4.grid(True, alpha=0.3)

    # Ajustar etiquetas del eje X limpiamente para evitar solapamiento en versiones antiguas
    plt.xticks(np.arange(0, len(tiempos), max(1, len(tiempos)//8)), rotation=45)
    plt.tight_layout()

    # Codificación de la imagen en buffer RAM a Base64 string
    buf = io.BytesIO()
    plt.savefig(buf, format='png', dpi=100)
    buf.seek(0)
    img_base64 = base64.b64encode(buf.read()).decode('utf-8')
    plt.close()

    return {
        "error": None,
        "total": total_samples,
        "promedio": round(np.mean(rodillas), 1),
        "optimo": round(pct_optimo, 1),
        "feedback": feedback_unificado,
        "grafica": f"data:image/png;base64,{img_base64}"
    }
