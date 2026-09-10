import asyncio
import threading
from bleak import BleakClient

# UUID estándar internacional para potenciómetros de ciclismo
POWER_MEASUREMENT_CHAR_UUID = "00002a63-0000-1000-8000-00805f9b34fb"

# Variable global que la IA leerá en tiempo real
potencia_actual = 0.0
WAHOO_MAC_ADDRESS = "F2:AB:16:3F:22:A9"

def _manejar_datos_potencia(sender, data):
    global potencia_actual
    try:
        # Desempaquetado estándar BLE para el vector de potencia instantánea
        potencia = int.from_bytes(data[2:4], byteorder='little', signed=True)
        potencia_actual = float(potencia)
    except Exception:
        pass

async def _conectar_y_leer():
    print(f"🔵 [BLE] Iniciando conexión directa a la MAC: {WAHOO_MAC_ADDRESS}...")
    try:
        async with BleakClient(WAHOO_MAC_ADDRESS) as client:
            if client.is_connected:
                print("⚡ [BLE] ¡Enlace de hardware establecido! (El LED debe estar azul fijo).")
                
                # ⚠️ TRUCO DE INGENIERÍA: En Linux y hardware antiguo, se necesita un tiempo muerto
                # de estabilización para que el sistema termine de negociar los parámetros MTU.
                print("⏳ Esperando 2.5 segundos para estabilizar el canal de radiofrecuencia...")
                await asyncio.sleep(2.5)
                
                print("🔍 Explorando mapa de servicios internos del Wahoo KICKR v1...")
                services = client.services
                
                # Verificar si la característica estándar existe en este rodillo v1
                char = services.get_characteristic(POWER_MEASUREMENT_CHAR_UUID)
                
                if not char:
                    print("\n⚠️ ALERTA DE COMPATIBILIDAD V1: No se encontró la característica de potencia estándar (0x2A63).")
                    print("📋 Listando características reales encontradas en tu dispositivo para mapear los Watts:")
                    for service in services:
                        print(f"  [Servicio] {service.uuid}")
                        for characteristic in service.characteristics:
                            print(f"    └── [Característica] {characteristic.uuid} | Propiedades: {characteristic.properties}")
                    print("\n💡 Copia lo que imprima la terminal aquí arriba para decirte cuál es el ID correcto de tu rodillo.")
                    return
                
                print(f"✍️ Suscribiéndose de forma segura a la característica de potencia: {POWER_MEASUREMENT_CHAR_UUID}")
                await client.start_notify(POWER_MEASUREMENT_CHAR_UUID, _manejar_datos_potencia)
                print("🚀 ¡SENSOR FUSIÓN ACTIVO! Capturando datos cinéticos en tiempo real.")
                
                # Mantener el hilo vivo capturando datos
                while True:
                    if not client.is_connected:
                        print("❌ [BLE] El rodillo se desconectó del sistema.")
                        break
                    await asyncio.sleep(1)
                    
    except Exception as e:
        print(f"\n❌ [BLE] ERROR DE ENLACE INALÁMBRICO: {e}")
        print("💡 Tip de desarrollo: Si el error persiste, reinicia el Bluetooth de la Jetson con: 'sudo systemctl restart bluetooth'")

def _iniciar_loop_asincrono():
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    loop.run_until_complete(_conectar_y_leer())

def iniciar_lector_potencia():
    """Lanzado desde main.py en un hilo daemon independiente."""
    t = threading.Thread(target=_iniciar_loop_asincrono, daemon=True)
    t.start()