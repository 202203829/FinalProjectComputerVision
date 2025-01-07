import cv2
import numpy as np
from picamera2 import Picamera2

# Función para configurar la cámara
def configure_camera():
    picam = Picamera2()
    picam.preview_configuration.main.size = (1280, 720)
    picam.preview_configuration.main.format = "RGB888"
    picam.preview_configuration.align()
    picam.configure("preview")
    picam.start()
    return picam

# Función para seleccionar los objetos a seguir
def select_objects(frame):
    cv2.imshow("Select Objects", frame)
    bboxes = cv2.selectROIs("Select Objects", frame, fromCenter=False, showCrosshair=True)
    cv2.destroyWindow("Select Objects")
    return bboxes

# Función para crear los trackers
def create_trackers(frame, bboxes):
    trackers = []
    colors = []  # Para almacenar los colores de cada objeto
    
    for bbox in bboxes:
        # Extraer las coordenadas del rectángulo
        x, y, w, h = [int(v) for v in bbox]

        # Calcular las coordenadas del punto central
        center_x, center_y = x + w // 2, y + h // 2

        # Obtener el color del píxel central (en formato BGR)
        central_pixel = frame[center_y, center_x]
        
        # Determinar el color de la caja
        blue, green, red = central_pixel
        color = (0, 0, 255) if red > green and red > blue else (0, 255, 0)  
        colors.append(color)

        # Crear un tracker para este objeto
        tracker = cv2.TrackerCSRT_create()
        tracker.init(frame, tuple(bbox))  # Inicializar el tracker para cada objeto
        trackers.append(tracker)  # Añadir el tracker a la lista

    return trackers, colors

# Función para dibujar los bordes del cuadrado en el centro
def draw_center_square(frame, square_size=200):
    height, width, _ = frame.shape
    square_x1 = (width - square_size) // 2
    square_y1 = (height - square_size) // 2
    square_x2 = square_x1 + square_size
    square_y2 = square_y1 + square_size
    cv2.rectangle(frame, (square_x1, square_y1), (square_x2, square_y2), (0, 0, 0), 2)  # Dibujar solo los bordes
    return square_x1, square_y1, square_x2, square_y2

# Función para realizar el seguimiento y detectar objetos dentro del cuadrado
def track_objects(frame, trackers, colors, square_x1, square_y1, square_x2, square_y2):
    alarm_triggered = False
    for i, tracker in enumerate(trackers):
        success, bbox = tracker.update(frame)
        if success:
            # Si el seguimiento es exitoso, dibujar un rectángulo con el color determinado
            x, y, w, h = [int(v) for v in bbox]
            color = colors[i]
            cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)

            # Calcular el centro del objeto
            object_center_x = x + w // 2
            object_center_y = y + h // 2

            # Verificar si el objeto rojo está dentro del cuadrado
            if color == (0, 0, 255):  # Solo verificar si el objeto es rojo
                if square_x1 < object_center_x < square_x2 and square_y1 < object_center_y < square_y2:
                    alarm_triggered = True
                    break  # Si un objeto rojo está dentro del cuadrado, ya no necesitamos verificar más

    return alarm_triggered

# Función principal que orquesta la ejecución
def main():
    picam = configure_camera()

    # Capturar el primer frame
    frame = picam.capture_array()

    # Seleccionar objetos a seguir
    bboxes = select_objects(frame)

    # Crear trackers y asignar colores
    trackers, colors = create_trackers(frame, bboxes)

    # Dibujar los bordes del cuadrado negro en el centro
    square_x1, square_y1, square_x2, square_y2 = draw_center_square(frame)

    # Mientras la cámara sigue funcionando, actualizamos los trackers
    while True:
        # Captura de frame desde Picamera2
        frame = picam.capture_array()

        # Dibujar los bordes del cuadrado negro en el centro de la pantalla
        cv2.rectangle(frame, (square_x1, square_y1), (square_x2, square_y2), (0, 0, 0), 2)

        # Actualizar los trackers y verificar si algún objeto rojo está dentro del cuadrado
        alarm_triggered = track_objects(frame, trackers, colors, square_x1, square_y1, square_x2, square_y2)

        # Mostrar el mensaje según si algún objeto rojo está dentro del cuadrado
        if alarm_triggered:
            cv2.putText(frame, "ALARM", (100, 80), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2)
        else:
            cv2.putText(frame, "Everything OK", (100, 80), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)

        # Mostrar el frame con los objetos seguidos
        cv2.imshow("Multi Object Tracking", frame)

        # Salir si se presiona la tecla 'q'
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Liberar recursos
    picam.close()
    cv2.destroyAllWindows()
