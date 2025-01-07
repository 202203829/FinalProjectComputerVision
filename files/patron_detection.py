import cv2
import numpy as np
from picamera2 import Picamera2
import time

# Funciones de deteccion y análisis de esquinas
def shi_tomasi_corner_detection(image: np.array, maxCorners: int, qualityLevel: float, minDistance: int, corner_color: tuple, radius: int):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    corners = cv2.goodFeaturesToTrack(gray, maxCorners, qualityLevel, minDistance)
    corners = np.intp(corners)
    corners_list = []
    for corner in corners:
        x, y = corner.ravel()
        corners_list.append([x, y])
        cv2.circle(image, (x, y), radius, corner_color, -1)
    return image, corners_list

def ordenar_puntos(puntos):
    puntos = np.array(puntos)
    origen = puntos[np.argmin(puntos[:, 0] + puntos[:, 1])]
    def calcular_angulo(p):
        return np.arctan2(p[1] - origen[1], p[0] - origen[0])
    return sorted(puntos, key=calcular_angulo)

def calcular_angulos(puntos_ordenados):
    angulos = []
    n = len(puntos_ordenados)
    for i in range(n):
        p1 = puntos_ordenados[i] - puntos_ordenados[i - 1]
        p2 = puntos_ordenados[(i + 1) % n] - puntos_ordenados[i]
        cos_theta = np.dot(p1, p2) / (np.linalg.norm(p1) * np.linalg.norm(p2))
        angulo = np.arccos(np.clip(cos_theta, -1, 1))
        angulos.append(np.degrees(angulo))
    return angulos

def clasificar_poligono(puntos):
    puntos_ordenados = ordenar_puntos(puntos)
    n = len(puntos_ordenados)
    if n == 3:
        return "Triangulo"
    elif n == 4:
        angulos = calcular_angulos(puntos_ordenados)
        if all(65 < ang < 105 for ang in angulos):
            return "Cuadrado"
        else:
            return "Otro"
    elif n == 10:
        return "Estrella"
    else:
        return "Otro"

# Configuracion de la cámara
picam = Picamera2()
picam.preview_configuration.main.size = (1280, 720)
picam.preview_configuration.main.format = "RGB888"
picam.preview_configuration.align()
picam.configure("preview")
picam.start()

# Variables
secuencia1 = [ "Cuadrado","Triangulo", "Estrella"]
secuencia2 = [ "Cuadrado","Triangulo", "Estrella"]

index = 0
mensaje = ""
ultimo_tiempo_mensaje = time.time()
tiempo_mostrar_mensaje = 2  
correcto = False


while True:
    frame = picam.capture_array() 
    imagen_suavizada = cv2.GaussianBlur(frame, (5, 5), 0)
    procesado, esquinas = shi_tomasi_corner_detection(frame.copy(), maxCorners=10, qualityLevel=0.3, minDistance=20)
    figura_detectada = clasificar_poligono(esquinas)

    # Mostrar mensaje en la pantalla
    if mensaje and time.time() - ultimo_tiempo_mensaje < tiempo_mostrar_mensaje:
        if correcto:
            cv2.putText(procesado, mensaje, (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        else:
            cv2.putText(procesado, mensaje, (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    secuencia = 1
    if cv2.waitKey(1) & 0xFF == ord('f'):  # Confirmar figura actual
        if secuencia == 1
        if figura_detectada == secuencia1[index]:
            correcto = True
            mensaje = f"Figura correcta: {figura_detectada}"
            index += 1
            if index == len(secuencia1):
                mensaje = "Secuencia 1 completada, acceda a Secuencia 2"
                cv2.putText(procesado, mensaje, (10, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                cv2.imshow("Deteccion de formas", procesado)
                cv2.waitKey(3000)
                break
        else:
            correcto = False
            mensaje = "Figura incorrecta. Reinicia la secuencia."
            index = 0
        ultimo_tiempo_mensaje = time.time()
    elif cv2.waitKey(1) & 0xFF == ord('q'):  # Salir del programa
        print("Programa terminado.")
        break

    cv2.imshow("Deteccion de formas", procesado)

cv2.destroyAllWindows()
picam.stop()