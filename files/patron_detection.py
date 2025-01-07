import cv2
import numpy as np
from picamera2 import Picamera2
import time

# Funciones de detección y análisis de esquinas
def shi_tomasi_corner_detection(image: np.array, maxCorners: int, qualityLevel: float, minDistance: int, corner_color: tuple, radius: int):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    corners = cv2.goodFeaturesToTrack(gray, maxCorners, qualityLevel, minDistance)
    corners = np.intp(corners)
    corners_list = []
    for corner in corners:
        x, y = corner.ravel()
        corners_list.append([x, y])
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

def configurar_camara():
    picam = Picamera2()
    picam.preview_configuration.main.size = (1280, 720)
    picam.preview_configuration.main.format = "RGB888"
    picam.preview_configuration.align()
    picam.configure("preview")
    picam.start()
    return picam

def procesar_frame(frame, secuencia, index, maxCorners):
    if secuencia == 2:
        # Convertir la imagen a HSV para filtrar el color rojo
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        # Definir el rango de colores rojos en HSV
        lower_red = np.array([0, 120, 70])
        upper_red = np.array([10, 255, 255])
        mask1 = cv2.inRange(hsv, lower_red, upper_red)

        lower_red = np.array([170, 120, 70])
        upper_red = np.array([180, 255, 255])
        mask2 = cv2.inRange(hsv, lower_red, upper_red)

        # Combinamos las dos máscaras
        mask = mask1 | mask2
        frame = cv2.bitwise_and(frame, frame, mask=mask)

    imagen_suavizada = cv2.GaussianBlur(frame, (5, 5), 0)
    procesado, esquinas = shi_tomasi_corner_detection(frame.copy(), maxCorners=10, qualityLevel=0.3, minDistance=20, corner_color=(0, 255, 0), radius=5)
    figura_detectada = clasificar_poligono(esquinas)
    
    return procesado, figura_detectada

def ejecutar_secuencia(secuencia, picam, secuencia1, secuencia2):
    index = 0
    mensaje = ""
    ultimo_tiempo_mensaje = time.time()
    tiempo_mostrar_mensaje = 2  
    correcto = False
    secuencia = 1
    while True:
        frame = picam.capture_array()

        procesado, figura_detectada = procesar_frame(frame, secuencia, index, 10)

        # Mostrar mensaje en la pantalla
        if mensaje and time.time() - ultimo_tiempo_mensaje < tiempo_mostrar_mensaje:
            if correcto:
                cv2.putText(procesado, mensaje, (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            else:
                cv2.putText(procesado, mensaje, (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

        if secuencia == 1:
            if cv2.waitKey(1) & 0xFF == ord('f'):  # Confirmar figura actual
                if figura_detectada == secuencia1[index]:
                    correcto = True
                    mensaje = f"Figura correcta: {figura_detectada}"
                    index += 1
                    if index == len(secuencia1):
                        mensaje = "Secuencia 1 completada, acceda a Secuencia 2"
                        secuencia = 2  # Cambiar a secuencia 2
                        index = 0  # Resetear el índice para la secuencia 2
                        ultimo_tiempo_mensaje = time.time()  # Reiniciar el temporizador de mensaje
                        # No salir del bucle, solo cambiar la secuencia
                    cv2.putText(procesado, mensaje, (10, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                    cv2.imshow("Deteccion de formas", procesado)
                    cv2.waitKey(3000)
                else:
                    correcto = False
                    mensaje = "Figura incorrecta. Reinicia la secuencia."
                    index = 0
                ultimo_tiempo_mensaje = time.time()
            elif cv2.waitKey(1) & 0xFF == ord('q'):
                print("Programa terminado.")
                break

        if secuencia == 2:
            if cv2.waitKey(1) & 0xFF == ord('f'):  # Confirmar figura actual
                if figura_detectada == secuencia2[index]:
                    correcto = True
                    mensaje = f"Figura correcta: {figura_detectada}"
                    index += 1
                    if index == len(secuencia2):
                        mensaje = "Acceso concedido"
                        secuencia = 2
                        cv2.putText(procesado, mensaje, (10, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                        cv2.imshow("Deteccion de formas", procesado)
                        cv2.waitKey(3000)
                        break
                else:
                    correcto = False
                    mensaje = "Figura incorrecta. Reinicia la secuencia."
                    index = 0
                ultimo_tiempo_mensaje = time.time()
            elif cv2.waitKey(1) & 0xFF == ord('q'):
                print("Programa terminado.")
                break

        cv2.imshow("Deteccion de formas", procesado)

    cv2.destroyAllWindows()
    picam.stop()
    picam.close()

def main():
    picam = configurar_camara()
    secuencia1 = ["Cuadrado", "Triangulo", "Estrella"]
    secuencia2 = ["Cuadrado", "Triangulo", "Estrella"]
    ejecutar_secuencia(1, picam, secuencia1, secuencia2)
