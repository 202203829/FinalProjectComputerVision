import cv2
import numpy as np

def main():
    # Cargar el video
    video_path = 'C:/Users/pablo/OneDrive/Escritorio/iMat/iMat 3º/Visión por ordenador/FinalProject/data/videos/video1.mp4'
    cap = cv2.VideoCapture(video_path)

    # Verificar si el video se abrió correctamente
    if not cap.isOpened():
        print("Error: No se puede abrir el video.")
        exit()

    # Obtener las dimensiones del frame original
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # Definir el tamaño de la ventana de visualización
    new_width = int(frame_width / 2)
    new_height = int(frame_height / 2)

    # Crear la ventana con el tamaño ajustado
    cv2.namedWindow('Frame', cv2.WINDOW_NORMAL)
    cv2.resizeWindow('Frame', new_width, new_height)

    # Mostrar el primer frame
    ret, frame = cap.read()
    if not ret:
        print("Error: No se pudo leer el primer frame.")
        exit()

    cv2.imshow('Frame', frame)

    # Esperar a que el usuario presione la tecla de espacio para avanzar
    while True:
        key = cv2.waitKey(0) & 0xFF
        if key == 32:  # Espacio
            ret, frame = cap.read()
            if not ret:
                break
            frame_resized = cv2.resize(frame, (new_width, new_height))
            cv2.imshow('Frame', frame_resized)
        elif key == ord('s'):  # Selección de objeto
            hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
            lower_red1 = np.array([0, 120, 70])
            upper_red1 = np.array([10, 255, 255])
            lower_red2 = np.array([170, 120, 70])
            upper_red2 = np.array([180, 255, 255])
            mask1 = cv2.inRange(hsv, lower_red1, upper_red1)
            mask2 = cv2.inRange(hsv, lower_red2, upper_red2)
            mask = cv2.bitwise_or(mask1, mask2)
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if contours:
                largest_contour = max(contours, key=cv2.contourArea)
                x, y, w, h = cv2.boundingRect(largest_contour)
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 2)
                cv2.imshow('Object Selected', frame)

                roi = frame[y:y+h, x:x+w]
                roi_hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
                roi_hist = cv2.calcHist([roi_hsv], [0], None, [180], [0, 180])
                cv2.normalize(roi_hist, roi_hist, 0, 255, cv2.NORM_MINMAX)

                track_window = (x, y, w, h)
                term_crit = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 1)

                lost_frames = 0  # Contador de frames consecutivos sin rastreo
                max_lost_frames =5 # Umbral para considerar el objeto perdido

                while True:
                    key = cv2.waitKey(0) & 0xFF
                    if key == 32:
                        ret, frame = cap.read()
                        if not ret:
                            break

                        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
                        back_proj = cv2.calcBackProject([hsv], [0], roi_hist, [0, 180], 1)
                        ret, track_window = cv2.meanShift(back_proj, track_window, term_crit)
                        x, y, w, h = track_window

                        if ret == 0:
                            lost_frames += 1
                        else:
                            lost_frames = 0

                        if lost_frames >= max_lost_frames:
                            cv2.putText(frame, "False - Object Lost", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                            print("Objeto perdido por más de 20 frames.")
                        else:
                            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 2)
                            cv2.putText(frame, "True", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

                        frame_resized = cv2.resize(frame, (new_width, new_height))
                        cv2.imshow('Tracking', frame_resized)

                    elif key == ord('q') or key == 27:
                        break
                break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
