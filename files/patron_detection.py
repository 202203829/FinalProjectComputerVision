import cv2
import numpy as np
from typing import List
from utils import non_max_suppression  # Asegúrate de que utils esté en tu proyecto

def show_image(img: np.array, img_name: str = "Image"):
    cv2.imshow(img_name, img) 
    cv2.waitKey(0) 
    cv2.destroyAllWindows()

def gaussian_blur(img: np.array, sigma: float, filter_shape: List | None = None, verbose: bool = False) -> np.array:
    if filter_shape is None:
        filter_size = int(8 * sigma + 1)
        filter_shape = [filter_size, filter_size]

    # Crear un filtro Gaussiano
    ax = np.linspace(-(filter_shape[0] // 2), filter_shape[0] // 2, filter_shape[0])
    gauss_filter = np.exp(-0.5 * (ax**2 + ax[:, None]**2) / sigma**2)
    gauss_filter /= np.sum(gauss_filter)  # Normaliza el filtro

    # Aplicar el filtro Gaussiano a la imagen
    gb_img = cv2.filter2D(img, -1, gauss_filter)
    
    if verbose:
        show_image(img=gb_img, img_name=f"Gaussian Blur: Sigma = {sigma}")
    
    return gauss_filter, gb_img.astype(np.uint8)

def sobel_edge_detector(img: np.array, filter: np.array, gauss_sigma: float, gauss_filter_shape: List | None = None, verbose: bool = False) -> np.array:
    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, blurred = gaussian_blur(gray_img, gauss_sigma, gauss_filter_shape, verbose)
    
    blurred = blurred / 255.0  
    v_edges = cv2.filter2D(blurred, -1, filter)
    h_edges = cv2.filter2D(blurred, -1, filter.T)
    
    sobel_edges_img = np.hypot(v_edges, h_edges)
    sobel_edges_img = np.clip(sobel_edges_img, 0, 1)

    theta = np.arctan2(h_edges, v_edges)
    
    if verbose:
        show_image(img=(sobel_edges_img * 255).astype(np.uint8), img_name="Sobel Edges")
    
    return np.squeeze(sobel_edges_img), np.squeeze(theta)

def canny_edge_detector(img: np.array, sobel_filter: np.array, gauss_sigma: float, gauss_filter_shape: List | None = None, verbose: bool = False):
    sobel_edges_img, theta = sobel_edge_detector(img, sobel_filter, gauss_sigma, gauss_filter_shape, verbose)
    canny_edges_img = non_max_suppression(sobel_edges_img, theta)
    
    if verbose:
        show_image((canny_edges_img * 255).astype(np.uint8), img_name="Canny Edges")
        
    return canny_edges_img

def detect_shapes_and_color(img, canny_sigma, sobel_filter, shape_name, target_color, min_area=1000):
    """
    Detects specific shapes with a given color in the input image.

    Args:
        img (np.array): The input image.
        canny_sigma (float): The sigma value for the Canny edge detection.
        sobel_filter (np.array): The Sobel filter to use for edge detection.
        shape_name (str): The name of the shape to detect.
        target_color (str): The desired color in BGR format ("Yellow", "Red", "Green").
        min_area (int, optional): Minimum area for contours to consider. Defaults to 1000.

    Returns:
        bool: True if the specified shape with the target color is detected, False otherwise.
    """
    canny_edges = canny_edge_detector(img, sobel_filter, canny_sigma)

    contours, _ = cv2.findContours((canny_edges * 255).astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    for contour in contours:
        # Filter small contours
        if cv2.contourArea(contour) < min_area:
            continue
        
        # Approximate the shape
        epsilon = 0.02 * cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, epsilon, True)
        
        # Classify the shape
        detected_shape = classify_shape(approx)
        
        # If the shape doesn't match the target, skip it
        if detected_shape != shape_name:
            continue

        # Create a mask for the contour to calculate the mean color
        mask = np.zeros_like(img[:, :, 0])
        cv2.drawContours(mask, [contour], -1, 255, -1)
        mean_color = cv2.mean(img, mask=mask)[:3]

        # Convert mean BGR color to HSV for color checking
        mean_color_hsv = cv2.cvtColor(np.uint8([[mean_color]]), cv2.COLOR_BGR2HSV)[0][0]

        # Define HSV ranges for the target color
        color_ranges = {
            "Yellow": [(20, 100, 100), (30, 255, 255)],
            "Red": [(0, 100, 100), (10, 255, 255), (160, 100, 100), (179, 255, 255)],
            "Green": [(40, 50, 50), (90, 255, 255)],
        }

        # Check if the color is in the target range
        if is_color_in_range(mean_color_hsv, color_ranges.get(target_color, [])):
            return True

    return False

def is_color_in_range(color_hsv, ranges):
    """Checks if an HSV color falls within specified ranges."""
    if len(ranges) > 2:  # For colors like red with two ranges
        return (cv2.inRange(np.uint8([[color_hsv]]), np.array(ranges[0]), np.array(ranges[1]))[0][0] or
                cv2.inRange(np.uint8([[color_hsv]]), np.array(ranges[2]), np.array(ranges[3]))[0][0])
    return cv2.inRange(np.uint8([[color_hsv]]), np.array(ranges[0]), np.array(ranges[1]))[0][0]

def classify_shape(approx):
    """Classifies a shape based on the number of vertices in the contour."""
    vertices = len(approx)
    if vertices == 3:
        return "Triangle"
    elif vertices == 4:
        return "Square"
    elif vertices >= 10: 
        return "Star"
    else:
        return "Unknown"

def classify_color(mean_color):
    blue, green, red = mean_color

    if red > 150 and green < 100 and blue < 100:
        return "Red"
    elif green > 150 and red < 100 and blue < 100:
        return "Green"
    elif red > 150 and green > 150 and blue < 100:
        return "Yellow"
    else:
        return "Unknown"

def detect_shapes(img: np.array, canny_sigma: float, sobel_filter: np.array, min_area: int = 1000):
    canny_edges = canny_edge_detector(img, sobel_filter, canny_sigma)
    
    contours, _ = cv2.findContours((canny_edges * 255).astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    detected_shapes = []
    
    for contour in contours:
        # Filtrar contornos pequeños (evita ruido)
        if cv2.contourArea(contour) < min_area:
            continue
        
        epsilon = 0.02 * cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, epsilon, True)
        
        shape_name = classify_shape(approx)
        
        # Calcula el color promedio dentro del contorno
        mask = np.zeros_like(img[:, :, 0])  # Máscara para el contorno actual
        cv2.drawContours(mask, [contour], -1, 255, -1)  # Dibuja el contorno en blanco sobre la máscara negra
        mean_color = cv2.mean(img, mask=mask)[:3]  # Promedio del color BGR
        
        # Clasificar el color basado en rangos
        color_name = classify_color(mean_color)
        
        if shape_name != "Unknown" and color_name != "Unknown":
            detected_shapes.append((shape_name, color_name))
    
    return detected_shapes

def detect_wrong_shape(frame, gauss_sigma, sobel_filter, expected_shape, expected_color):
    # Detecta todas las formas y colores presentes en el frame
    detected_shapes = detect_shapes(frame, gauss_sigma, sobel_filter)
    
    for shape, color in detected_shapes:
        if (shape, color) != (expected_shape, expected_color):
            return True  # Figura incorrecta encontrada

    return False  # No se encontraron figuras incorrectas


def main():
    cap = cv2.VideoCapture(0)  # Accede a la cámara
    sobel_filter = np.array([[1, 0, -1], [2, 0, -2], [1, 0, -1]])
    gauss_sigma = 1.0  # Parámetro de sigma para el filtro Gaussiano
    state = 1  # Estado inicial
    phase = 1

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: No se pudo capturar el video.")
            break

        if phase==1:
            if state == 1:
                if detect_shapes_and_color(frame, gauss_sigma, sobel_filter, "Square", "Yellow"):
                    print("Square Yellow detected!")
                    state = 2
                elif detect_wrong_shape(frame, gauss_sigma, sobel_filter, "Square", "Yellow"):  # Verifica si hay figuras incorrectas
                    print("Wrong shape detected. Returning to state 1.")
                    state = 1

            elif state == 2:
                if detect_shapes_and_color(frame, gauss_sigma, sobel_filter, "Triangle", "Yellow"):
                    print("Triangle Yellow detected!")
                    state = 3
                elif detect_wrong_shape(frame, gauss_sigma, sobel_filter, "Triangle", "Yellow"):
                    print("Wrong shape detected. Returning to state 1.")
                    state = 1

            elif state == 3:
                if detect_shapes_and_color(frame, gauss_sigma, sobel_filter, "Square", "Red"):
                    print("Square Red detected! First Sequence complete.")
                    phase =2
                elif detect_wrong_shape(frame, gauss_sigma, sobel_filter, "Square", "Red"):
                    print("Wrong shape detected. Returning to state 1.")
                    state = 1

        if phase==2:
            if state == 1:
                if detect_shapes_and_color(frame, gauss_sigma, sobel_filter, "Star", "Green"):
                    print("Star Green detected!")
                    state = 2
                elif detect_wrong_shape(frame, gauss_sigma, sobel_filter, "Star", "Green"):  # Verifica si hay figuras incorrectas
                    print("Wrong shape detected. Returning to state 1.")
                    state = 1

            elif state == 2:
                if detect_shapes_and_color(frame, gauss_sigma, sobel_filter, "Star", "Yellow"):
                    print("Star Yellow detected!")
                    state = 3
                elif detect_wrong_shape(frame, gauss_sigma, sobel_filter, "Star", "Yellow"):
                    print("Wrong shape detected. Returning to state 1.")
                    state = 1

            elif state == 3:
                if detect_shapes_and_color(frame, gauss_sigma, sobel_filter, "Star", "Red"):
                    print("Star Red detected! Second sequence complete.")
                    break
                elif detect_wrong_shape(frame, gauss_sigma, sobel_filter, "Star", "Red"):
                    print("Wrong shape detected. Returning to state 1.")
                    state = 1

        # Mostrar el frame procesado
        cv2.imshow("Shape Detection", frame)

        # Salir con la tecla 'q'
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
