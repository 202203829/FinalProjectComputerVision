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


def classify_shape(approx):
    # Clasifica la figura según la cantidad de vértices
    num_vertices = len(approx)
    if num_vertices == 3:
        return "Triangle"
    elif num_vertices == 4:
        # Detecta si es un cuadrado o un rectángulo
        x, y, w, h = cv2.boundingRect(approx)
        aspect_ratio = float(w) / h
        if 0.95 < aspect_ratio < 1.05:
            return "Square"
        else:
            return "Rectangle"
    elif num_vertices == 5:
        return "Pentagon"
    elif num_vertices ==10:
        return "Star"
    else:
        return "Unknown"

def detect_shapes(img: np.array, canny_sigma: float, sobel_filter: np.array, min_area: int = 1000):
    canny_edges = canny_edge_detector(img, sobel_filter, canny_sigma)
    
    contours, _ = cv2.findContours((canny_edges * 255).astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    detected_shapes = img.copy()  # Fondo original de la imagen
    
    for contour in contours:
        # Filtrar contornos pequeños (evita ruido)
        if cv2.contourArea(contour) < min_area:
            continue
        
        epsilon = 0.02 * cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, epsilon, True)
        
        shape_name = classify_shape(approx)
        
        # Calcula el color promedio dentro del contorno
        mask = np.zeros_like(img[:,:,0])  # Máscara para el contorno actual
        cv2.drawContours(mask, [contour], -1, 255, -1)  # Dibuja el contorno en blanco sobre la máscara negra
        mean_color = cv2.mean(img, mask=mask)[:3]  # Promedio del color BGR
        
        # Convierte el color a formato de texto legible
        color_name = f"BGR({int(mean_color[0])}, {int(mean_color[1])}, {int(mean_color[2])})"
        
        # Dibuja la forma detectada en la imagen original
        cv2.drawContours(detected_shapes, [approx], 0, (0, 255, 0), 2)
        
        if shape_name != "Unknown":
            print(f"Detected: {shape_name}, Color: {color_name}")
    
    return detected_shapes


def main():
    cap = cv2.VideoCapture(0)  # Accede a la cámara por defecto
    
    # Filtro Sobel utilizado para la detección de bordes
    sobel_filter = np.array([[1, 0, -1], [2, 0, -2], [1, 0, -1]])
    gauss_sigma = 1.0   # Parámetro de sigma para el filtro Gaussiano
    
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: No se pudo capturar el video.")
            break
        
        # Aplica la detección de formas
        detected_shapes_img = detect_shapes(frame, gauss_sigma, sobel_filter)
        
        # Muestra la imagen resultante con las formas detectadas
        cv2.imshow("Detected Shapes on Black Background", detected_shapes_img)
        
        # Salir con la tecla 'q'
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
