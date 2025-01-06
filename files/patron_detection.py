import cv2
import numpy as np

# Variable global para evitar detecciones repetidas
last_detected = None

def is_color_in_range(mean_color, color_range):
    """Check if a color is within the given HSV range."""
    lower, upper = color_range
    return all(lower[i] <= mean_color[i] <= upper[i] for i in range(3))

def detect_shapes_and_color(frame, expected_shape, expected_color, color_ranges, min_area=100):
    """
    Detect shapes and colors in the frame and verify if they match the expected ones.
    """
    global last_detected

    hsv_img = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    gray_img = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray_img, (5, 5), 0)
    edges = cv2.Canny(blurred, 50, 150)
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    for contour in contours:
        if cv2.contourArea(contour) < min_area:
            continue

        # Compute contour properties
        perimeter = cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, 0.04 * perimeter, True)
        num_vertices = len(approx)
        aspect_ratio = calculate_aspect_ratio(contour)
        shape_area = cv2.contourArea(contour)

        # Detect shape based on properties
        detected_shape = None
        if num_vertices == 3:
            detected_shape = "Triangle"
        elif num_vertices == 4:
            if 0.9 < aspect_ratio < 1.1:  # Square (aspect ratio close to 1)
                detected_shape = "Square"
        elif num_vertices ==10:  # Many vertices -> likely a Star
            detected_shape = "Star"

        # Check color
        mask = np.zeros_like(gray_img)
        cv2.drawContours(mask, [contour], 0, 255, -1)
        mean_color = cv2.mean(hsv_img, mask=mask)[:3]

        if (
            detected_shape == expected_shape and
            is_color_in_range(mean_color, color_ranges[expected_color]) and
            last_detected != (detected_shape, expected_color)
        ):
            last_detected = (detected_shape, expected_color)
            return True

    return False


def calculate_aspect_ratio(contour):
    """
    Calculate the aspect ratio of a contour's bounding rectangle.
    """
    x, y, w, h = cv2.boundingRect(contour)
    return float(w) / h if h != 0 else 0


def detect_wrong_shape(frame, expected_shape, expected_color, color_ranges, min_area=100):
    """
    Detect if there is a wrong shape or color in the frame and print detailed error messages.
    """
    global last_detected

    hsv_img = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    gray_img = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray_img, (5, 5), 0)
    edges = cv2.Canny(blurred, 50, 150)
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    for contour in contours:
        if cv2.contourArea(contour) < min_area:
            continue

        epsilon = 0.02 * cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, epsilon, True)
        num_vertices = len(approx)

        mask = np.zeros_like(gray_img)
        cv2.drawContours(mask, [contour], 0, 255, -1)
        mean_color = cv2.mean(hsv_img, mask=mask)[:3]

        detected_shape = None
        if num_vertices == 3:
            detected_shape = "Triangle"
        elif num_vertices == 4:
            detected_shape = "Square"
        elif num_vertices > 8:
            detected_shape = "Star"

        if (
            detected_shape and
            last_detected != (detected_shape, expected_color)
        ):
            if detected_shape != expected_shape:
                print(f"Detected wrong shape: {detected_shape} instead of {expected_shape}.")
                last_detected = (detected_shape, expected_color)
                return True
            elif not is_color_in_range(mean_color, color_ranges[expected_color]):
                print(f"Detected wrong color: {mean_color} instead of {expected_color}.")
                last_detected = (detected_shape, expected_color)
                return True

    return False

def main():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: No se pudo abrir la cámara.")
        return

    state = 1
    phase = 1
    color_ranges = {
        "red": [(0, 100, 100), (10, 255, 255)],
        "green": [(35, 100, 100), (85, 255, 255)],
        "yellow": [(0, 0, 100), (100, 55, 255)],
    }

    print("Correct Sequence 1 is Yellow Square, Green Triangle, Red Square")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: No se pudo capturar el video.")
            break

        if phase == 1:
            if state == 1:
                if detect_shapes_and_color(frame, "Square", "yellow", color_ranges):
                    print("Square Yellow detected!")
                    state = 2
                elif detect_wrong_shape(frame, "Square", "yellow", color_ranges):
                    state = 1

            elif state == 2:
                if detect_shapes_and_color(frame, "Triangle", "green", color_ranges):
                    print("Triangle Green detected!")
                    state = 3
                elif detect_wrong_shape(frame, "Triangle", "green", color_ranges):
                    state = 1

            elif state == 3:
                if detect_shapes_and_color(frame, "Square", "red", color_ranges):
                    print("Square Red detected! First Sequence complete.")
                    phase = 2
                    state = 1
                    print("Correct Sequence 2 is Green Star, Yellow Star, Red Star")
                elif detect_wrong_shape(frame, "Square", "red", color_ranges):
                    state = 1

        elif phase == 2:
            if state == 1:
                if detect_shapes_and_color(frame, "Star", "green", color_ranges):
                    print("Star Green detected!")
                    state = 2
                elif detect_wrong_shape(frame, "Star", "green", color_ranges):
                    state = 1

            elif state == 2:
                if detect_shapes_and_color(frame, "Star", "yellow", color_ranges):
                    print("Star Yellow detected!")
                    state = 3
                elif detect_wrong_shape(frame, "Star", "yellow", color_ranges):
                    state = 1

            elif state == 3:
                if detect_shapes_and_color(frame, "Star", "red", color_ranges):
                    print("Star Red detected! Second Sequence complete.")
                    break
                elif detect_wrong_shape(frame, "Star", "red", color_ranges):
                    state = 1

        cv2.imshow("Shape Detection", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
