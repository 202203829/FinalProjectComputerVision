from typing import List
import numpy as np
import imageio
import cv2
import copy
import glob
import os
from os.path import dirname, join

def load_images(filenames: List) -> List:
    return [imageio.v2.imread(filename) for filename in filenames]

def find_corners(imgs: List) -> List:
    return [cv2.findChessboardCorners(img, (7,7), None) for img in imgs]

def refine_corners(corners: List) -> List:
    corners_copy = copy.deepcopy(corners)
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.01)
    imgs_gray = [cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) for img in imgs]
    return [cv2.cornerSubPix(i, cor[1], (7, 7), (-1, -1), criteria) if cor[0] else [] for i, cor in zip(imgs_gray, corners_copy)]

def draw_show_and_save_corners(imgs_copy: List, corners: List, corners_refined: List, output_dir: str) -> None:
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)  # Crea el directorio si no existe
    
    for i, (img, cor, found) in enumerate(zip(imgs_copy, corners_refined, [c[0] for c in corners])):
        if found:
            cv2.drawChessboardCorners(img, (7, 7), cor, found)  # Dibuja las esquinas
            
            # Muestra la imagen con las esquinas dibujadas
            cv2.imshow(f'Image {i}', img)
            cv2.waitKey(500)  # Muestra cada imagen durante 500 ms (ajustable)

            # Guarda la imagen si no existe
            # filename = os.path.join(os.getcwd(), f"foto_{i}.png")
            if not os.path.exists("./data/foto_{i}.jpg"):
                cv2.imwrite(f"./data/foto_{i}.jpg", img)
    
    cv2.destroyAllWindows()  # Cierra todas las ventanas de OpenCV

def get_chessboard_points(chessboard_shape, dx, dy):
    points = []
    for y in range(chessboard_shape[1]):
        for x in range(chessboard_shape[0]):
            points.append([x*dx, y*dy, 0])
    return np.array(points, np.float32)
    
if __name__ == "__main__":
    imgs_path = [f"..\Visión por ordenador\FinalProject\data\calibration\\foto_{i}.jpg" for i in range(1,15)]
    imgs = load_images(imgs_path)
    corners = find_corners(imgs)
    corners_refined = refine_corners(corners)
    imgs_copy = copy.deepcopy(imgs)
    output_dir = "C:\\Users\\pablo\\OneDrive\\Escritorio\\iMat\\iMat 3º\\Visión por ordenador\\FinalProject\\data\\output\\"
    draw_show_and_save_corners(imgs_copy, corners, corners_refined, output_dir)
    
    chessboard_points = get_chessboard_points((7, 7), 30, 30)
    valid_corners = [cor[1] for cor in corners if cor[0]]
    valid_corners = np.asarray(valid_corners, dtype=np.float32)
    objpoints = []  
    imgpoints = []  

    for corners in valid_corners:
        objpoints.append(chessboard_points)  
        imgpoints.append(corners)   

    rms, intrinsics, dist_coeffs, rvecs, tvecs = cv2.calibrateCamera(
        objpoints, imgpoints, (320, 240), None, None
    )

    extrinsics = list(map(lambda rvec, tvec: np.hstack((cv2.Rodrigues(rvec)[0], tvec)), rvecs, tvecs))
    print("Intrinsics:\n", intrinsics)
    print("Distortion coefficients:\n", dist_coeffs)
    print("Root mean squared reprojection error:\n", rms)