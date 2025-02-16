from NLPD import NLP_dist
import cv2
import os

script_dir = os.path.dirname(os.path.abspath(__file__))
os.chdir(script_dir)

# Cargar las imágenes
im1 = cv2.imread("./IMAGENES EJEMPLO/cocinero.png", cv2.IMREAD_GRAYSCALE)  # Cambia la ruta si es necesario
im2 = cv2.imread("./IMAGENES EJEMPLO/tractor.png", cv2.IMREAD_GRAYSCALE)
im3 = cv2.imread("./IMAGENES EJEMPLO/tractor rojo.png", cv2.IMREAD_GRAYSCALE)

print(NLP_dist(im1, im2))
print(NLP_dist(im3, im2))
print(NLP_dist(im3, im1))

