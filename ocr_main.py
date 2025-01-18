import cv2
import numpy as np
import os
from PIL import Image
from sklearn.metrics import pairwise_distances_argmin_min

# Función para detectar las regiones de texto


def detect_text_regions(image_path):
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise ValueError(
            "La imagen no se puede cargar. Verifique la ruta del archivo.")

    _, thresh = cv2.threshold(img, 128, 255, cv2.THRESH_BINARY)  # Binarización
    contours, _ = cv2.findContours(
        thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    bounding_boxes = []

    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        if w * h > 100:  # Filtrar contornos pequeños
            bounding_boxes.append((x, y, w, h))

    return bounding_boxes, thresh  # Devolvemos la imagen binarizada también

# Función para cargar las plantillas desde las carpetas


def load_templates_from_folders(template_folders):
    templates = {"mayusculas": [], "minusculas": [], "numeros": []}

    for folder in template_folders:
        # Usando las carpetas en plural
        for letter_case in ["Mayusculas", "minusculas", "numeros"]:
            case_folder = os.path.join(folder, letter_case)
            if os.path.exists(case_folder):  # Asegúrate de que la carpeta exista
                for img_name in os.listdir(case_folder):
                    img_path = os.path.join(case_folder, img_name)
                    img = Image.open(img_path).convert('L')
                    img = img.resize((32, 32))
                    # Suponiendo que el nombre del archivo es la letra
                    letter = img_name[0]
                    if letter_case == "Mayusculas":
                        templates["mayusculas"].append((letter, np.array(img)))
                    elif letter_case == "minusculas":
                        templates["minusculas"].append((letter, np.array(img)))
                    else:
                        templates["numeros"].append((letter, np.array(img)))
            else:
                print(f"Error: La carpeta {case_folder} no existe.")

    return templates

# Función para comparar imágenes usando una métrica de distancia


def compare_images(image1, image2):
    return np.mean((image1.astype(np.float32) - image2.astype(np.float32)) ** 2)

# Función para reconocer el carácter usando las plantillas cargadas


def recognize_character(image, templates, letter_case):
    best_match = None
    best_score = float('inf')

    for letter, template in templates[letter_case]:
        score = compare_images(image, template)
        if score < best_score:
            best_score = score
            best_match = letter

    return best_match

# Función principal


def main():
    test_image_path = "test_image.jpg"  # Ruta de la imagen de prueba
    bounding_boxes, thresh = detect_text_regions(test_image_path)
    print("Bounding boxes detected:", bounding_boxes)

    # Cargar plantillas desde las carpetas
    # Asegúrate de que estas rutas sean correctas
    template_folders = ['Templates/1', 'Templates/2']
    templates = load_templates_from_folders(template_folders)
    print(f"Plantillas cargadas. Mayusculas: {
          [template[0] for template in templates['mayusculas']]}")
    print(f"Minusculas: {[template[0]
          for template in templates['minusculas']]}")
    print(f"Numeros: {[template[0] for template in templates['numeros']]}")

    # Procesar la imagen de prueba y reconocer caracteres
    for (x, y, w, h) in bounding_boxes:
        char_image = thresh[y:y + h, x:x + w]  # Usamos la imagen binarizada
        char_image_resized = cv2.resize(char_image, (32, 32))

        # Adivinar si es mayúscula, minúscula o número (esto es básico, puede mejorarse)
        if w > h:  # Esto es solo un ejemplo, puedes modificar esta lógica
            letter_case = "mayusculas"
        elif w < h:
            letter_case = "minusculas"
        else:
            letter_case = "numeros"

        recognized_char = recognize_character(
            char_image_resized, templates, letter_case)
        print(f"Caracter reconocido: {recognized_char}, Caso: {letter_case}")

        # Dibujar un cuadrado alrededor del área del caracter en la imagen binarizada
        cv2.rectangle(thresh, (x, y), (x + w, y + h),
                      (0, 255, 0), 2)  # Cuadrado verde

    # Redimensionar la imagen binarizada para mostrarla más grande
    scale_percent = 200  # Aumentamos el tamaño de la imagen al 200%
    width = int(thresh.shape[1] * scale_percent / 100)
    height = int(thresh.shape[0] * scale_percent / 100)
    dim = (width, height)

    resized_image = cv2.resize(thresh, dim, interpolation=cv2.INTER_AREA)

    # Mostrar la imagen binarizada redimensionada con los cuadros
    cv2.imshow('Image with Bounding Boxes (Binarized)', resized_image)
    cv2.waitKey(0)  # Espera hasta que se presione una tecla
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
