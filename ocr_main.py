import cv2
import numpy as np
import os
from PIL import Image, ImageTk
import tkinter as tk
from tkinter import filedialog, messagebox
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

# Función principal para cargar imagen y mostrar resultados


def run_ocr(image_path, templates):
    bounding_boxes, thresh = detect_text_regions(image_path)
    print("Bounding boxes detected:", bounding_boxes)

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

    return resized_image, recognized_char

# Función para abrir el diálogo de selección de archivo


def open_file_dialog():
    file_path = filedialog.askopenfilename(
        filetypes=[("Image files", "*.jpg *.png *.jpeg")])
    if file_path:
        image_label.config(
            text=f"Imagen {os.path.basename(file_path)} cargada exitosamente")
        return file_path
    else:
        return None

# Función para actualizar la interfaz con los resultados


def update_ui(image_path):
    templates = load_templates_from_folders(['Templates/1', 'Templates/2'])
    resized_image, recognized_char = run_ocr(image_path, templates)

    # Convertir la imagen a formato que Tkinter pueda mostrar
    image_pil = Image.fromarray(resized_image)
    image_tk = ImageTk.PhotoImage(image_pil)

    # Mostrar la imagen en el UI
    img_label.config(image=image_tk)
    img_label.image = image_tk  # Necesario para evitar que la imagen se pierda

    result_label.config(text=f"Caracter reconocido: {recognized_char}")


# Crear la ventana principal
root = tk.Tk()
root.title("OCR Simple UI")

# Configurar la interfaz
frame = tk.Frame(root)
frame.pack(pady=20)

instruction_label = tk.Label(frame, text="Hey there, please select your image")
instruction_label.pack()

# Botón para seleccionar la imagen
select_button = tk.Button(frame, text="Select Image",
                          command=lambda: update_ui(open_file_dialog()))
select_button.pack()

image_label = tk.Label(frame, text="")
image_label.pack()

img_label = tk.Label(frame)
img_label.pack(pady=20)

result_label = tk.Label(frame, text="")
result_label.pack()

# Iniciar la interfaz
root.mainloop()
