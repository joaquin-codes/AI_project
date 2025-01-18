import cv2
import numpy as np
import os
from PIL import Image, ImageTk
import tkinter as tk
from tkinter import filedialog, messagebox
from tkinter import ttk

# Function to detect text regions


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

# Function to load templates from folders


def load_templates_from_folders(template_folders):
    templates = {"mayusculas": [], "minusculas": [], "numeros": []}

    for folder in template_folders:
        for letter_case in ["mayusculas", "minusculas", "numeros"]:  # Use lowercase keys
            # Capitalize to match folder names
            case_folder = os.path.join(folder, letter_case.capitalize())
            if os.path.exists(case_folder):
                for img_name in os.listdir(case_folder):
                    img_path = os.path.join(case_folder, img_name)
                    img = Image.open(img_path).convert('L')
                    img = img.resize((32, 32))
                    letter = img_name[0]
                    templates[letter_case].append(
                        (letter, np.array(img)))  # Use lowercase keys here
            else:
                print(f"Error: La carpeta {case_folder} no existe.")

    return templates

# Function to compare images using cross-correlation


def compare_images(image1, image2):
    image1 = image1.astype(np.float32)
    image2 = image2.astype(np.float32)
    corr = np.sum(image1 * image2) / (np.sqrt(np.sum(image1**2))
                                      * np.sqrt(np.sum(image2**2)))
    return 1 - corr  # Queremos minimizar la diferencia

# Function to recognize character using loaded templates


def recognize_character(image, templates):
    best_match = None
    best_score = float('inf')
    letter_case = "numeros"  # Default case

    # Check numbers first
    for letter, template in templates["numeros"]:
        score = compare_images(image, template)
        if score < best_score:
            best_score = score
            best_match = letter
            letter_case = "numeros"

    # Check uppercase letters
    for letter, template in templates["mayusculas"]:
        score = compare_images(image, template)
        if score < best_score:
            best_score = score
            best_match = letter
            letter_case = "mayusculas"

    # Check lowercase letters
    for letter, template in templates["minusculas"]:
        score = compare_images(image, template)
        if score < best_score:
            best_score = score
            best_match = letter
            letter_case = "minusculas"

    return best_match, letter_case, best_score

# Function to preprocess image


def preprocess_image(image_path):
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError("No se puede cargar la imagen.")
    resized_img = cv2.resize(img, (50, 50))  # Resize to 50x50
    return resized_img

# Function to run OCR on the image


def run_ocr(image_path, templates):
    bounding_boxes, thresh = detect_text_regions(image_path)
    print("Bounding boxes detected:", bounding_boxes)

    recognized_chars = []  # Store recognized characters
    best_score = None

    for (x, y, w, h) in bounding_boxes:
        char_image = thresh[y:y + h, x:x + w]  # Extract region of interest
        char_image_resized = cv2.resize(char_image, (32, 32))

        # Recognize the character
        recognized_char, letter_case, score = recognize_character(
            char_image_resized, templates)
        recognized_chars.append((recognized_char, letter_case))

        if best_score is None or score < best_score:
            best_score = score

        # Draw bounding box around the character
        cv2.rectangle(thresh, (x, y), (x + w, y + h),
                      (0, 255, 0), 2)  # Green rectangle

    return thresh, recognized_chars, best_score

# File dialog function


def open_file_dialog():
    file_path = filedialog.askopenfilename(
        filetypes=[("Image files", "*.jpg *.png *.jpeg")])
    if file_path:
        image_label.config(
            text=f"Imagen {os.path.basename(file_path)} cargada exitosamente")
        return file_path
    else:
        return None

# Update UI with results


def update_ui(image_path):
    try:
        # Resize uploaded image
        resized_img_path = "resized_image.jpg"
        resized_img = preprocess_image(image_path)
        cv2.imwrite(resized_img_path, resized_img)

        # Load templates and run OCR
        templates = load_templates_from_folders(["Templates/1", "Templates/2"])
        thresh_image, recognized_chars, best_score = run_ocr(
            resized_img_path, templates)

        # Display processed image
        image_pil = Image.fromarray(thresh_image)
        image_tk = ImageTk.PhotoImage(image_pil)

        img_label.config(image=image_tk)
        img_label.image = image_tk  # Prevent garbage collection

        result_label.config(text=f"Caracteres reconocidos: {', '.join(
            [f'{char} ({case})' for char, case in recognized_chars])}")
        score_label.config(text=f"Mejor score: {best_score:.4f}")
    except Exception as e:
        messagebox.showerror("Error", str(e))


# Create the main application window
root = tk.Tk()
root.title("OCR Mejorado")
root.geometry("800x600")

# Modern UI styling
style = ttk.Style()
style.configure("TButton", font=("Arial", 12), padding=10)
style.configure("TLabel", font=("Arial", 12), padding=5)

# Layout
frame = ttk.Frame(root, padding=20)
frame.pack(fill=tk.BOTH, expand=True)

instruction_label = ttk.Label(
    frame, text="Selecciona una imagen para reconocer los caracteres")
instruction_label.pack(pady=20)

select_button = ttk.Button(
    frame, text="Seleccionar Imagen", command=lambda: update_ui(open_file_dialog()))
select_button.pack(pady=10)

image_label = ttk.Label(frame, text="")
image_label.pack()

img_label = ttk.Label(frame)
img_label.pack(pady=20)

result_label = ttk.Label(frame, text="Caracteres reconocidos: N/A")
result_label.pack(pady=10)

score_label = ttk.Label(frame, text="Mejor score: N/A")
score_label.pack(pady=10)

# Start the application
root.mainloop()
