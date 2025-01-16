from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from scipy import ndimage
import os


def load_image(filepath):
    """Load an image from the disk and convert it to grayscale."""
    image = Image.open(filepath).convert('L')  # Convert to grayscale
    return np.array(image)


def preprocess_image(image):
    """Apply preprocessing steps to improve image quality."""
    image = ((image - image.min()) / (image.max() -
             image.min()) * 255).astype(np.uint8)
    blurred = ndimage.gaussian_filter(image, sigma=0.5)
    enhanced = np.clip((blurred - blurred.mean()) * 1.5 +
                       blurred.mean(), 0, 255).astype(np.uint8)
    return enhanced


def binarize_image(image, threshold=None):
    """Convert a grayscale image to binary using Otsu's method if no threshold provided."""
    if threshold is None:
        hist, bins = np.histogram(image.flatten(), 256, [0, 256])
        total_pixels = sum(hist)
        current_max = 0
        threshold = 0
        sumT = sum(i * hist[i] for i in range(256))
        sumB = 0
        wB = 0

        for i in range(256):
            wB += hist[i]
            if wB == 0:
                continue
            wF = total_pixels - wB
            if wF == 0:
                break
            sumB += i * hist[i]
            mB = sumB / wB
            mF = (sumT - sumB) / wF
            between = wB * wF * (mB - mF) ** 2
            if between > current_max:
                current_max = between
                threshold = i

    binary_image = (image <= threshold).astype(np.uint8) * 255
    return binary_image


def segment_characters(binary_image, min_size=50):
    """Segment individual characters using connected components analysis."""
    inverted = binary_image == 0
    labeled_array, num_features = ndimage.label(inverted)
    objects = ndimage.find_objects(labeled_array)
    bounding_boxes = []

    for slice_obj in objects:
        if slice_obj is not None:
            component = inverted[slice_obj]
            if np.sum(component) > min_size:
                y_slice, x_slice = slice_obj
                top = y_slice.start
                bottom = y_slice.stop
                left = x_slice.start
                right = x_slice.stop
                bounding_boxes.append((top, bottom, left, right))

    return bounding_boxes


def extract_character(binary_image, bounding_box, padding=2):
    """Extract character from image with optional padding."""
    top, bottom, left, right = bounding_box
    height, width = binary_image.shape
    top = max(0, top - padding)
    bottom = min(height, bottom + padding)
    left = max(0, left - padding)
    right = min(width, right + padding)
    return binary_image[top:bottom, left:right]


def normalize_character(char_image, target_size=(32, 32)):
    """Normalize character size and centering."""
    normalized = np.full(target_size, 255, dtype=np.uint8)
    aspect = char_image.shape[1] / char_image.shape[0]
    if aspect > 1:
        new_width = target_size[1]
        new_height = int(new_width / aspect)
    else:
        new_height = target_size[0]
        new_width = int(new_height * aspect)

    resized = np.array(Image.fromarray(
        char_image).resize((new_width, new_height)))
    y_offset = (target_size[0] - new_height) // 2
    x_offset = (target_size[1] - new_width) // 2

    normalized[y_offset:y_offset+new_height,
               x_offset:x_offset+new_width] = resized
    return normalized


def display_characters(binary_image, bounding_boxes):
    """Display all detected characters in a grid."""
    n = len(bounding_boxes)
    cols = min(5, n)
    rows = (n + cols - 1) // cols

    plt.figure(figsize=(12, 2*rows))
    for i, bbox in enumerate(bounding_boxes):
        char_image = extract_character(binary_image, bbox)
        normalized = normalize_character(char_image)

        plt.subplot(rows, cols, i+1)
        plt.imshow(normalized, cmap='gray')
        plt.axis('off')
        plt.title(f'Char {i+1}')

    plt.tight_layout()
    plt.show()


def load_templates(template_folder):
    """Load character templates from a folder, including subfolders."""
    templates = {}
    for root, _, files in os.walk(template_folder):
        for file_name in files:
            if file_name.endswith(".png") or file_name.endswith(".jpg"):
                char = os.path.splitext(file_name)[0]
                image = Image.open(os.path.join(root, file_name)).convert('L')
                image = np.array(image)
                templates[char] = normalize_character(
                    image, target_size=(32, 32))
    return templates


def recognize_character(char_image, templates):
    """Recognize a character by comparing it to templates."""
    best_match = None
    best_score = float('inf')

    for char, template in templates.items():
        resized_char = normalize_character(char_image, target_size=(32, 32))
        score = np.sum((resized_char - template) ** 2)

        if score < best_score:
            best_score = score
            best_match = char

    return best_match


if __name__ == "__main__":
    image_path = "AI_project/test_image.jpg"
    template_folder = "AI_project/templates"

    gray_image = load_image(image_path)
    preprocessed = preprocess_image(gray_image)
    binary_image = binarize_image(preprocessed)

    bounding_boxes = segment_characters(binary_image)
    print(f"Found {len(bounding_boxes)} characters")

    templates = load_templates(template_folder)
    print(f"Loaded {len(templates)} templates for recognition.")

    recognized_text = ""
    for bbox in bounding_boxes:
        char_image = extract_character(binary_image, bbox)
        normalized_char = normalize_character(char_image, target_size=(32, 32))
        recognized_char = recognize_character(normalized_char, templates)
        recognized_text += recognized_char if recognized_char else "?"

    print(f"Recognized text: {recognized_text}")

    display_characters(binary_image, bounding_boxes)
