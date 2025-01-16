from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from scipy import ndimage


def load_image(filepath):
    """Load an image from the disk and convert it to grayscale."""
    image = Image.open(filepath).convert('L')  # Convert to grayscale
    return np.array(image)


def preprocess_image(image):
    """Apply preprocessing steps to improve image quality."""
    # Normalizar el rango de intensidades
    image = ((image - image.min()) / (image.max() -
             image.min()) * 255).astype(np.uint8)

    # Aplicar un filtro gaussiano más suave
    blurred = ndimage.gaussian_filter(image, sigma=0.5)

    # Mejorar el contraste
    enhanced = np.clip((blurred - blurred.mean()) * 1.5 +
                       blurred.mean(), 0, 255).astype(np.uint8)

    return enhanced


def binarize_image(image, threshold=None):
    """Convert a grayscale image to binary using Otsu's method if no threshold provided."""
    if threshold is None:
        # Calculate Otsu's threshold
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

    # Invertimos la binarización aquí
    binary_image = (image <= threshold).astype(np.uint8) * 255
    return binary_image


def segment_characters(binary_image, min_size=50):
    """Segment individual characters using connected components analysis."""
    # Ahora el texto es negro (0) y el fondo blanco (255)
    inverted = binary_image == 0

    # Label connected components
    labeled_array, num_features = ndimage.label(inverted)

    # Find properties of each connected component
    objects = ndimage.find_objects(labeled_array)
    bounding_boxes = []

    for i, slice_obj in enumerate(objects):
        if slice_obj is not None:
            # Get component size
            component = inverted[slice_obj]
            if np.sum(component) > min_size:  # Filtrar componentes pequeños (ruido)
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

    # Add padding while keeping within image bounds
    top = max(0, top - padding)
    bottom = min(height, bottom + padding)
    left = max(0, left - padding)
    right = min(width, right + padding)

    return binary_image[top:bottom, left:right]


def normalize_character(char_image, target_size=(28, 28)):
    """Normalize character size and centering."""
    # Create a blank target image (white background)
    normalized = np.full(target_size, 255, dtype=np.uint8)

    # Resize character while maintaining aspect ratio
    aspect = char_image.shape[1] / char_image.shape[0]
    if aspect > 1:
        new_width = target_size[1]
        new_height = int(new_width / aspect)
    else:
        new_height = target_size[0]
        new_width = int(new_height * aspect)

    resized = np.array(Image.fromarray(
        char_image).resize((new_width, new_height)))

    # Center the character in the target image
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


if __name__ == "__main__":
    # Test the improved processing pipeline
    image_path = "AI_project/test_image.jpg"

    # Load and preprocess
    gray_image = load_image(image_path)
    preprocessed = preprocess_image(gray_image)
    binary_image = binarize_image(preprocessed)

    # Find and display characters
    bounding_boxes = segment_characters(binary_image)
    print(f"Found {len(bounding_boxes)} characters")

    # Display results
    plt.figure(figsize=(12, 4))
    plt.subplot(131)
    plt.imshow(gray_image, cmap='gray')
    plt.title('Original')

    plt.subplot(132)
    plt.imshow(preprocessed, cmap='gray')
    plt.title('Preprocessed')

    plt.subplot(133)
    plt.imshow(binary_image, cmap='gray')
    plt.title('Binarized')
    plt.show()

    # Display individual characters
    display_characters(binary_image, bounding_boxes)
