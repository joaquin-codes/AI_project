from PIL import Image
import numpy as np
import matplotlib.pyplot as plt


def load_image(filepath):
    """Load an image from the disk and convert it to grayscale."""
    image = Image.open(filepath).convert('L')  # Convert to grayscale
    return np.array(image)


def binarize_image(image, threshold=128):
    """Convert a grayscale image to binary using a threshold."""
    binary_image = (image > threshold).astype(np.uint8) * 255
    return binary_image


def find_character_bounding_boxes(binary_image):
    """Find bounding boxes for regions of text in the binary image."""
    # Detect connected components
    rows, cols = np.where(binary_image == 0)  # Find black pixels (characters)
    if len(rows) == 0 or len(cols) == 0:
        raise ValueError("No characters detected in the image.")

    # Define bounding box for the entire image
    top, bottom = rows.min(), rows.max()
    left, right = cols.min(), cols.max()

    # Return as a list of bounding boxes for scalability
    return [(top, bottom, left, right)]


def extract_character(binary_image, bounding_box):
    """Extract the region of the image defined by the bounding box."""
    top, bottom, left, right = bounding_box
    return binary_image[top:bottom + 1, left:right + 1]


def display_bounding_boxes(binary_image, bounding_boxes):
    """Display the binary image with bounding boxes overlaid."""
    plt.imshow(binary_image, cmap="gray")
    for bbox in bounding_boxes:
        top, bottom, left, right = bbox
        # Draw rectangles
        plt.gca().add_patch(plt.Rectangle((left, top), right - left, bottom - top,
                                          edgecolor='red', facecolor='none', linewidth=2))
    plt.title("Bounding Boxes")
    plt.show()


if __name__ == "__main__":
    # Step 1: Load and preprocess the image
    image_path = "test_image.jpg"  # Change this to your image file
    gray_image = load_image(image_path)
    binary_image = binarize_image(gray_image)

    # Step 2: Find bounding boxes for text regions
    bounding_boxes = find_character_bounding_boxes(binary_image)
    print(f"Bounding boxes detected: {bounding_boxes}")

    # Step 3: Display the bounding boxes
    display_bounding_boxes(binary_image, bounding_boxes)

    # Step 4: Extract and display each character
    for i, bbox in enumerate(bounding_boxes):
        char_image = extract_character(binary_image, bbox)
        plt.imshow(char_image, cmap="gray")
        plt.title(f"Character {i+1}")
        plt.show()
