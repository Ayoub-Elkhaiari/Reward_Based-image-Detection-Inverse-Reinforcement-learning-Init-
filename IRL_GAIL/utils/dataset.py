import numpy as np
import cv2

def generate_synthetic_image(size=64):
    """Generate a synthetic image with a random rectangle."""
    image = np.zeros((size, size, 3), dtype=np.uint8)
    x, y = np.random.randint(0, size-16, 2)
    w, h = np.random.randint(10, 16, 2)
    color = np.random.randint(0, 255, 3).tolist()
    cv2.rectangle(image, (x, y), (x+w, y+h), color, -1)
    bbox = np.array([x, y, x+w, y+h])  # [x_min, y_min, x_max, y_max]
    return image, bbox

def load_synthetic_data(num_samples=1000):
    images, bboxes = [], []
    for _ in range(num_samples):
        img, bbox = generate_synthetic_image()
        images.append(img)
        bboxes.append(bbox)
    return images, bboxes
