import cv2
import matplotlib.pyplot as plt

def compute_iou(box1, box2):
    """Compute Intersection over Union (IoU)."""
    x1_min, y1_min, x1_max, y1_max = box1
    x2_min, y2_min, x2_max, y2_max = box2
    inter_x_min = max(x1_min, x2_min)
    inter_y_min = max(y1_min, y2_min)
    inter_x_max = min(x1_max, x2_max)
    inter_y_max = min(y1_max, y2_max)
    inter_area = max(0, inter_x_max - inter_x_min) * max(0, inter_y_max - inter_y_min)
    union_area = (x1_max - x1_min)*(y1_max - y1_min) + (x2_max - x2_min)*(y2_max - y2_min) - inter_area
    return inter_area / union_area if union_area > 0 else 0
