import cv2
import matplotlib.pyplot as plt
import torch 

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

def visualize_predictions(detector, test_image, test_bbox):
    """Visualize ground truth and predicted bounding boxes."""
    img_tensor = torch.tensor(test_image).permute(2, 0, 1).float().unsqueeze(0) / 255.0
    pred_bbox = detector(img_tensor).detach().numpy().squeeze()
    
    test_bbox = [int(coord) for coord in test_bbox]
    pred_bbox = [int(coord) for coord in pred_bbox]

    img_with_boxes = test_image.copy()
    cv2.rectangle(img_with_boxes, (test_bbox[0], test_bbox[1]), (test_bbox[2], test_bbox[3]), (0, 255, 0), 2)
    cv2.rectangle(img_with_boxes, (pred_bbox[0], pred_bbox[1]), (pred_bbox[2], pred_bbox[3]), (0, 0, 255), 2)

    plt.imshow(cv2.cvtColor(img_with_boxes, cv2.COLOR_BGR2RGB))
    plt.title(f"Expert BBox (Green): {test_bbox}\nPredicted BBox (Red): {pred_bbox}")
    plt.show()

    iou = compute_iou(test_bbox, pred_bbox)
    print(f"IoU: {iou:.2f}")