from models.reward_network import RewardNetwork
from models.detection_model import DetectionModel
from utils.dataset import generate_synthetic_image
from utils.train import train_irl, train_detector
from utils.evaluation import compute_iou
import torch
import matplotlib.pyplot as plt
import cv2

# Generate dataset
num_samples = 1000
images, bboxes = zip(*[generate_synthetic_image() for _ in range(num_samples)])

# Train reward network
reward_net = RewardNetwork()
train_irl(reward_net, images, bboxes, epochs=5)

# Train detector
detector = DetectionModel()
train_detector(detector, reward_net, images, epochs=5)

# Evaluate model
test_img, test_bbox = generate_synthetic_image()
img_tensor = torch.tensor(test_img).permute(2, 0, 1).float().unsqueeze(0) / 255.0
pred_bbox = detector(img_tensor).detach().numpy().squeeze()

iou = compute_iou(test_bbox, pred_bbox)
print(f"IoU: {iou:.2f}")
