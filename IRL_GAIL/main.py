from utils.dataset import load_synthetic_data
from models.reward_network import RewardNetwork
from models.detection_model import DetectionModel
from utils.train import train_reward_model, train_detection_model
from utils.evaluation import visualize_predictions

# Load synthetic dataset
images, bboxes = load_synthetic_data(1000)

# Train reward model using GAIL
reward_net = RewardNetwork()
print("Training Reward Model using GAIL")
train_reward_model(reward_net, images, bboxes)

# Train detection model
detector = DetectionModel()
print("Training Detection Model using the Reward model")
train_detection_model(detector, reward_net, images)

# Test & visualize results
test_img, test_bbox = load_synthetic_data(1)
visualize_predictions(detector, test_img[0], test_bbox[0])
