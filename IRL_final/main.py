from utils.dataset import load_data
from models.reward_model import RewardNetwork
from models.detection_model import DetectionModel
from utils.train import train_reward_model, train_detection_model
from utils.evaluation import visualize_predictions

path_images = "data/images"
path_labels = "data/train_labels/xView_train.geojson"

# Load synthetic dataset
print("Loading our data: ...")
our_data = load_data(path_images, path_labels)

test = our_data[len(our_data)-1]
our_data.pop()

images = []

for entry in our_data:
    images.append(entry["image"])
    
    
bboxes = []

for entry in our_data:
    bboxes.append(entry["bbox"])


# Train reward model using GAIL
reward_net = RewardNetwork()
print("Training Reward Model using GAIL")
train_reward_model(reward_net, images, bboxes)

# Train detection model
detector = DetectionModel()
print("Training Detection Model using the Reward model")
train_detection_model(detector, reward_net, images)

# Test & visualize results
# test_img, test_bbox = load_data(1)
visualize_predictions(detector, test["image"], test["bbox"])
