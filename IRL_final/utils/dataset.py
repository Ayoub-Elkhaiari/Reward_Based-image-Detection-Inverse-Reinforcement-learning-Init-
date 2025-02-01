# import os 
# import json
# import numpy as np 
# import cv2

# path_images = "data/images"
# path_labels = "data/train_labels/xView_train.geojson"

# def load_data(path_img, path_lab):
    
#     with open(path_lab, "r") as f :
#         data = json.load(f) 
        
#     images = []

#     for image in os.listdir(path_images):
#         if image.endswith((".jpg", ".jpeg", ".png", ".tif")):
#             img_path = os.path.join(path_images, image)
#             img = cv2.imread(img_path)
            
#             if img is not None: 
#                 images.append((image, img))
                
#     whole = []

#     for d in data["features"]:
#         image_id = d["properties"]["image_id"]  # Extract image filename
#         bbox_coords = d["properties"]["bounds_imcoords"]  # Extract bounding box coordinates

#         for img in images:
#             if img[0] == image_id:  # Compare image filenames
#                 whole.append({
#                     "name": img[0],   # Image filename
#                     "image": img[1],  # Image data (numpy array)
#                     "bbox": bbox_coords  # Bounding box coordinates
#                 })  
                
                
#     return whole        


import os 
import json
import numpy as np 
import cv2

def load_data(path_img, path_lab):
    with open(path_lab, "r") as f:
        data = json.load(f) 
        
    images = []

    for image in os.listdir(path_img):
        if image.endswith((".jpg", ".jpeg", ".png", ".tif")):
            img_path = os.path.join(path_img, image)
            img = cv2.imread(img_path)
            
            if img is not None: 
                img = cv2.resize(img, (256, 256))  # Resize to a fixed size
                images.append((image, img))
                
    whole = []

    for d in data["features"]:
        image_id = d["properties"]["image_id"]  # Extract image filename
        bbox_coords = list(map(float, d["properties"]["bounds_imcoords"].split(',')))  # Extract bounding box coordinates

        for img in images:
            if img[0] == image_id:  # Compare image filenames
                whole.append({
                    "name": img[0],   # Image filename
                    "image": img[1],  # Image data (numpy array)
                    "bbox": bbox_coords  # Bounding box coordinates
                })  
                
    return whole