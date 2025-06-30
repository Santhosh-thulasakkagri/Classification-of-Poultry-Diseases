from keras.applications.vgg16 import VGG16
from keras.applications.vgg16 import preprocess_input
import os
import cv2

image_size = (224, 224, 3)

train_data = 'data/train'
val_data = 'data/val'
test_data = 'data/test'

def read_data(folder):
	data, label, paths = [], [], []
	labels = os.listdir(folder)  # Define 'labels' as the subdirectories in the folder
	for label_name in labels:
		path = f"{folder}/{label_name}/"
		folder_data = os.listdir(path)[:500]  # Limit to 500 images per label
		for image_path in folder_data:
			img = cv2.imread(os.path.join(path, image_path))
			if img is not None:  # Ensure the image is read successfully
				data.append(img)
				label.append(label_name)
				paths.append(os.path.join(path, image_path))
	return data, label, paths

all_data, all_labels, all_paths = read_data(train_data)