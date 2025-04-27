import os
import cv2
import numpy as np
from sklearn.utils import shuffle

# Define image size
IMG_SIZE = 256

# Class names
CLASSES = ['glioma', 'meningioma', 'notumor', 'pituitary']
CLASS_MAP = {label: idx for idx, label in enumerate(CLASSES)}


def load_images_from_folder(folder_path):
    data = []
    labels = []

    for class_name in CLASSES:
        class_path = os.path.join(folder_path, class_name)
        class_index = CLASS_MAP[class_name]

        for img_file in os.listdir(class_path):
            img_path = os.path.join(class_path, img_file)

            #Read and preprocess image
            try:
                img = cv2.imread(img_path)
                img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

                # 0 → glioma, 1 → meningioma, 2 → notumor, 3 → pituitary
                data.append(img)
                labels.append(class_index)

            except Exception as e:
                print(f"Error loading image {img_path}: {e}")

    return np.array(data), np.array(labels)

def load_dataset(subset_ratio =1.0):
    train_dir = 'Brain_tumor_dataset/Training'
    test_dir = 'Brain_tumor_dataset/Testing'

    x_train, y_train = load_images_from_folder(train_dir)
    x_test, y_test = load_images_from_folder(test_dir)

    # Shuffle and subset the data if needed
    if subset_ratio < 1.0:
        x_train, y_train = shuffle(x_train, y_train, random_state=42)
        x_test, y_test = shuffle(x_test, y_test, random_state=42)

        train_size = int(len(x_train) * subset_ratio)
        test_size = int(len(x_test) * subset_ratio)

        x_train = x_train[:train_size]
        y_train = y_train[:train_size]
        x_test = x_test[:test_size]
        y_test = y_test[:test_size]

    return x_train, y_train, x_test, y_test
