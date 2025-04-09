import numpy as np
import cv2
import random

# Preprocessing Functions
def apply_clahe(img):
    if len(img.shape) == 3:
        img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    img = clahe.apply(img)
    img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
    return img

def normalize_images(images):
    images = images.astype('float32')
    return images / 255.0

def preprocess_image(img):
    img = apply_clahe(img)
    return img

def preprocess_dataset(images):
    return np.array([preprocess_image(img) for img in images])

#Augmentating data
AUGMENT_COUNT = 2  # Number of augmented images per original image

def random_transform(img):
    rows, cols, _ = img.shape

    # Random rotation
    angle = random.uniform(-30, 30)
    M_rot = cv2.getRotationMatrix2D((cols / 2, rows / 2), angle, 1)
    img = cv2.warpAffine(img, M_rot, (cols, rows), borderMode=cv2.BORDER_REFLECT)

    # Random shift
    dx = int(random.uniform(-0.1, 0.1) * cols)
    dy = int(random.uniform(-0.1, 0.1) * rows)
    M_trans = np.float32([[1, 0, dx], [0, 1, dy]])
    img = cv2.warpAffine(img, M_trans, (cols, rows), borderMode=cv2.BORDER_REFLECT)

    # Horizontal flip
    if random.random() < 0.5:
        img = cv2.flip(img, 1)

    return img

def augment_data(X_train, y_train):
    augmented_images = []
    augmented_labels = []

    for img, label in zip(X_train, y_train):
        for _ in range(AUGMENT_COUNT):
            aug_img = random_transform(img)
            augmented_images.append(aug_img)
            augmented_labels.append(label)

    X_aug = np.array(augmented_images)
    y_aug = np.array(augmented_labels)

    X_total = np.concatenate((X_train, X_aug), axis=0)
    y_total = np.concatenate((y_train, y_aug), axis=0)

    return X_total, y_total

#preprocessing data
def preprocess_data(X_train, y_train, X_test, y_test):
    print(">> Preprocessing training and test sets...")
    X_train = preprocess_dataset(X_train)
    X_test = preprocess_dataset(X_test)

    print(">> Normalizing...")
    X_train = normalize_images(X_train)
    X_test = normalize_images(X_test)

    print(">> Augmenting training data...")
    X_train, y_train = augment_data(X_train, y_train)

    return X_train, y_train, X_test, y_test
