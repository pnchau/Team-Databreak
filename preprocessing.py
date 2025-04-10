import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from collections import Counter
from sklearn.metrics import confusion_matrix

def show_samples_per_class(images, labels, class_names, samples_per_class=5):
    plt.figure(figsize=(15, 8))
    num_classes = len(class_names)

    for class_index, class_name in enumerate(class_names):
        class_imgs = images[labels == class_index]

        for i in range(samples_per_class):
            idx = class_index * samples_per_class + i + 1
            plt.subplot(num_classes, samples_per_class, idx)
            plt.imshow(class_imgs[i])
            plt.axis('off')
            plt.title(class_name, fontsize=10)

    plt.suptitle("Sample Images Per Class")
    plt.tight_layout()
    plt.subplots_adjust(top=0.9)  # Adjust to fit suptitle
    plt.show()

# Plot class distribution
def plot_class_distribution(y, class_names, title="Class Distribution"):
    counts = Counter(y)
    labels = [class_names[i] for i in range(len(class_names))]
    values = [counts[i] for i in range(len(class_names))]

    plt.figure(figsize=(8, 4))
    plt.bar(labels, values, color='skyblue')
    plt.title(title)
    plt.xlabel("Classes")
    plt.ylabel("Number of Images")
    plt.grid(axis='y')
    plt.show()
