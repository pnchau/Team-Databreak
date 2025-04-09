import matplotlib.pyplot as plt
import numpy as np
from collections import Counter


# Show a few sample images per class
def show_samples_per_class(X, y, class_names, samples_per_class=2):
    plt.figure(figsize=(12, 6))
    shown = {cls: 0 for cls in range(len(class_names))}
    total_needed = samples_per_class * len(class_names)
    count = 0
    i = 0

    while count < total_needed and i < len(X):
        label = y[i]
        if shown[label] < samples_per_class:
            plt.subplot(len(class_names), samples_per_class, count + 1)
            plt.imshow(X[i].squeeze(), cmap='gray' if X[i].shape[-1] == 1 else None)
            plt.title(class_names[label])
            plt.axis('off')
            shown[label] += 1
            count += 1
        i += 1

    plt.tight_layout()
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

# Pixel intensity histogram (mean image per class)
def plot_mean_intensity(X, y, class_names):
    plt.figure(figsize=(12, 4))

    for i, class_name in enumerate(class_names):
        class_images = X[y == i]
        mean_image = np.mean(class_images, axis=0)
        mean_image = (mean_image * 255).astype(np.uint8)

        if mean_image.shape[-1] == 1:
            mean_image = mean_image.squeeze()

        plt.subplot(1, len(class_names), i + 1)
        plt.imshow(mean_image, cmap='gray' if mean_image.ndim == 2 else None)
        plt.title(f"Mean: {class_name}")
        plt.axis('off')

    plt.tight_layout()
    plt.show()
