import matplotlib.pyplot as plt
from collections import Counter

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

def plot_training_history(history):
    acc = history['accuracy']
    val_acc = history['val_accuracy']
    loss = history['loss']
    val_loss = history['val_loss']
    epochs = range(1, len(acc) + 1)

    plt.figure(figsize=(12, 5))

    # Accuracy plot
    plt.subplot(1, 2, 1)
    plt.plot(epochs, acc, 'b', label='Training acc')
    plt.plot(epochs, val_acc, 'r', label='Validation acc')
    plt.title('Training and Validation Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()

    # Loss plot
    plt.subplot(1, 2, 2)
    plt.plot(epochs, loss, 'b-', label='Training loss')
    plt.plot(epochs, val_loss, 'r-', label='Validation loss')
    plt.title('Training and Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()

    plt.tight_layout()
    plt.show()

