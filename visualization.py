import random
import cv2
import matplotlib.pyplot as plt
from collections import Counter
import numpy as np
import tensorflow as tf

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

def visualize_model_results(model, x_test, y_test, class_names, history, num_images=10):
    preds = model.predict(x_test)
    pred_labels = np.argmax(preds, axis=1)

    # Random Predictions (correct + incorrect)
    print(">> Showing random predictions (correct/incorrect):")
    plt.figure(figsize=(20, 8))
    for i in range(num_images):
        idx = random.randint(0, len(x_test) - 1)
        plt.subplot(2, 5, i + 1)
        plt.imshow(x_test[idx].astype("uint8"))
        plt.axis('off')
        true_cls = class_names[y_test[idx]]
        pred_cls = class_names[pred_labels[idx]]
        color = "green" if pred_cls == true_cls else "red"
        plt.title(f"Pred: {pred_cls}\nTrue: {true_cls}", color=color)
    plt.tight_layout()
    plt.show()
    


def make_gradcam_heatmap(img_array, model, last_conv_layer_name, pred_index=None):
    grad_model = tf.keras.models.Model(
        [model.inputs], [model.get_layer(last_conv_layer_name).output, model.output]
    )

    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(np.expand_dims(img_array, axis=0))
        if pred_index is None:
            pred_index = tf.argmax(predictions[0])
        class_channel = predictions[:, pred_index]

    grads = tape.gradient(class_channel, conv_outputs)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
    conv_outputs = conv_outputs[0]
    heatmap = conv_outputs @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)
    heatmap = tf.maximum(heatmap, 0) / tf.math.reduce_max(heatmap)
    return heatmap.numpy()

def show_gradcam_overlay(image, heatmap, class_label=None, save_path=None):
    plt.figure(figsize=(8, 4))

    plt.subplot(1, 2, 1)
    plt.imshow(image)
    plt.axis('off')
    plt.title(f"Original Image\nLabel: {class_label}" if class_label else "Original")

    plt.subplot(1, 2, 2)
    heatmap_resized = upscale_heatmap(heatmap, image.shape[:2])
    superimposed_img = cv2.addWeighted(image.astype("uint8"), 0.6, heatmap_resized, 0.4, 0)

    plt.imshow(superimposed_img)

    plt.axis('off')
    plt.title("Grad-CAM Heatmap")

    if save_path:
        plt.savefig(save_path)
        print(f">> Grad-CAM saved to {save_path}")
    else:
        plt.show()


def upscale_heatmap(heatmap, target_size):
    heatmap_resized = cv2.resize(heatmap, (target_size[1], target_size[0]))  # (width, height)
    heatmap_resized = np.uint8(255 * heatmap_resized)
    heatmap_color = cv2.applyColorMap(heatmap_resized, cv2.COLORMAP_JET)
    return heatmap_color


def generate_gradcam_grid(model, x_test, y_test, class_names, num_classes=4, save_path="gradcam_grid.png"):
    selected_images = []
    selected_labels = []
    selected_preds = []
    selected_heatmaps = []

    found_classes = set()

    # Go through test set and collect 1 example for each class
    for idx in range(len(x_test)):
        true_label = y_test[idx]
        if true_label in found_classes:
            continue

        img = x_test[idx]
        pred_probs = model.predict(np.expand_dims(img, axis=0))[0]
        pred_label = np.argmax(pred_probs)

        # Generate Grad-CAM
        heatmap = make_gradcam_heatmap(img, model, "conv5_block3_out")
        heatmap_resized = cv2.resize(heatmap, (img.shape[1], img.shape[0]))
        heatmap_resized = np.uint8(255 * heatmap_resized)
        heatmap_color = cv2.applyColorMap(heatmap_resized, cv2.COLORMAP_JET)
        overlay = cv2.addWeighted(img.astype("uint8"), 0.6, heatmap_color, 0.4, 0)

        # Save results
        selected_images.append(img)
        selected_labels.append(true_label)
        selected_preds.append(pred_label)
        selected_heatmaps.append(overlay)

        found_classes.add(true_label)
        if len(found_classes) == num_classes:
            break

    # Plot
    plt.figure(figsize=(20, 5))
    for i in range(num_classes):
        # Row 1: original with labels
        plt.subplot(2, num_classes, i+1)
        plt.imshow(selected_images[i].astype("uint8"))
        plt.axis('off')
        pred = class_names[selected_preds[i]]
        true = class_names[selected_labels[i]]
        color = 'green' if selected_preds[i] == selected_labels[i] else 'red'
        plt.title(f"Pred: {pred}\nTrue: {true}", color=color)

        # Row 2: Grad-CAM overlay
        plt.subplot(2, num_classes, i+1+num_classes)
        plt.imshow(selected_heatmaps[i])
        plt.axis('off')

    plt.tight_layout()
    plt.savefig(save_path)
    print(f">> Grad-CAM grid saved to {save_path}")
    plt.show()
