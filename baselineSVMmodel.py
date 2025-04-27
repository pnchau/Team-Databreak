from sklearn.decomposition import PCA
from sklearn.svm import SVC
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.model_selection import train_test_split
import numpy as np

# Flatten images for PCA and SVM
def flatten_images(X):
    return X.reshape(X.shape[0], -1)

# Apply PCA + SVM
def train_pca_svm(X_train, y_train, X_test, y_test, n_components=100):
    print(">> Flattening images...")
    X_train_flat = flatten_images(X_train)
    X_test_flat = flatten_images(X_test)

    print(f">> Applying PCA with {n_components} components...")
    pca = PCA(n_components=n_components)

    print(">> Training SVM...")
    svm = SVC(kernel='rbf', C=1.0, gamma='scale')

    # Create pipeline
    model = Pipeline([
        ('pca', pca),
        ('svm', svm)
    ])

    model.fit(X_train_flat, y_train)
    y_pred = model.predict(X_test_flat)

    # Evaluation
    print(">> Classification Report:")
    print(classification_report(y_test, y_pred))

    print(">> Confusion Matrix:")
    print(confusion_matrix(y_test, y_pred))

    print(f">> Accuracy: {accuracy_score(y_test, y_pred):.4f}")

    return model