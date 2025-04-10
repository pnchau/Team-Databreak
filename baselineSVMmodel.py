from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report

def svm_baseline(X_train, y_train, X_test, y_test, kernel='linear'):
    """
    Train and evaluate an SVM baseline model for image classification.
    """
    # Flatten images: (samples, height, width, channels) -> (samples, features)
    num_train_samples = X_train.shape[0]
    num_test_samples = X_test.shape[0]

    X_train_flat = X_train.reshape(num_train_samples, -1)
    X_test_flat = X_test.reshape(num_test_samples, -1)

    print(f"Training SVM with kernel = '{kernel}' on flattened images...")

    model = SVC(kernel=kernel)
    model.fit(X_train_flat, y_train)

    y_pred = model.predict(X_test_flat)

    accuracy = accuracy_score(y_test, y_pred)
    print(f"\nSVM Accuracy: {accuracy:.4f}")

    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))

    return model
