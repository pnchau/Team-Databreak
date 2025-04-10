from load_data import load_dataset, CLASS_MAP
from preprocessing import preprocess_data
from visualization import show_samples_per_class, plot_class_distribution
from baselineSVMmodel import train_pca_svm


def main():
    print(">> Loading data...")
    X_train, y_train, X_test, y_test = load_dataset()
    class_names = list(CLASS_MAP.keys())

    print(">> Visualizing raw samples and class distribution...")
    show_samples_per_class(X_train, y_train, class_names)
    plot_class_distribution(y_train, class_names, title="Raw Training Set Distribution")
    plot_class_distribution(y_test, class_names, title="Raw Test Set Distribution")

    print(">> Preprocessing clahe, normalization) and data augmentation...")
    X_train, y_train, X_test, y_test = preprocess_data(X_train, y_train, X_test, y_test)

    print(">> Visualizing preprocessed and augmented data...")
    show_samples_per_class(X_train, y_train, class_names)
    plot_class_distribution(y_train, class_names, title="Preprocessed + Augmented Training Set Distribution")

    print(">> Training PCA + SVM baseline model...")
    svm_model = train_pca_svm(X_train, y_train, X_test, y_test, n_components=100)

    print(">> All steps completed successfully")

if __name__ == "__main__":
    main()
