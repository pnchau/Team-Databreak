from load_data import load_dataset, CLASS_MAP
from preprocessing import preprocess_data
from visualization import (
    show_samples_per_class,
    plot_class_distribution,
    plot_mean_intensity
)


def main():
    print(">> Loading data...")
    X_train, y_train, X_test, y_test = load_dataset()

    class_names = list(CLASS_MAP.keys())

    print(">> Displaying raw data samples...")
    show_samples_per_class(X_train, y_train, class_names)

    print(">> Raw class distribution:")
    plot_class_distribution(y_train, class_names, title="Raw Training Set Distribution")
    plot_class_distribution(y_test, class_names, title="Raw Test Set Distribution")

    print(">> Preprocessing and augmenting data...")
    X_train, y_train, X_test, y_test = preprocess_data(X_train, y_train, X_test, y_test)

    print(">> Displaying preprocessed samples...")
    show_samples_per_class(X_train, y_train, class_names)

    print(">> Preprocessed class distribution:")
    plot_class_distribution(y_train, class_names, title="Preprocessed + Augmented Training Set Distribution")
    plot_class_distribution(y_test, class_names, title="Preprocessed Test Set Distribution")

    print(">> Visualizing average image intensity per class...")
    plot_mean_intensity(X_train, y_train, class_names)

    print(">> Done.")

if __name__ == "__main__":
    main()
