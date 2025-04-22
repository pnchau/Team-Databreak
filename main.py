import argparse
from load_data import load_dataset, CLASS_MAP
from preprocessing import preprocess_for_svm, preprocess_for_resnet
from visualization import show_samples_per_class, plot_class_distribution, plot_training_history
from baselineSVMmodel import train_pca_svm
from resNet50 import train_resnet50_model

def main():
    # CLI: --model svm or --model resnet
    parser = argparse.ArgumentParser(description="Brain Tumor Classification")
    parser.add_argument("--model", type=str, choices=["svm", "resnet"], default="svm", help="Model to train")
    args = parser.parse_args()

    print(">> Loading data...")
    X_train, y_train, X_test, y_test = load_dataset(subset_ratio=0.8)
    class_names = list(CLASS_MAP.keys())

    print(">> Visualizing raw samples and class distribution...")
    show_samples_per_class(X_train, y_train, class_names)
    plot_class_distribution(y_train, class_names, title="Raw Training Set Distribution")
    plot_class_distribution(y_test, class_names, title="Raw Test Set Distribution")

    if args.model == "svm":
        print(">> Preprocessing for SVM: CLAHE + normalization...")
        X_train, y_train, X_test, y_test = preprocess_for_svm(X_train, y_train, X_test, y_test)

        print(">> Visualizing preprocessed SVM data...")
        plot_class_distribution(y_train, class_names, title="SVM Training Set Distribution")
        plot_class_distribution(y_test, class_names, title="SVM Testing Set Distribution")

        print(">> Training PCA + SVM baseline model...")
        svm_model = train_pca_svm(X_train, y_train, X_test, y_test, n_components=100)

    elif args.model == "resnet":
        print(">> Preprocessing for ResNet50: CLAHE + keras.applications preprocess_input...")
        X_train, y_train, X_test, y_test = preprocess_for_resnet(X_train, y_train, X_test, y_test)

        print(">> Visualizing preprocessed ResNet data...")
        plot_class_distribution(y_train, class_names, title="ResNet Training Set Distribution")
        plot_class_distribution(y_test, class_names, title="ResNet Testing Set Distribution")

        model, history = train_resnet50_model(X_train, y_train, X_test, y_test, class_names)

        print(">> Visualizing model training performance...")
        plot_training_history(history)

    print(">> Done.")

if __name__ == "__main__":
    main()
