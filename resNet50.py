from tensorflow.keras.applications import ResNet50
from tensorflow.keras.applications.resnet50 import preprocess_input as resnet_preprocess
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.layers import Dense, Dropout, GlobalAveragePooling2D
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.regularizers import l2
import pickle
import os
import numpy as np
from sklearn.utils.class_weight import compute_class_weight
from tensorflow.keras.layers import Input



def build_resnet50_model(input_shape=(224, 224, 3), num_classes=4):
    input_layer = Input(shape=input_shape, name='input_layer')
    base_model = ResNet50(include_top=False, weights='imagenet', input_tensor=input_layer)

    # Fine-tune the last few layers
    for layer in base_model.layers[-10:]:
        layer.trainable = True

    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dense(512, activation='relu', kernel_regularizer=l2(0.001))(x)
    x = Dropout(0.5)(x)
    x = Dense(256, activation='relu')(x)
    x = Dropout(0.5)(x)
    predictions = Dense(num_classes, activation='softmax')(x)

    model = Model(inputs=base_model.input, outputs=predictions)
    return model

def train_resnet50_model(x_train, y_train, x_test, y_test, class_names, force_retrain=False):
    model_path = "resnet50_brain_tumor_model.keras"
    history_path = "resnet50_training_history.pkl"

    if os.path.exists(model_path) and os.path.exists(history_path) and not force_retrain:
        print(">> Found saved model and training history. Loading...")
        model = load_model(model_path)
        with open(history_path, 'rb') as f:
            history = pickle.load(f)
        return model, history

    print(">> Training a new ResNet50 model...")

    model = build_resnet50_model(input_shape=(224, 224, 3), num_classes=len(class_names))
    model.compile(optimizer=Adam(learning_rate=1e-4),
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])

    # One-hot encoding
    y_train_cat = to_categorical(y_train, num_classes=len(class_names))
    y_test_cat = to_categorical(y_test, num_classes=len(class_names))

    # Compute class weights to handle imbalance
    class_weights = compute_class_weight(class_weight='balanced',
                                         classes=np.unique(y_train),
                                         y=y_train)
    class_weight_dict = dict(enumerate(class_weights))

    # Preprocess for ResNet50
    x_train = resnet_preprocess(x_train)
    x_test = resnet_preprocess(x_test)

    # Image augmentation
    datagen = ImageDataGenerator(
        rotation_range=30,
        width_shift_range=0.1,
        height_shift_range=0.1,
        zoom_range=0.1,
        horizontal_flip=True,
        fill_mode='nearest'
    )
    datagen.fit(x_train)

    early_stop = EarlyStopping(monitor='val_accuracy', mode = 'max', patience=2, restore_best_weights=True)

    history = model.fit(datagen.flow(x_train, y_train_cat, batch_size=32),
                        epochs=8,
                        validation_data=(x_test, y_test_cat),
                        callbacks=[early_stop],
                        class_weight=class_weight_dict)

    # Save model and training history
    model.save("resnet50_brain_tumor_model.keras")
    with open(history_path, 'wb') as f:
        pickle.dump(history.history, f)

    print(f">> Model saved to {model_path}")
    print(f">> Training history saved to {history_path}")

    return model, history.history