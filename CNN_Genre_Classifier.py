import numpy as np
import json
from sklearn.model_selection import train_test_split
import tensorflow.keras as keras
import matplotlib.pyplot as plt

DATA_PATH = "data.json"


def load_data(data_path):
    with open(data_path, "r") as fp:
        data = json.load(fp)

    x = np.array(data["mfcc"])
    y = np.array(data["labels"])

    return x, y


def plot_history(history):
    fig, ax = plt.subplots(2)

    # Creating Accuracy Subplot
    ax[0].plot(history.history["accuracy"], label="Train Accuracy")
    ax[0].plot(history.history["val_accuracy"], label="Test Accuracy")
    ax[0].set_ylabel("Accuracy")
    ax[0].legend(loc="lower right")
    ax[0].set_title("Accuracy Eval")

    # Creating Error Subplot
    ax[1].plot(history.history["loss"], label="Train Error")
    ax[1].plot(history.history["val_loss"], label="Test Error")
    ax[1].set_ylabel("Error")
    ax[1].set_xlabel("Epochs")
    ax[1].legend(loc="upper right")
    ax[1].set_title("Error Eval")

    plt.show()


def prepare_datasets(test_size, validation_size):
    x, y = load_data(DATA_PATH)

    # Split the train and Test Dataset
    X_train, X_test, Y_train, Y_test = train_test_split(x, y, test_size=test_size)

    # Split the Train and Validation Dataset
    X_train, X_validation, Y_train, Y_validation = train_test_split(X_train, Y_train, test_size=validation_size)

    # Change it to 3D array
    X_train = X_train[..., np.newaxis]
    X_validation = X_validation[..., np.newaxis]
    X_test = X_test[..., np.newaxis]

    return X_train, X_validation, X_test, Y_train, Y_validation, Y_test


def build_model(Input_shape):
    # Create Model
    Model = keras.Sequential()

    # 1st Convolutional Layer
    Model.add(keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=Input_shape))
    Model.add(keras.layers.MaxPool2D((3, 3), strides=(2, 2), padding='same'))
    Model.add(keras.layers.BatchNormalization())

    # 2nd Convolutional Layer
    Model.add(keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=Input_shape))
    Model.add(keras.layers.MaxPool2D((3, 3), strides=(2, 2), padding='same'))
    Model.add(keras.layers.BatchNormalization())

    # 3rd Convolutional Layer
    Model.add(keras.layers.Conv2D(32, (2, 2), activation='relu', input_shape=Input_shape))
    Model.add(keras.layers.MaxPool2D((2, 2), strides=(2, 2), padding='same'))
    Model.add(keras.layers.BatchNormalization())

    # Flatten the output and feed it into dense layer
    Model.add(keras.layers.Flatten())
    Model.add(keras.layers.Dense(64, activation='relu'))
    Model.add(keras.layers.Dropout(0.3))

    # Output Layer
    Model.add(keras.layers.Dense(10, activation='softmax'))

    return Model


def predict(Model, X, Y):
    X = X[np.newaxis, ...]  # array shape (1, 130, 13, 1)

    # Perform the prediction
    prediction = Model.predict(X)

    print("Prediction Shape", prediction.shape)

    # Get the Index of the max value
    predicted_index = np.argmax(prediction, axis=1)

    print("Output : {}, Predicted label: {}".format(Y, predicted_index))


if __name__ == "__main__":
    # Prepare the dataset
    x_train, x_validation, x_test, y_train, y_validation, y_test = prepare_datasets(0.25, 0.2)

    # Build the model
    input_shape = (x_train.shape[1], x_train.shape[2], x_train.shape[3])
    model = build_model(input_shape)

    # Compile the model
    optimizer = keras.optimizers.Adam(learning_rate=0.0001)
    model.compile(optimizer=optimizer, loss="sparse_categorical_crossentropy", metrics=["accuracy"])

    model.summary()

    # Train the CNN
    history = model.fit(x_train, y_train, validation_data=(x_validation, y_validation), epochs=50, batch_size=32)

    # Evaluate the CNN
    test_error, test_accuracy = model.evaluate(x_test, y_test, verbose=1)
    print("Accuracy on the Test Set is : {}".format(test_accuracy))

    # Plot the train and test error
    plot_history(history)

    # Prediction/Inference
    x = x_test[100]
    y = y_test[100]
    predict(model, x, y)
    
    # Save the trained model
    model.save("music_genre_classifier.h5")
