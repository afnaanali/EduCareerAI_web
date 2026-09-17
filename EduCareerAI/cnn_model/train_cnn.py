import tensorflow as tf
import numpy as np


# ==========================================
# 1. Load MNIST dataset
# ==========================================

print("Loading MNIST dataset...")

(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

print("Dataset loaded successfully!")
print("Training images:", x_train.shape)
print("Testing images:", x_test.shape)


# ==========================================
# 2. Normalize the images
# ==========================================

x_train = x_train.astype("float32") / 255.0
x_test = x_test.astype("float32") / 255.0


# Add channel dimension
# 28x28 → 28x28x1

x_train = np.expand_dims(x_train, axis=-1)
x_test = np.expand_dims(x_test, axis=-1)


# ==========================================
# 3. Build CNN model
# ==========================================

model = tf.keras.Sequential([

    tf.keras.Input(shape=(28, 28, 1)),

    # Convolution layer
    tf.keras.layers.Conv2D(
        32,
        (3, 3),
        activation="relu"
    ),

    # Pooling layer
    tf.keras.layers.MaxPooling2D(
        (2, 2)
    ),

    # Second convolution layer
    tf.keras.layers.Conv2D(
        64,
        (3, 3),
        activation="relu"
    ),

    # Second pooling layer
    tf.keras.layers.MaxPooling2D(
        (2, 2)
    ),

    # Convert feature maps into a vector
    tf.keras.layers.Flatten(),

    # Fully connected layer
    tf.keras.layers.Dense(
        64,
        activation="relu"
    ),

    # Output layer: digits 0-9
    tf.keras.layers.Dense(
        10,
        activation="softmax"
    )
])


# ==========================================
# 4. Display model architecture
# ==========================================

print("\nCNN Architecture:")
model.summary()


# ==========================================
# 5. Compile model
# ==========================================

model.compile(
    optimizer="adam",
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)


# ==========================================
# 6. Train CNN
# ==========================================

print("\nTraining CNN...")

model.fit(
    x_train,
    y_train,
    epochs=5,
    batch_size=64,
    validation_split=0.1
)


# ==========================================
# 7. Evaluate model
# ==========================================

print("\nEvaluating CNN...")

test_loss, test_accuracy = model.evaluate(
    x_test,
    y_test,
    verbose=0
)


print("\n===================================")
print("        CNN MODEL RESULTS")
print("===================================")

print(f"Test Accuracy: {test_accuracy * 100:.2f}%")

print("===================================")


# ==========================================
# 8. Save model
# ==========================================

model.save("mnist_cnn.keras")

print("\nCNN model saved as:")
print("mnist_cnn.keras")