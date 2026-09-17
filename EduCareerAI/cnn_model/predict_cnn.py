import tensorflow as tf
import numpy as np


# Load the trained CNN
model = tf.keras.models.load_model("mnist_cnn.keras")


# Load MNIST test data
(_, _), (x_test, y_test) = tf.keras.datasets.mnist.load_data()


# Normalize images
x_test = x_test.astype("float32") / 255.0

# Add channel dimension
x_test = np.expand_dims(x_test, axis=-1)


print("\n===================================")
print("       EduCareerAI - CNN Demo")
print("===================================")

print(f"Test dataset contains {len(x_test)} images.")

# Ask which image to test
while True:
    try:
        index = int(input(
            f"\nChoose an image number (0-{len(x_test)-1}): "
        ))

        if 0 <= index < len(x_test):
            break

        print("Please enter a valid image number.")

    except ValueError:
        print("Please enter a number.")


# Select image
image = x_test[index]
actual_digit = y_test[index]


# Make prediction
prediction = model.predict(
    np.expand_dims(image, axis=0),
    verbose=0
)[0]


# Get predicted digit
predicted_digit = np.argmax(prediction)

# Get confidence
confidence = prediction[predicted_digit] * 100


# Display result
print("\n===================================")
print("           CNN PREDICTION")
print("===================================")

print(f"Actual Digit:    {actual_digit}")
print(f"Predicted Digit: {predicted_digit}")
print(f"Confidence:      {confidence:.2f}%")

print("===================================")


# Show probabilities
print("\nDigit Probabilities:")

for digit, probability in enumerate(prediction):
    print(f"{digit}: {probability * 100:.2f}%")