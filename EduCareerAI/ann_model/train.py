import pandas as pd
import numpy as np
import tensorflow as tf

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout


# 1. Load dataset
df = pd.read_csv("career_dataset.csv")

print("Dataset loaded successfully!")
print(df.head())


# 2. Separate input and output
X = df.drop("Career", axis=1)
y = df["Career"]


# 3. Convert career names into numbers
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)

print("\nCareer classes:")
for i, career in enumerate(label_encoder.classes_):
    print(i, "=", career)


# 4. Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y_encoded,
    test_size=0.4,
    random_state=42,
    stratify=y_encoded
)


# 5. Build ANN
model = Sequential([
    tf.keras.Input(shape=(8,)),
    Dense(16, activation="relu"),
    Dropout(0.2),
    Dense(8, activation="relu"),
    Dense(len(label_encoder.classes_), activation="softmax")
])


# 6. Compile model
model.compile(
    optimizer="adam",
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)


# 7. Train model
print("\nTraining ANN...")

history = model.fit(
    X_train,
    y_train,
    epochs=100,
    batch_size=4,
    verbose=1
)


# 8. Evaluate model
loss, accuracy = model.evaluate(X_test, y_test, verbose=0)

print("\n==============================")
print("ANN MODEL RESULTS")
print("==============================")
print(f"Test Accuracy: {accuracy * 100:.2f}%")
print("==============================")


# 9. Save model
model.save("career_ann.keras")

# Save career labels
np.save("career_classes.npy", label_encoder.classes_)

print("\nModel saved as:")
print("career_ann.keras")

print("\nCareer classes saved as:")
print("career_classes.npy")