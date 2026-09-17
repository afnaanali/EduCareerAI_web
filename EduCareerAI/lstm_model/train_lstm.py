import tensorflow as tf
from tensorflow.keras.datasets import imdb
from tensorflow.keras.preprocessing.sequence import pad_sequences

# Settings
VOCAB_SIZE = 10000
MAX_LENGTH = 200

# Load IMDB dataset
(x_train, y_train), (x_test, y_test) = imdb.load_data(num_words=VOCAB_SIZE)

# Pad reviews to same length
x_train = pad_sequences(x_train, maxlen=MAX_LENGTH)
x_test = pad_sequences(x_test, maxlen=MAX_LENGTH)

# Build LSTM model
model = tf.keras.Sequential([
    tf.keras.Input(shape=(MAX_LENGTH,)),
    tf.keras.layers.Embedding(
        input_dim=VOCAB_SIZE,
        output_dim=32
    ),
    tf.keras.layers.LSTM(32),
    tf.keras.layers.Dense(1, activation="sigmoid")
])

# Compile
model.compile(
    optimizer="adam",
    loss="binary_crossentropy",
    metrics=["accuracy"]
)

# Show model architecture
model.summary()

# Train
history = model.fit(
    x_train,
    y_train,
    epochs=3,
    batch_size=64,
    validation_split=0.1
)

# Evaluate
loss, accuracy = model.evaluate(x_test, y_test)

print(f"\nTest Accuracy: {accuracy * 100:.2f}%")

# Save model
model.save("imdb_lstm.keras")

print("\nLSTM model saved as imdb_lstm.keras")