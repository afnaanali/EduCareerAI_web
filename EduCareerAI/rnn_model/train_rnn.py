import tensorflow as tf


# ==========================================
# 1. Load IMDB dataset
# ==========================================

print("Loading IMDB dataset...")

vocab_size = 10000

(x_train, y_train), (x_test, y_test) = tf.keras.datasets.imdb.load_data(
    num_words=vocab_size
)

print("Dataset loaded successfully!")
print("Training reviews:", len(x_train))
print("Testing reviews:", len(x_test))


# ==========================================
# 2. Make all reviews the same length
# ==========================================

max_length = 200

x_train = tf.keras.utils.pad_sequences(
    x_train,
    maxlen=max_length
)

x_test = tf.keras.utils.pad_sequences(
    x_test,
    maxlen=max_length
)


# ==========================================
# 3. Build RNN model
# ==========================================

model = tf.keras.Sequential([

    tf.keras.Input(shape=(max_length,)),

    # Convert word IDs into vectors
    tf.keras.layers.Embedding(
        input_dim=vocab_size,
        output_dim=32
    ),

    # RNN layer
    tf.keras.layers.SimpleRNN(
        32
    ),

    # Classification layer
    tf.keras.layers.Dense(
        1,
        activation="sigmoid"
    )
])


# ==========================================
# 4. Display model
# ==========================================

print("\nRNN Architecture:")

model.summary()


# ==========================================
# 5. Compile model
# ==========================================

model.compile(
    optimizer="adam",
    loss="binary_crossentropy",
    metrics=["accuracy"]
)


# ==========================================
# 6. Train RNN
# ==========================================

print("\nTraining RNN...")

model.fit(
    x_train,
    y_train,
    epochs=3,
    batch_size=64,
    validation_split=0.1
)


# ==========================================
# 7. Evaluate model
# ==========================================

print("\nEvaluating RNN...")

test_loss, test_accuracy = model.evaluate(
    x_test,
    y_test,
    verbose=0
)


print("\n===================================")
print("        RNN MODEL RESULTS")
print("===================================")

print(f"Test Accuracy: {test_accuracy * 100:.2f}%")

print("===================================")


# ==========================================
# 8. Save model
# ==========================================

model.save("imdb_rnn.keras")

print("\nRNN model saved as:")
print("imdb_rnn.keras")