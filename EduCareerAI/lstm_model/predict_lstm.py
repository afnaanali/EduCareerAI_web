import tensorflow as tf
from tensorflow.keras.datasets import imdb
from tensorflow.keras.preprocessing.sequence import pad_sequences

# Settings
VOCAB_SIZE = 10000
MAX_LENGTH = 200

# Load trained model
model = tf.keras.models.load_model("imdb_lstm.keras")

# Load IMDB word index
word_index = imdb.get_word_index()

# Function to convert text into numbers
def encode_review(text):
    words = text.lower().split()

    encoded = []

    for word in words:
        word_id = word_index.get(word)

        if word_id is not None:
            encoded.append(word_id + 3)
        else:
            encoded.append(2)  # Unknown word

    return pad_sequences(
        [encoded],
        maxlen=MAX_LENGTH
    )

# Get user input
review = input("\nEnter a movie review: ")

# Convert text to model input
input_data = encode_review(review)

# Predict
prediction = model.predict(input_data, verbose=0)[0][0]

# Display result
if prediction >= 0.5:
    sentiment = "Positive 😊"
    confidence = prediction * 100
else:
    sentiment = "Negative 😞"
    confidence = (1 - prediction) * 100

print("\n--- LSTM Prediction ---")
print("Sentiment:", sentiment)
print(f"Confidence: {confidence:.2f}%")