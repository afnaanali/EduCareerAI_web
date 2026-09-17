import tensorflow as tf
import numpy as np


# ==========================================
# 1. Load trained RNN
# ==========================================

model = tf.keras.models.load_model("imdb_rnn.keras")


# ==========================================
# 2. Load IMDB word index
# ==========================================

word_index = tf.keras.datasets.imdb.get_word_index()

# Reserve special tokens
word_index = {
    word: index + 3
    for word, index in word_index.items()
}

word_index["<PAD>"] = 0
word_index["<START>"] = 1
word_index["<UNK>"] = 2


# ==========================================
# 3. Convert text to numbers
# ==========================================

def encode_review(text, max_length=200):

    words = text.lower().split()

    encoded = [1]  # <START>

    for word in words:
        encoded.append(word_index.get(word, 2))

    # Keep only first 200 tokens
    encoded = encoded[:max_length]

    # Pad sequence
    encoded = tf.keras.utils.pad_sequences(
        [encoded],
        maxlen=max_length
    )

    return encoded


# ==========================================
# 4. Get user review
# ==========================================

print("\n===================================")
print("       EduCareerAI - RNN Demo")
print("===================================")

review = input("\nEnter a movie review: ")


# ==========================================
# 5. Convert review
# ==========================================

input_data = encode_review(review)


# ==========================================
# 6. Make prediction
# ==========================================

prediction = model.predict(
    input_data,
    verbose=0
)[0][0]


# ==========================================
# 7. Determine sentiment
# ==========================================

if prediction >= 0.5:
    sentiment = "Positive 😊"
    confidence = prediction * 100
else:
    sentiment = "Negative 😞"
    confidence = (1 - prediction) * 100


# ==========================================
# 8. Display result
# ==========================================

print("\n===================================")
print("          RNN PREDICTION")
print("===================================")

print(f"Sentiment: {sentiment}")
print(f"Confidence: {confidence:.2f}%")

print("===================================")