import numpy as np
import tensorflow as tf


# Load trained ANN
model = tf.keras.models.load_model("career_ann.keras")

# Load career names
career_classes = np.load("career_classes.npy", allow_pickle=True)


# Skill names — must match the training dataset order
skills = [
    "Python",
    "SQL",
    "AWS",
    "Docker",
    "Linux",
    "Git",
    "PowerBI",
    "Excel"
]


print("\n===================================")
print("     EduCareerAI - ANN Predictor")
print("===================================")
print("Enter 1 if you have the skill.")
print("Enter 0 if you don't have the skill.\n")


# Get skills from user
user_skills = []

for skill in skills:
    while True:
        value = input(f"Do you know {skill}? (1/0): ")

        if value in ["0", "1"]:
            user_skills.append(int(value))
            break

        print("Please enter only 1 or 0.")


# Convert input into NumPy array
input_data = np.array([user_skills])


# Make prediction
prediction = model.predict(input_data, verbose=0)[0]


# Find highest probability
predicted_index = np.argmax(prediction)
predicted_career = career_classes[predicted_index]
confidence = prediction[predicted_index] * 100


# Display result
print("\n===================================")
print("          PREDICTION")
print("===================================")

print(f"Recommended Career: {predicted_career}")
print(f"Confidence: {confidence:.2f}%")

print("\nCareer Probabilities:")

for career, probability in zip(career_classes, prediction):
    print(f"{career}: {probability * 100:.2f}%")

print("===================================")