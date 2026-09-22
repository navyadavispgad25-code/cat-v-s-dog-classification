import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing import image

# Load trained model
model = tf.keras.models.load_model("cat_dog_model.h5")

# Image to predict
img_path = "test_image.jpg"

# Load image
img = image.load_img(
    img_path,
    target_size=(128, 128)
)

# Convert image to array
img_array = image.img_to_array(img)

# Add batch dimension
img_array = np.expand_dims(img_array, axis=0)

# Normalize pixel values
img_array = img_array / 255.0

# Make prediction
prediction = model.predict(img_array)[0][0]

# Display result
if prediction >= 0.5:
    result = "Dog"
    confidence = prediction * 100
else:
    result = "Cat"
    confidence = (1 - prediction) * 100

print("Prediction:", result)
print("Confidence:", round(confidence, 2), "%")
