import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

st.set_page_config(
    page_title="Cat vs Dog Classifier",
    page_icon="🐱",
    layout="centered"
)

st.title("🐱🐶 Cat vs Dog Image Classifier")
st.write("Upload an image to classify it as a Cat or Dog.")

# Load model
model = tf.keras.models.load_model("cat_dog_model.keras")

# Upload image
uploaded_file = st.file_uploader(
    "Upload a Cat or Dog image",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None:

    img = Image.open(uploaded_file).convert("RGB")

    st.image(
        img,
        caption="Uploaded Image",
        use_container_width=True
    )

    # Preprocessing
    img_resized = img.resize((160, 160))

    img_array = np.array(img_resized)
    img_array = np.expand_dims(img_array, axis=0)

    # Prediction
    prediction = model.predict(img_array, verbose=0)[0][0]

    if prediction >= 0.5:
        result = "🐶 Dog"
        confidence = prediction * 100
    else:
        result = "🐱 Cat"
        confidence = (1 - prediction) * 100

    st.subheader("Prediction")
    st.success(result)

    st.metric(
        "Confidence",
        f"{confidence:.2f}%"
    )

    st.progress(float(confidence / 100))
    