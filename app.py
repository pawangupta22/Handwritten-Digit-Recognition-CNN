
import streamlit as st
import numpy as np
from PIL import Image, ImageOps
from tensorflow import keras
from streamlit_drawable_canvas import st_canvas

st.set_page_config(
    page_title="Handwritten Digit Recognition",
    page_icon="...",
    layout="centered"
)

st.title("Handwritten Digit Recognition")
st.write("Recognize handwritten digits from 0 to 9 using a CNN model.")

@st.cache_resource
def load_model():
    return keras.models.load_model("digit_model.keras")


def prepare_image(image):
    image = image.convert("L")
    image_array = np.array(image)

    # Convert to white digit on a black background
    if image_array.mean() > 127:
        image = ImageOps.invert(image)

    image_array = np.array(image)

    # Find the digit and remove extra background
    mask = image_array > 30

    if not mask.any():
        return None

    rows, cols = np.where(mask)

    digit = image.crop((
        cols.min(),
        rows.min(),
        cols.max() + 1,
        rows.max() + 1
    ))

    # Resize while keeping the digit's shape
    digit.thumbnail((20, 20))

    # Place digit in the center of a 28 x 28 image
    result = Image.new("L", (28, 28), 0)

    x = (28 - digit.width) // 2
    y = (28 - digit.height) // 2

    result.paste(digit, (x, y))

    # Normalize pixel values
    array = np.array(result).astype("float32") / 255.0

    # CNN input shape: (1, 28, 28, 1)
    array = array.reshape(1, 28, 28, 1)

    return array


def predict_digit(image):
    model = load_model()
    image_array = prepare_image(image)

    if image_array is None:
        st.warning("No digit detected. Please try again.")
        return

    scores = model.predict(image_array, verbose=0)[0]
    digit = int(np.argmax(scores))
    confidence = float(scores[digit]) * 100

    st.success(f"Predicted Digit: {digit}")
    st.metric("Prediction Confidence", f"{confidence:.2f}%")

    with st.expander("View scores for all digits"):
        for number, score in enumerate(scores):
            st.write(f"Digit {number}: {score * 100:.2f}%")


tab1, tab2 = st.tabs(["Upload Image", "Draw Digit"])

# Option 1: Upload an image
with tab1:
    st.subheader("Upload a handwritten digit")

    uploaded_file = st.file_uploader(
        "Choose a PNG or JPG image",
        type=["png", "jpg", "jpeg"],
        key="upload"
    )

    if uploaded_file is not None:
        image = Image.open(uploaded_file).convert("RGB")

        st.image(image, caption="Uploaded Image", width=250)

        if st.button("Predict Uploaded Digit"):
            predict_digit(image)


# Option 2: Draw a digit
with tab2:
    st.subheader("Draw a digit from 0 to 9")
    st.write("Use your mouse to draw one digit in the black box.")

    canvas = st_canvas(
        fill_color="rgba(0, 0, 0, 1)",
        stroke_width=18,
        stroke_color="#FFFFFF",
        background_color="#000000",
        height=280,
        width=280,
        drawing_mode="freedraw",
        key="digit_canvas"
    )

    if st.button("Predict Drawn Digit"):
        if canvas.image_data is not None:
            image = Image.fromarray(
                canvas.image_data.astype("uint8")
            ).convert("RGB")

            predict_digit(image)
        else:
            st.warning("Please draw a digit first.")

st.divider()
st.caption("CNN model trained on the MNIST dataset.")
