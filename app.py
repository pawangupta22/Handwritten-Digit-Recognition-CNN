
import streamlit as st
import numpy as np
from PIL import Image, ImageOps
from tensorflow import keras

st.set_page_config(
    page_title="Handwritten Digit Recognition",
    page_icon=None,
    layout="wide"
)

st.markdown("""
<style>
.stApp {
    background-color: #111827;
    color: #f9fafb;
}

.block-container {
    max-width: 1000px;
    padding-top: 2rem;
}

h1, h2, h3 {
    color: #c4b5fd;
}

.subtitle {
    color: #9ca3af;
    text-align: center;
    font-size: 17px;
}

div[data-testid="stVerticalBlockBorderWrapper"] {
    background-color: #1f2937;
    border: 1px solid #374151;
    border-radius: 14px;
    padding: 18px;
}

.stButton button {
    background-color: #7c3aed;
    color: white;
    border: none;
    border-radius: 8px;
    width: 100%;
    min-height: 42px;
    font-weight: 600;
}

.stButton button:hover {
    background-color: #6d28d9;
    color: white;
}

div[data-testid="stMetric"] {
    background-color: #1f2937;
    padding: 18px;
    border-radius: 12px;
}

hr {
    border-color: #374151;
}
</style>
""", unsafe_allow_html=True)


@st.cache_resource
def load_model():
    return keras.models.load_model("digit_model.keras")


def prepare_image(image):
    image = ImageOps.exif_transpose(image).convert("L")
    image_array = np.array(image)

    if image_array.mean() > 127:
        image_array = 255 - image_array

    mask = image_array > 30

    if not mask.any():
        return None

    rows, cols = np.where(mask)

    digit = image_array[
        rows.min():rows.max() + 1,
        cols.min():cols.max() + 1
    ]

    digit = Image.fromarray(digit.astype("uint8"))
    digit.thumbnail((20, 20))

    canvas = Image.new("L", (28, 28), 0)

    x = (28 - digit.width) // 2
    y = (28 - digit.height) // 2

    canvas.paste(digit, (x, y))

    array = np.array(canvas).astype("float32") / 255.0

    return array.reshape(1, 28, 28, 1)


def predict_digit(image):
    processed = prepare_image(image)

    if processed is None:
        st.warning("No digit detected. Please try another image.")
        return

    model = load_model()
    predictions = model.predict(processed, verbose=0)[0]

    digit = int(np.argmax(predictions))
    confidence = float(predictions[digit]) * 100

    st.markdown("### Prediction Result")

    col1, col2 = st.columns(2)

    col1.metric("Predicted Digit", digit)
    col2.metric("Confidence", f"{confidence:.2f}%")

    st.write("Confidence for each digit")

    for i, score in enumerate(predictions):
        st.write(f"Digit {i}: {score * 100:.2f}%")
        st.progress(float(score))


st.markdown(
    "<h1 style='text-align:center;'>Handwritten Digit Recognition</h1>",
    unsafe_allow_html=True
)

st.markdown(
    "<p class='subtitle'>Recognize handwritten digits using a CNN model</p>",
    unsafe_allow_html=True
)

st.divider()

try:
    load_model()
except Exception:
    st.error(
        "Unable to load digit_model.keras. "
        "Make sure the model file is uploaded to your GitHub repository."
    )
    st.stop()

upload_tab, draw_tab = st.tabs(["Upload Image", "Draw Digit"])

with upload_tab:
    st.subheader("Upload a handwritten digit")

    uploaded_file = st.file_uploader(
        "Choose an image",
        type=["png", "jpg", "jpeg"]
    )

    if uploaded_file is not None:
        image = Image.open(uploaded_file).convert("RGB")
        st.image(image, caption="Uploaded Image", width=250)

        if st.button("Recognize Digit", key="upload_button"):
            predict_digit(image)

with draw_tab:
    st.subheader("Draw a digit")

    try:
        from streamlit_drawable_canvas import st_canvas

        canvas = st_canvas(
            stroke_width=18,
            stroke_color="#FFFFFF",
            background_color="#000000",
            height=280,
            width=280,
            drawing_mode="freedraw",
            key="digit_canvas"
        )

        if st.button("Recognize Digit", key="draw_button"):
            try:
                if canvas.image_data is None:
                    st.warning("Please draw a digit first.")
                else:
                    image = Image.fromarray(
                        canvas.image_data.astype("uint8")
                    ).convert("RGB")

                    predict_digit(image)

            except RuntimeError:
                st.warning("Please draw a digit and try again.")

    except ImportError:
        st.error(
            "Drawing support is missing. "
            "Add streamlit-drawable-canvas to requirements.txt."
        )

st.divider()

st.markdown(
    "<p style='text-align:center;color:#9ca3af;'>"
    "Built with Python, TensorFlow and Streamlit"
    "</p>",
    unsafe_allow_html=True
)
