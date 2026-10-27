
import streamlit as st
import numpy as np
from PIL import Image, ImageOps
from tensorflow import keras

st.set_page_config(
    page_title="Handwritten Digit Recognition",
    layout="centered"
)

st.markdown("""
<style>
.stApp {
    background-color: #111827;
    color: #f9fafb;
}
.block-container {
    max-width: 850px;
    padding-top: 2rem;
}
h1, h2, h3 {
    color: #c4b5fd;
}
.stButton button {
    background-color: #7c3aed;
    color: white;
    border: none;
    border-radius: 8px;
    min-height: 42px;
    font-weight: 600;
}
.stButton button:hover {
    background-color: #6d28d9;
    color: white;
}
div[data-testid="stMetric"] {
    background-color: #1f2937;
    padding: 16px;
    border-radius: 12px;
}
</style>
""", unsafe_allow_html=True)


@st.cache_resource
def load_model():
    return keras.models.load_model("digit_model.keras")


def prepare_image(image):
    image = ImageOps.exif_transpose(image).convert("L")
    arr = np.array(image)

    border = np.concatenate((
        arr[0, :],
        arr[-1, :],
        arr[:, 0],
        arr[:, -1]
    ))

    if np.median(border) > 127:
        arr = 255 - arr

    mask = arr > 35

    if not np.any(mask):
        return None, None

    ys, xs = np.where(mask)

    digit = arr[
        ys.min():ys.max() + 1,
        xs.min():xs.max() + 1
    ]

    h, w = digit.shape
    scale = 20 / max(h, w)

    new_w = max(1, int(round(w * scale)))
    new_h = max(1, int(round(h * scale)))

    digit_image = Image.fromarray(digit).resize(
        (new_w, new_h),
        Image.Resampling.LANCZOS
    )

    canvas = Image.new("L", (28, 28), 0)

    x = (28 - new_w) // 2
    y = (28 - new_h) // 2

    canvas.paste(digit_image, (x, y))

    input_array = np.asarray(
        canvas, dtype=np.float32
    ) / 255.0

    input_array = input_array.reshape(1, 28, 28, 1)

    return canvas, input_array


st.markdown(
    "<h1 style='text-align:center;'>Handwritten Digit Recognition</h1>",
    unsafe_allow_html=True
)

st.markdown(
    "<p style='text-align:center;color:#9ca3af;'>Recognize one handwritten digit using a CNN model</p>",
    unsafe_allow_html=True
)

st.divider()

try:
    model = load_model()
except Exception as error:
    st.error(
        "Unable to load digit_model.keras. "
        "Make sure it is in the same folder as app.py."
    )
    st.exception(error)
    st.stop()


uploaded_file = st.file_uploader(
    "Upload an image containing one digit (0–9)",
    type=["png", "jpg", "jpeg"]
)

if uploaded_file is not None:
    try:
        original = Image.open(uploaded_file)

        st.image(
            original,
            caption="Uploaded image",
            width=250
        )

        if st.button("Recognize Digit", type="primary"):
            processed_image, input_data = prepare_image(original)

            if input_data is None:
                st.warning(
                    "No digit was detected. "
                    "Please upload a clearer image."
                )
            else:
                scores = model.predict(
                    input_data, verbose=0
                )[0]

                predicted_digit = int(np.argmax(scores))
                confidence = float(scores[predicted_digit]) * 100

                col1, col2 = st.columns(2)

                col1.metric(
                    "Predicted Digit",
                    predicted_digit
                )

                col2.metric(
                    "Model Confidence",
                    f"{confidence:.2f}%"
                )

                with st.expander("View image sent to the model"):
                    st.image(
                        processed_image,
                        caption="Processed 28 × 28 image",
                        width=180
                    )

                with st.expander("Confidence for each digit"):
                    for digit, score in enumerate(scores):
                        st.write(
                            f"Digit {digit}: {score * 100:.2f}%"
                        )
                        st.progress(float(score))

    except Exception as error:
        st.error(f"Could not process this image: {error}")


st.divider()

st.caption(
    "Best results: upload one clear, centered handwritten digit, "
    "not a full page or multiple digits."
)
