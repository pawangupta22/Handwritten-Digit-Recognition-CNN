
    # Convert image to array and normalize pixel values
    input_array = np.asarray(canvas, dtype=np.float32) / 255.0
    input_array = input_array.reshape(1, 28, 28, 1)
    return canvas, input_array


st.markdown("<h1 style='text-align:center;'>Handwritten Digit Recognition</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align:center;color:#9ca3af;'>Recognize one handwritten digit using a CNN model</p>", unsafe_allow_html=True)
st.divider()

try:
    model = load_model()
except Exception as error:
    st.error("Unable to load digit_model.keras. Make sure it is in the same folder as app.py.")
    st.exception(error)
    st.stop()

uploaded_file = st.file_uploader(
    "Upload an image containing one digit (0–9)",
    type=["png", "jpg", "jpeg"]
)

if uploaded_file is not None:
    try:
        original = Image.open(uploaded_file)
        st.image(original, caption="Uploaded image", width=250)

        if st.button("Recognize Digit", type="primary"):
            processed_image, input_data = prepare_image(original)

            if input_data is None:
                st.warning("No digit was detected. Please upload a clearer image.")
            else:
                scores = model.predict(input_data, verbose=0)[0]
                predicted_digit = int(np.argmax(scores))
                confidence = float(scores[predicted_digit]) * 100

                col1, col2 = st.columns(2)
                col1.metric("Predicted Digit", predicted_digit)
                col2.metric("Model confidence", f"{confidence:.2f}%")

                with st.expander("View image sent to the model"):
                    st.image(processed_image, caption="Processed 28 × 28 image", width=180)

                with st.expander("Confidence for each digit"):
                    for digit, score in enumerate(scores):
                        st.write(f"Digit {digit}: {score * 100:.2f}%")
                        st.progress(float(score))

    except Exception as error:
        st.error(f"Could not process this image: {error}")

st.divider()
st.caption("Best results: upload one clear, centered handwritten digit, not a full page or multiple digits.")
