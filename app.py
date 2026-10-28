
import streamlit as st
import tensorflow as tf
import numpy as np
import cv2
from PIL import Image, ImageFilter, ImageEnhance
from pathlib import Path
import io

st.set_page_config(
    page_title="Temple Image Colorization GAN",
    page_icon="🏛️",
    layout="wide"
)

st.title("Temple Image Colorization and Restoration Using GAN")
st.write(
    "Upload a grayscale or old temple photograph to generate "
    "a colorized image using your trained Pix2Pix GAN model."
)

MODEL_PATH = Path(__file__).parent / "temple_colorization_generator.keras"


@st.cache_resource
def load_model():
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            "The trained model file is missing. "
            "Check that temple_colorization_generator.keras "
            "is available to the application."
        )

    return tf.keras.models.load_model(
        str(MODEL_PATH),
        compile=False
    )


def colorize_temple(image, generator):
    original = image.convert("RGB")
    width, height = original.size

    gray = original.convert("L").resize(
        (256, 256),
        Image.Resampling.LANCZOS
    )

    gray_array = np.asarray(gray, dtype=np.float32)
    normalized = (gray_array / 127.5) - 1.0
    normalized = normalized.reshape(1, 256, 256, 1)

    prediction = generator.predict(normalized, verbose=0)[0]
    prediction = np.clip(
        (prediction + 1.0) * 127.5,
        0,
        255
    ).astype(np.uint8)

    generated = Image.fromarray(prediction).convert("RGB")
    generated = generated.resize(
        (width, height),
        Image.Resampling.LANCZOS
    )

    original_gray = np.asarray(
        original.convert("L"),
        dtype=np.uint8
    )

    generated_array = np.asarray(generated)
    lab = cv2.cvtColor(generated_array, cv2.COLOR_RGB2LAB)
    lab[:, :, 0] = original_gray

    result_array = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)
    result = Image.fromarray(result_array)

    result = result.filter(
        ImageFilter.UnsharpMask(
            radius=1.5,
            percent=110,
            threshold=3
        )
    )

    return ImageEnhance.Contrast(result).enhance(1.05)


uploaded_file = st.file_uploader(
    "Choose a temple image",
    type=["png", "jpg", "jpeg", "webp"]
)

if uploaded_file is not None:
    try:
        input_image = Image.open(uploaded_file).convert("RGB")

        st.subheader("Original Image")
        st.image(input_image, use_container_width=True)

        if st.button("Colorize Image", type="primary"):
            try:
                with st.spinner("Loading model and colorizing image..."):
                    generator = load_model()
                    output_image = colorize_temple(
                        input_image,
                        generator
                    )

                st.subheader("Colorized Image")
                st.image(output_image, use_container_width=True)

                buffer = io.BytesIO()
                output_image.save(buffer, format="PNG")

                st.download_button(
                    "Download Colorized Image",
                    data=buffer.getvalue(),
                    file_name="colorized_temple.png",
                    mime="image/png"
                )

            except Exception as error:
                st.error("Unable to colorize the image.")
                st.code(str(error))

    except Exception as error:
        st.error(f"Unable to open image: {error}")

st.divider()
st.caption("Academic Project | Pix2Pix GAN | TensorFlow | Streamlit")
