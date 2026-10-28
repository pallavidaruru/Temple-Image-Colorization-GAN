
import streamlit as st
import tensorflow as tf
import numpy as np
import cv2
import io

from PIL import Image, ImageFilter, ImageEnhance
from pathlib import Path

# ---------------- PAGE SETTINGS ----------------

st.set_page_config(
    page_title="Temple Image Colorization GAN",
    page_icon="🏛️",
    layout="wide"
)

st.title("🏛️ Temple Image Colorization and Restoration Using GAN")

st.write(
    "Upload an old or grayscale temple photograph. "
    "The trained Pix2Pix GAN generates a colorized result."
)

st.info(
    "For the best test, upload a genuinely grayscale temple photograph. "
    "A photo that is already colored may show little visible difference."
)

# ---------------- MODEL SETTINGS ----------------

MODEL_PATH = (
    Path(__file__).parent /
    "temple_colorization_generator.keras"
)


@st.cache_resource
def load_generator():
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            "The trained model is missing from the Streamlit server. "
            "Check that temple_colorization_generator.keras is "
            "available to the application."
        )

    return tf.keras.models.load_model(
        str(MODEL_PATH),
        compile=False
    )


# ---------------- COLORIZATION FUNCTION ----------------

def colorize_temple(image, generator, color_strength):
    original = image.convert("RGB")
    original_width, original_height = original.size

    # Convert input to grayscale for the GAN
    grayscale = original.convert("L")

    resized_gray = grayscale.resize(
        (256, 256),
        Image.Resampling.LANCZOS
    )

    gray_array = np.asarray(
        resized_gray,
        dtype=np.float32
    )

    # Normalize image to the range expected by the GAN
    normalized = (gray_array / 127.5) - 1.0
    normalized = normalized.reshape(1, 256, 256, 1)

    # Generate predicted colors
    prediction = generator.predict(
        normalized,
        verbose=0
    )[0]

    prediction = np.clip(
        (prediction + 1.0) * 127.5,
        0,
        255
    ).astype(np.uint8)

    generated = Image.fromarray(prediction).convert("RGB")

    generated = generated.resize(
        (original_width, original_height),
        Image.Resampling.LANCZOS
    )

    # Preserve brightness and structure from the original image
    original_gray = np.asarray(
        grayscale,
        dtype=np.uint8
    )

    generated_array = np.asarray(generated)

    lab_image = cv2.cvtColor(
        generated_array,
        cv2.COLOR_RGB2LAB
    )

    lab_image[:, :, 0] = original_gray

    restored_array = cv2.cvtColor(
        lab_image,
        cv2.COLOR_LAB2RGB
    )

    result = Image.fromarray(restored_array)

    # Increase color visibility
    result = ImageEnhance.Color(result).enhance(
        float(color_strength)
    )

    # Improve contrast and sharpness
    result = ImageEnhance.Contrast(result).enhance(1.10)

    result = result.filter(
        ImageFilter.UnsharpMask(
            radius=1.5,
            percent=110,
            threshold=3
        )
    )

    return result


# ---------------- USER INTERFACE ----------------

uploaded_file = st.file_uploader(
    "Choose a temple photograph",
    type=["png", "jpg", "jpeg", "webp"]
)

color_strength = st.slider(
    "Color intensity",
    min_value=1.0,
    max_value=3.0,
    value=2.0,
    step=0.25,
    help="Increase this value to make the predicted colors more vivid."
)

if uploaded_file is not None:

    try:
        input_image = Image.open(uploaded_file).convert("RGB")

        st.subheader("Original Image")

        st.image(
            input_image,
            caption="Uploaded photograph",
            use_container_width=True
        )

        if st.button("Colorize Image", type="primary"):

            try:
                with st.spinner(
                    "Loading the model and generating colors..."
                ):
                    generator = load_generator()

                    output_image = colorize_temple(
                        input_image,
                        generator,
                        color_strength
                    )

                st.success("Colorization completed!")

                st.subheader("Original vs Colorized Result")

                col1, col2 = st.columns(2)

                with col1:
                    st.image(
                        input_image,
                        caption="Original image",
                        use_container_width=True
                    )

                with col2:
                    st.image(
                        output_image,
                        caption="GAN colorized image",
                        use_container_width=True
                    )

                # Prepare image for download
                image_buffer = io.BytesIO()
                output_image.save(
                    image_buffer,
                    format="PNG"
                )

                st.download_button(
                    label="Download Colorized Image",
                    data=image_buffer.getvalue(),
                    file_name="colorized_temple.png",
                    mime="image/png"
                )

            except Exception as error:
                st.error("Unable to colorize the image.")
                st.code(str(error))

    except Exception as error:
        st.error("Unable to open the uploaded image.")
        st.code(str(error))

st.divider()

st.caption(
    "Academic Project | Pix2Pix GAN | TensorFlow | Streamlit"
)
