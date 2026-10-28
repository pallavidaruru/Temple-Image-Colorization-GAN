
import os
import numpy as np
import gradio as gr
import tensorflow as tf
import cv2

from PIL import Image, ImageFilter, ImageEnhance

# --------------------------------------------------
# 1. MODEL PATH
# --------------------------------------------------

MODEL_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "temple_colorization_generator.keras"
)

# --------------------------------------------------
# 2. LOAD TRAINED GAN MODEL
# --------------------------------------------------

generator = None
model_error = None

try:
    if os.path.exists(MODEL_PATH):
        generator = tf.keras.models.load_model(
            MODEL_PATH,
            compile=False
        )
        print("GAN model loaded successfully!")
    else:
        model_error = (
            "Model file not found. Place "
            "temple_colorization_generator.keras "
            "in the same folder as app.py."
        )
        print(model_error)

except Exception as error:
    model_error = str(error)
    print("Model loading error:", model_error)


# --------------------------------------------------
# 3. TEMPLE IMAGE COLORIZATION
# --------------------------------------------------

def colorize_temple(image):

    if image is None:
        raise gr.Error("Please upload a temple image.")

    if generator is None:
        raise gr.Error(
            "GAN model could not be loaded: "
            + str(model_error)
        )

    try:
        # Preserve the uploaded image
        original = image.convert("RGB")
        original_width, original_height = original.size

        # Convert image to grayscale
        gray = original.convert("L")

        # Resize input to model size
        model_input = gray.resize(
            (256, 256),
            Image.Resampling.LANCZOS
        )

        gray_array = np.asarray(
            model_input,
            dtype=np.float32
        )

        # Normalize input to [-1, 1]
        normalized = (
            gray_array / 127.5
        ) - 1.0

        # Shape: (1, 256, 256, 1)
        normalized = normalized[np.newaxis, ..., np.newaxis]

        # Generate color prediction
        prediction = generator.predict(
            normalized,
            verbose=0
        )[0]

        # Convert model output to [0, 255]
        prediction = np.clip(
            (prediction + 1.0) * 127.5,
            0,
            255
        ).astype(np.uint8)

        # Convert prediction to RGB
        colorized = Image.fromarray(
            prediction,
            mode="RGB"
        )

        # Resize to uploaded image dimensions
        colorized = colorized.resize(
            (original_width, original_height),
            Image.Resampling.LANCZOS
        )

        # --------------------------------------------------
        # 4. PRESERVE ORIGINAL BRIGHTNESS AND STRUCTURE
        # --------------------------------------------------

        original_gray = np.asarray(
            gray,
            dtype=np.uint8
        )

        predicted_rgb = np.asarray(
            colorized,
            dtype=np.uint8
        )

        # Convert RGB to LAB color space
        lab_image = cv2.cvtColor(
            predicted_rgb,
            cv2.COLOR_RGB2LAB
        )

        # Match the original grayscale brightness.
        # This may preserve some visible temple details.
        lab_image[:, :, 0] = original_gray

        # Convert back to RGB
        enhanced_rgb = cv2.cvtColor(
            lab_image,
            cv2.COLOR_LAB2RGB
        )

        output = Image.fromarray(
            enhanced_rgb
        )

        # --------------------------------------------------
        # 5. MILD SHARPENING
        # --------------------------------------------------

        output = output.filter(
            ImageFilter.UnsharpMask(
                radius=1.5,
                percent=110,
                threshold=3
            )
        )

        # Slightly improve contrast
        output = ImageEnhance.Contrast(
            output
        ).enhance(1.05)

        # Return final colorized image
        return output

    except Exception as error:
        print("Colorization error:", error)
        raise gr.Error(
            "Could not colorize this image. Details: "
            + str(error)
        )


# --------------------------------------------------
# 6. GRADIO WEBSITE
# --------------------------------------------------

with gr.Blocks(
    title="Temple Image Colorization and Restoration"
) as demo:

    gr.Markdown(
        """
        # Temple Image Colorization and Restoration Using GAN

        Upload a grayscale or old temple photograph
        to generate a colorized image using a trained
        Pix2Pix Generative Adversarial Network (GAN).

        **How to use**
        1. Upload a temple photograph.
        2. Click **Colorize Image**.
        3. View and download the generated result.

        *Note: The predicted colors may not represent
        the historically accurate original colors.
        Sharpening can improve perceived clarity, but
        it cannot restore details the model did not learn.*
        """
    )

    with gr.Row():

        input_image = gr.Image(
            label="Upload Temple Image",
            type="pil"
        )

        output_image = gr.Image(
            label="Colorized Temple Image",
            type="pil"
        )

    colorize_button = gr.Button(
        "Colorize Image",
        variant="primary"
    )

    colorize_button.click(
        fn=colorize_temple,
        inputs=input_image,
        outputs=output_image
    )

    gr.Markdown(
        """
        **Technologies:** Python, TensorFlow, Pix2Pix GAN,
        OpenCV, Pillow, and Gradio.
        """
    )


# --------------------------------------------------
# 7. START WEBSITE
# --------------------------------------------------

if __name__ == "__main__":
    demo.launch()
