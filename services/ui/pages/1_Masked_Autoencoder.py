import time
from pathlib import Path

import cv2
import imutils as imutils
import numpy as np
import streamlit as st
from PIL import Image

# from services.streamlit.utils.encoder import MAE
from ui.models.autoencoder.autoencoder_class import Autoencoder


# Define a function to perform the inference using the selected model
def perform_inference(image, model, masking_percentage):
    current_dir = Path(__file__).parent.resolve()
    main_dir = current_dir.parent.parent.parent
    checkpoint_path = main_dir / 'research' / 'mae' / 'output_dir' / model

    model = MAE('mae_vit_base_patch16', str(checkpoint_path))

    masked, reconstructed, paste = model(image, masking_percentage / 100)

    return masked, reconstructed, paste


def main():
    # Create a sidebar
    st.sidebar.title("Model Configuration")

    current_dir = Path(__file__).parent.resolve()
    checkpoint_dir = current_dir.parent.parent.parent / 'research' / 'mae' / 'output_dir'
    checkpoint_list = [checkpoint_path.name for checkpoint_path in checkpoint_dir.glob('*.pth')]

    # Add components to the sidebar
    # selected_model = st.sidebar.selectbox("Select Model", checkpoint_list)
    selected_model = st.sidebar.selectbox("Select Model",
                                          ["facebook/vit-mae-base", "facebook/vit-mae-large", "facebook/vit-mae-huge"]
                                          + checkpoint_list)

    with st.spinner("Loading Model..."):
        # Load the model
        model = Autoencoder(model=selected_model)

    masking_percentage = st.sidebar.slider("Masking Percentage", 0, 100, 75)

    # Create the main content area
    st.title("Masked Autoencoder Model Inference")

    # Add a file uploader to upload the image
    uploaded_file = st.file_uploader("Upload Image", type=["jpg", "jpeg", "png"])

    # Check if an image has been uploaded
    if uploaded_file is not None:
        # Display the uploaded image
        st.image(uploaded_file, caption=uploaded_file.name, use_column_width=True)

        # Add a button to start the inference
        if st.button("Start Inference"):
            with st.spinner("Inferencing..."):
                # Perform the inference using the selected model
                start_time = time.time()

                # Load image and convert to BGR
                image = Image.open(uploaded_file)
                original_width, original_height = image.size

                reconstructed = model.inference(image)
                reconstructed = cv2.resize(reconstructed, (original_width, original_height))

                end_time = time.time()

                # Display the inference result
                st.subheader("Inference Result")

                # st.image(masked, caption="Masked Image", use_column_width=True)
                st.image(reconstructed, caption="Reconstructed Image", use_column_width=True)
                # st.image(paste, caption="Paste Image", use_column_width=True)

                # Display the inference time
                st.write(f"Inference Time: **{end_time - start_time:.2f}** seconds")


main()
