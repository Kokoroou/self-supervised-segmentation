import time
from pathlib import Path

import cv2
import streamlit as st
from PIL import Image

from ui.models.segmentation.segmentation_class import Segmentor


def load_model(selected_model):
    with st.spinner("Loading Model..."):
        # Load the model
        model = Segmentor(model=selected_model)

    return model


def main():
    # Create a sidebar
    st.sidebar.title("Model Configuration")

    current_dir = Path(__file__).parent.resolve()
    checkpoint_dir = current_dir.parent / "models" / "segmentation" / "vit_seg" / "checkpoint"

    default_choice = ["--Select--"]
    local_checkpoint_list = [checkpoint_path.name for checkpoint_path in checkpoint_dir.glob('*')]
    online_checkpoint_list = ["nvidia/segformer-b0-finetuned-ade-512-512",
                              "nvidia/segformer-b5-finetuned-ade-640-640"]

    # Add components to the sidebar
    selected_model = st.sidebar.selectbox("Select Model", default_choice +
                                          online_checkpoint_list + local_checkpoint_list)

    if selected_model not in default_choice:
        if selected_model in local_checkpoint_list:
            selected_model = str(checkpoint_dir / selected_model)

        # Check if the model has been loaded
        if "model" not in st.session_state:
            # Load the model
            st.session_state.selected_model = selected_model
            st.session_state.model = load_model(selected_model)
        else:
            # Check if the selected model has changed
            if st.session_state.selected_model != selected_model:
                # Load the model
                st.session_state.selected_model = selected_model
                st.session_state.model = load_model(selected_model)

    # Create the main content area
    st.title("Semantic Segmentation Model Inference")

    # Add a file uploader to upload the image
    uploaded_file = st.file_uploader("Upload Image", type=["jpg", "jpeg", "png"])

    # Check if an image has been uploaded
    if uploaded_file is not None:
        # Display the uploaded image
        st.image(uploaded_file, caption=uploaded_file.name, use_column_width=True)

        if selected_model in default_choice:
            st.warning("Please select a model to perform inference.")

        else:
            # Perform the inference using the selected model
            with st.spinner("Inferencing..."):
                start_time = time.time()

                # Load image and convert to BGR
                image = Image.open(uploaded_file)
                original_width, original_height = image.size

                # Mask the image and reconstruct it with model Masked Autoencoder
                result = st.session_state.model.inference(image)

                # Get the segmentation and pasted image from the result
                segment, pasted = result.segment, result.pasted

                end_time = time.time()

            # Display the inference result
            st.subheader("Inference Result")
            st.image(segment, caption="Segment Image", use_column_width=True)
            st.image(pasted, caption="Pasted Image", use_column_width=True)

            # Display the inference time
            st.write(f"Inference Time: **{end_time - start_time:.2f}** seconds")


main()
