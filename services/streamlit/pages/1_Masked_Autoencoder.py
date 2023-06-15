import time

import streamlit as st


# Define a function to perform the inference using the selected model
def perform_inference(image, model, masking_percentage):
    # Your inference code here
    # Replace this with your actual inference code using the selected model
    time.sleep(2)  # Simulating inference time
    return image


# Create a sidebar
st.sidebar.title("Model Configuration")

# Add components to the sidebar
selected_model = st.sidebar.selectbox("Select Model", ["Model 1", "Model 2", "Model 3"])
masking_percentage = st.sidebar.slider("Masking Percentage", 0, 100, 75)

# Create the main content area
st.title("Masked Autoencoder Model Inference")

# Add a file uploader to upload the image
uploaded_file = st.file_uploader("Upload Image", type=["jpg", "jpeg", "png"])

# Check if an image has been uploaded
if uploaded_file is not None:
    # Display the uploaded image
    image = uploaded_file.read()
    img = st.image(uploaded_file, caption=uploaded_file.name, use_column_width=True)

    # Add a button to start the inference
    if st.button("Start Inference"):
        # Perform the inference using the selected model
        start_time = time.time()
        result = perform_inference(image, selected_model, masking_percentage)
        end_time = time.time()

        # Display the inference result
        st.subheader("Inference Result")
        st.image(result, caption="Inference Result", use_column_width=True)

        # Display the inference time
        st.write(f"Inference Time: **{end_time - start_time:.2f}** seconds")
