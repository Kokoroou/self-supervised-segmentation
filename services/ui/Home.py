import streamlit as st


def homepage():
    """
    Build homepage for a multi-page app.
    """
    st.title("Welcome to the Homepage!")

    st.subheader("Masked Autoencoder")
    st.image("https://raw.githubusercontent.com/Kokoroou/self-supervised-segmentation/"
             "main/services/ui/image/demo_autoencoder.png")
    st.caption("Visualize the output of the autoencoder. The left image is the original image, the middle image "
               "is the masked image, and the right image is the reconstructed image.")

    st.subheader("Semantic Segmentation")
    st.image("https://raw.githubusercontent.com/Kokoroou/self-supervised-segmentation/"
             "main/services/ui/image/demo_segmentation.png")
    st.caption("Visualize the output of the semantic segmentation. The left image is the original image, the right "
               "image is the segmented image.")


homepage()
