import streamlit as st


def homepage():
    """
    Build homepage for a multi-page app.
    """
    st.title("Welcome to the Homepage!")

    st.subheader("To test the autoencoder, go to the Autoencoder page.")
    # Need to change to our custom image later or add caption about origin of image
    st.image("https://mchromiak.github.io/articles/2021/Nov/14/Masked-Autoencoders-Are-Scalable-Vision-Learners/img/MaskedAE.png")

    st.subheader("To test the semantic segmentation, go to the Semantic Segmentation page.")
    # Need to change to our custom image later or add caption about origin of image
    st.image("https://theaisummer.com/static/8b58a02198e13d2e29a41b40e7c6a035/14b42/semseg.jpg")


homepage()
