import streamlit as st


def homepage():
    """
    Build homepage for a multi-page app.
    """
    st.title("Home")

    st.write(
        """
        ## Welcome to the Homepage!
        
        ### This is a demo app to process image with neural network models. 
        ### To navigate to the other pages, please use the sidebar.
        
        ### The following are the list of pages:
        - Home
        - Masked Autoencoder
        - Self-Supervised Semantic Segmentation
        
        ### The following are the list of models:
        - Masked Autoencoder with Vision Transformer
        - Self-Supervised Semantic Segmentation with Vision Transformer
        
        ### The following are the list of datasets:
        - Polyp Dataset
        """
    )


homepage()
