## Library imports
import streamlit as st
from PIL import Image
import numpy as np

## Local imports
from model.run_inference import run_model_inference, prepare_image
from model.model_config import DEVICE


def new_image_enhancer_UI(model):
    """
    The main UI function to display the recommended recipes page
    """

    ## Pager subheader
    st.divider()
    st.subheader("Choose a Low Resolution X-Ray image ...")

    ## Display the warning
    st.caption("⚠️ Check Solution Risks in the Project Home Page")

    ## Image uploader
    image_upload = st.file_uploader("Upload A Low Resolution X-Ray Image (Png)", type="png")

    ## If image is uploaded
    if image_upload is not None:
        ## Load image
        input_image = Image.open(image_upload).convert('RGB')

        ## Prepare the image for model
        input_image_for_model = prepare_image(image_upload, is_hr_image=True)

        ## Run the model
        output_image = run_model_inference(input_image_for_model, model, device=DEVICE)

        ## Place holder container for the results
        results = st.empty()

        ## Display the results
        with results.container():
            col1, col2 = st.columns(2)

            ## Display the low resolution image
            with col1:
                st.subheader("Low Resolution Input Image")
                st.image(input_image_for_model, use_column_width=True)

            ## Display the super resolution image
            with col2:
                st.subheader("Super Resolution Output Image")
                st.image(output_image, use_column_width=True)


def clahe_example_UI():
    """
    The UI function to display the CLAHE example
    """
    st.title("CLAHE Image Enhancement Example")
    uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        image = Image.open(uploaded_file).convert('L')  # Convert to grayscale
        img_array = np.array(image)

        clahe_img = clahe(img_array, 2, 0, 0)

        st.image([img_array, clahe_img], caption=["Original Image", "CLAHE Enhanced Image"], use_column_width=True)