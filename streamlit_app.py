import streamlit as st
import tensorflow as tf
import numpy as np
import logging

# Configure logging
logging.basicConfig(filename='streamlit_app.log', level=logging.DEBUG)

#Tensorflow Model Prediction
def model_prediction(test_image):
    try:
        model = tf.keras.models.load_model("test_final_model.keras")
        logging.info("Model loaded successfully")
        
        image = tf.keras.preprocessing.image.load_img(test_image, target_size=(160,160))
        input_arr = tf.keras.preprocessing.image.img_to_array(image)
        input_arr = np.array([input_arr]) #convert single image to batch
        
        logging.info("Input array shape: {}".format(input_arr.shape))
        
        predictions = model.predict(input_arr)
        logging.info("Predictions: {}".format(predictions))
        
        result_index = np.argmax(predictions)
        logging.info("Result index: {}".format(result_index))
        
        return result_index #return index of max element
    
    except Exception as e:
        logging.error("Error occurred: {}".format(str(e)))
        return None

#Sidebar
st.sidebar.title("Dashboard")
app_mode = st.sidebar.selectbox("Select Page",["Home","About Project","Prediction"])

#Main Page
if app_mode == "Home":
    st.header("Florist aid flower recognition webapp")
    image_path = "home_img.jpg"
    st.image(image_path)

#About Project
elif app_mode == "About Project":
    st.header("About Project")
    st.subheader("About Dataset")
    st.text("This dataset contains images of the following flowers:")
    st.code("astilbe ,bellflower ,black_eyed_susan ,calendula ,california_poppy ,carnation ,common_daisy ,coreopsis ,dandelion ,iris ,rose ,sunflower ,tulip ,water_lily")
    st.subheader("Content")
    st.text("This dataset was split into three folders:")
    st.text("1. train (9542 images split between the 14 flowers)")
    st.text("2. test (2052 images split between the 14 flowers)")
    st.text("3. validation (2048 images split between the 14 flowers)")

#Prediction Page
elif app_mode == "Prediction":
    st.header("Model Prediction")
    test_image = st.file_uploader("Choose an Image:")
    if st.button("Show Image"):
        st.image(test_image, width=4, use_column_width=True)
        
    #Predict button
    if st.button("Predict"):
        st.write("Our Prediction")
        result_index = model_prediction(test_image)
        
        if result_index is not None:
            # Reading Labels
            with open("labels.txt") as f:
                content = f.readlines()
            label = [i.strip() for i in content]
            
            st.success("Model is Predicting it's a {}".format(label[result_index]))
        else:
            st.error("Failed to make prediction. Please check the logs for details.")
