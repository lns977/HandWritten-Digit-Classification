import streamlit as st
from PIL import Image, ImageDraw
import numpy as np
import pickle

def preprocess_image (image):
    image = image.resize((28,28)).convert('L')

    image = np.array(image)

    image = image/255.0

    image = image.reshape(28,28,1)
    return image

def load_model():
    with open("D:\B-tech\Machine-Learning\Digit_Recognisation\model.pkl",'rb')as f:
        model = pickle.load(f)
    return model
def predict_digit(self):
        model = load_model()

        image = preprocess_image(image)

        prediction = model.predict(image)

        predict_digit=np.argmax(prediction)
        return predict_digit
def main():
    st.title("Handwritten Digit Recognisation")

    uploaded_image = st.file_uploader("Upload Image",type=['png','jpg','jpeg'])

    if uploaded_image is not None:
        image = Image.open(uploaded_image)
        st.image(image,caption='Uploaded Image',use_column_width=True)
        if st.button('Predict'):
            predicted_digit = predict_digit(image)
            st.write(f'Predicted Digit: {predicted_digit}')

if __name__ =='__main__':
    main()
