import os, json
import numpy as np
import pandas as pd
import streamlit as st
import tensorflow as tf
from PIL import Image
from pathlib import Path

st.set_page_config(page_title='Bharatanatyam Mudra Recognition', page_icon='🎭', layout='wide')
BASE = Path(__file__).parent
MODEL_FILE = BASE / 'model.keras'
LABEL_FILE = BASE / 'mudra_class_names.json'

@st.cache_resource
def get_model():
    return tf.keras.models.load_model(str(MODEL_FILE), compile=False)

try:
    model = get_model()
    names = json.loads(LABEL_FILE.read_text())
    if int(model.output_shape[-1]) != len(names):
        raise ValueError('Model output classes and label count do not match')
except Exception as exc:
    st.error(f'Model loading failed: {exc}')
    st.stop()

st.sidebar.title('🎭 Mudra Recognition')
page = st.sidebar.radio('Navigate', ['Home', 'Predict', 'About'])

if page == 'Home':
    st.title('Bharatanatyam Mudra Recognition')
    st.write('An AI-based learning demonstration that classifies images of five Bharatanatyam hand mudras using a custom CNN.')
    st.subheader('Supported mudras')
    st.write(', '.join(names))
    st.info('Go to Predict, upload a clear hand image, and select Predict Mudra.')
    st.write('Workflow: Upload → RGB resize (224 × 224) → Normalize → CNN → Class probabilities')

elif page == 'Predict':
    st.title('Predict a Bharatanatyam Mudra')
    upload = st.file_uploader('Upload JPG or PNG image', type=['jpg','jpeg','png'])
    if upload is not None:
        try:
            image = Image.open(upload).convert('RGB')
            st.image(image, caption='Uploaded hand gesture', width=340)
            if st.button('Predict Mudra', type='primary'):
                image = image.resize((224,224))
                arr = np.asarray(image, dtype=np.float32) / 255.0
                arr = arr[None, ...]
                probabilities = model.predict(arr, verbose=0)[0]
                probabilities = np.asarray(probabilities, dtype=float)
                idx = int(np.argmax(probabilities))
                st.success(f'Predicted mudra: {names[idx]}')
                st.metric('Model confidence', f'{probabilities[idx]*100:.2f}%')
                df = pd.DataFrame({'Mudra': names, 'Probability (%)': probabilities*100})
                st.subheader('Probability distribution across all classes')
                st.bar_chart(df.set_index('Mudra'))
                st.dataframe(df.sort_values('Probability (%)', ascending=False), hide_index=True)
                st.caption('Softmax confidence is not a guarantee that the prediction is correct.')
        except Exception as exc:
            st.error(f'Could not process image: {exc}')

else:
    st.title('About the Project')
    st.write('Dataset: 2,000 images; training: 1,400; validation: 300; testing: 300.')
    st.write('Classes:', ', '.join(names))
    st.write('Preprocessing: RGB conversion, 224 × 224 resizing, pixel normalization to [0, 1].')
    st.write('Training augmentation: small rotations, translations, zoom, and contrast changes.')
    st.write('Model: custom CNN with convolution, batch normalization, pooling, dropout, and softmax.')
    st.write(f'Model parameters: {model.count_params():,}')
    st.write('Frameworks: TensorFlow/Keras and Streamlit.')
    st.info('Educational demonstration; performance depends on the training dataset and image quality.')
