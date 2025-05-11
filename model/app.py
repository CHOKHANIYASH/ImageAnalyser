import numpy as np
import pandas as pd
import tensorflow as tf
from tqdm import tqdm
from tensorflow.keras.preprocessing.image import ImageDataGenerator, load_img, img_to_array
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.utils import Sequence
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Conv2D, MaxPooling2D, GlobalAveragePooling2D, Activation, Dropout, Flatten, Dense, Input, Layer
from tensorflow.keras.layers import Embedding, LSTM, add, Concatenate, Reshape, concatenate, Bidirectional
from tensorflow.keras.applications import VGG16, ResNet50, DenseNet201
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.applications.densenet import preprocess_input
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
import pickle
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.densenet import preprocess_input
from tensorflow.keras.models import load_model
from fastapi import FastAPI, HTTPException
import requests
from io import BytesIO
from fastapi import FastAPI, HTTPException
from tensorflow.keras.preprocessing import image
from PIL import Image
from fastapi.middleware.cors import CORSMiddleware
from mangum import Mangum
from pydantic import BaseModel

# Load your saved model
model_path = "model_captioning.keras"  # replace with your model file path
caption_model = load_model(model_path)
feature_model = load_model("denseNet201.keras")
with open('tokenizer_flickr8k.pkl', 'rb') as file:
    tokenizer = pickle.load(file)
vocab_size = len(tokenizer.word_index) + 1
max_length = 34

# model = DenseNet201()
# feature_model = Model(inputs=model.input, outputs=model.layers[-2].output)

def idx_to_word(integer,tokenizer):
    for word, index in tokenizer.word_index.items():
        if index==integer:
            return word
    return None

def extract_features(img):
    img = img_to_array(img)
    img = img/255.
    img = np.expand_dims(img,axis=0)
    feature = feature_model.predict(img, verbose=0)
    return feature

def predict_caption(model, image, tokenizer, max_length):

    feature = extract_features(image)
    print(feature)
    in_text = "startseq"
    for i in range(max_length):
        sequence = tokenizer.texts_to_sequences([in_text])[0]
        sequence = pad_sequences([sequence], max_length)

        y_pred = model.predict([feature,sequence])
        y_pred = np.argmax(y_pred)

        word = idx_to_word(y_pred, tokenizer)

        if word is None:
            break

        in_text+= " " + word

        if word == 'endseq':
            break
    print(in_text)
    return in_text

app = FastAPI(title="Image Analysis API", description="API to analyze images and generate captions", version="0.1")
handler = Mangum(app)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
@app.get("/")
async def main():
    return {"message": "Hello World"}

@app.post("/predict/caption")
async def predict(image_url: str):
    try:
        # Fetch the image from the URL
        response = requests.get(image_url)
        if response.status_code != 200:
            raise HTTPException(status_code=400, detail="Failed to retrieve image from URL")

        # Load the image from the response content
        img = Image.open(BytesIO(response.content))

        # Resize the image to match the input size expected by your model
        img = img.resize((224, 224))
        
        # Use the predict_caption function to generate a caption
        caption = predict_caption(caption_model, img, tokenizer, max_length)
        
        return {"caption": caption}
    
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

model_path = "cnn_20_epochs_colab.keras"  # replace with your model file path
tagging_model = load_model(model_path)

model_path = "voc_densenet_model_densenet.keras"  # replace with your model file path
tagging_model_multi_label = load_model(model_path)

@app.post("/predict/tags")
async def predict(image_url: str):
    try:
        # Define the labels
        labels = ['airplane', 'automobile', 'bird', 'cat', 'deer',
                  'dog', 'frog', 'horse', 'ship', 'truck']

        # Fetch the image from the URL
        response = requests.get(image_url)
        if response.status_code != 200:
            raise HTTPException(status_code=400, detail="Failed to retrieve image from URL")

        # Load the image from the response content
        img = Image.open(BytesIO(response.content))

        # Resize the image to match the input size expected by your model
        img = img.resize((32, 32))

        # Convert the image to the format expected by Keras
        img = image.img_to_array(img)
        img = img/255.
        img = np.expand_dims(img,axis=0)

        # Predict the label using the tagging model
        predictions = tagging_model.predict(img)
        tag_index = np.argmax(predictions)
        tag = labels[tag_index]

        return {"Tag": tag}

    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.post("/predict/tags/multiple")
async def predict(image_url: str):
    try:
        # Define the labels
        labels = [
                    "aeroplane", "bicycle", "bird", "boat", "bottle",
                    "bus", "car", "cat", "chair", "cow", "diningtable",
                    "dog", "horse", "motorbike", "person", "pottedplant",
                    "sheep", "sofa", "train", "tvmonitor"
                ]

        # Fetch the image from the URL
        response = requests.get(image_url)
        if response.status_code != 200:
            raise HTTPException(status_code=400, detail="Failed to retrieve image from URL")

        # Load the image from the response content
        img = Image.open(BytesIO(response.content))

        # Resize the image to match the input size expected by your model
        img = img.resize((224, 224))

        # Convert the image to the format expected by Keras
        img = image.img_to_array(img)
        img = preprocess_input(img)  # Use DenseNet preprocess
        img = np.expand_dims(img, axis=0)

        # Predict the label using the tagging model
        predictions = tagging_model_multi_label.predict(img)  # shape: (1, num_classes)
        threshold = 0.5  # You can adjust this based on validation performance

        # Get all indices where prediction is above the threshold
        tag_indices = np.where(predictions[0] >= threshold)[0]

        # Map indices to class names
        tags = [labels[i] for i in tag_indices]
        print(tags)
        return {"Tags": tags}


    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

CLASSES = [
    "aeroplane", "bicycle", "bird", "boat", "bottle",
    "bus", "car", "cat", "chair", "cow", "diningtable",
    "dog", "horse", "motorbike", "person", "pottedplant",
    "sheep", "sofa", "train", "tvmonitor"
]
def preprocess_image_from_url(image_url, image_size=(224, 224)):
    response = requests.get(image_url)
    image = Image.open(BytesIO(response.content))
    # img = load_img(img_path, target_size=image_size)
    img = image.resize((224, 224))
    img_array = img_to_array(img)
    img_array = preprocess_input(img_array)
    return img_array, img

def decode_prediction(pred_vector, threshold=0.5):
    indices = np.where(pred_vector >= threshold)[0]
    return [CLASSES[i] for i in indices]
    
@app.post("/predict/test")
async def predict_image(image_url: str):
    try:
        # Preprocess
        image_array, original_image = preprocess_image_from_url(image_url)
        prediction = tagging_model_multi_label.predict(np.expand_dims(image_array, axis=0))[0]
        predicted_labels = decode_prediction(prediction)
        print(predicted_labels)
        return {"Tags": predicted_labels}
    
    except Exception as e:
        print(e)
        raise HTTPException(status_code=400, detail=str(e))