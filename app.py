import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import load_model
import numpy as np

import streamlit as st

model = load_model('C:\\Users\\luqma\\Desktop\\Personal\\Projects\\Image-Classification\\fruits_vegetables_model.keras')

data_cat = ['apple',
 'banana',
 'beetroot',
 'bell pepper',
 'cabbage',
 'capsicum',
 'carrot',
 'cauliflower',
 'chilli pepper',
 'corn',
 'cucumber',
 'eggplant',
 'garlic',
 'ginger',
 'grapes',
 'jalepeno',
 'kiwi',
 'lemon',
 'lettuce',
 'mango',
 'onion',
 'orange',
 'paprika',
 'pear',
 'peas',
 'sweetcorn',
 'sweetpotato',
 'tomato',
 'turnip',
 'watermelon']

img_height = 180
img_width = 180
image = 'Corn.jpg'

st.header('Fruit and Vegetable Classification')


image_load = tf.keras.utils.load_img(image, target_size = (img_width, img_height))
img_arr = tf.keras.utils.array_to_img(image_load)
img_bat = tf.expand_dims(img_arr, axis=0)

predict = model.predict(img_bat)

score = tf.nn.softmax(predict)

st.image(image_load)

st.write('Veg/Fruit in Image is {} with accuracy of {:0.2f}'.format(data_cat[np.argmax(score)], 100 * np.max(score)))