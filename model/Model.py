import tensorflow as tf
from tensorflow import keras
import pandas as pd
import os
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import train_test_split
import random
import matplotlib
from PIL import Image


def Model(
    model = tf.keras.applications.ResNet50,
    ):

    base = model(weights = "imagenet", include_top = False)
    avg = keras.layers.GlobalAveragePooling2D()(base.output)
    output = keras.layers.Dense(4, activation = "softmax")(avg)
    model = keras.Model(inputs = base.input, outputs=output)

    for layer in base.layers:
             layer.trainable = False

    print('Model complete!')
    return model

