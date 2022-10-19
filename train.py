from tensorflow import keras
import tensorflow as tf
from utils.utils import get_run_logdir
import os
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from model.callbacks import *
from utils.utils import DataGenerator
import numpy as np
import sklearn
import io

def train(model,
    epochs = 500,
    optimizer = keras.optimizers.Adam(), 
    dir_name = None,
    model_name = "model.h5",
    batch_size=16,
    patience = 10):

    if dir_name == None:
        dir_name = get_run_logdir()
        os.mkdir(dir_name)
    
    model.compile(loss="categorical_crossentropy", optimizer = optimizer, metrics=['accuracy'])
    train_generator, validation_generator = DataGenerator(batch_size=batch_size)

    history = model.fit(train_generator, epochs=epochs, steps_per_epoch=56, 
                    validation_data = validation_generator, validation_steps=12, callbacks=callbacks_list(dir_name, model_name, patience))

    return history

