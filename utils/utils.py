import os
import pandas as pd
import shutil
from shutil import copyfile
import numpy as np
from sklearn.model_selection import train_test_split
import tensorflow as tf
import keras_preprocessing
from keras_preprocessing import image
from keras_preprocessing.image import ImageDataGenerator


def train_valid_split():
    # load data
    data_path = './datasets/plant-pathology-2020-fgvc7/'

    train_set = pd.read_csv(data_path + 'train.csv', index_col = 0)

    train_set, valid_set = train_test_split(train_set, 
                            test_size=0.3,
                            stratify=train_set[['healthy', 'multiple_diseases', 'rust', 'scab']],
                            random_state=42)


    if os.path.exists(data_path+'temp/'):
        shutil.rmtree(data_path+'temp/')

    os.mkdir(data_path+'temp/')

    # train directory
    os.mkdir(data_path+'temp/train')
    os.mkdir(data_path+'temp/train/healthy')
    os.mkdir(data_path+'temp/train/multiple_diseases')
    os.mkdir(data_path+'temp/train/rust')
    os.mkdir(data_path+'temp/train/scab')

    # validation directory
    os.mkdir(data_path+'temp/valid')
    os.mkdir(data_path+'temp/valid/healthy')
    os.mkdir(data_path+'temp/valid/multiple_diseases')
    os.mkdir(data_path+'temp/valid/rust')
    os.mkdir(data_path+'temp/valid/scab')

    SOURCE = data_path+'images/'

    TRAIN_DIR = data_path+'temp/train/'

    for index, data in train_set.iterrows():
        label = train_set.columns[np.argmax(data)]
        filepath = os.path.join(SOURCE, index + ".jpg")
        destination = os.path.join(TRAIN_DIR, label, index + ".jpg")
        copyfile(filepath, destination)
        
    print('Train')
    for subdir in os.listdir(TRAIN_DIR):
        print(subdir, len(os.listdir(os.path.join(TRAIN_DIR, subdir))))

    VALID_DIR = data_path+'temp/valid/'

    for index, data in valid_set.iterrows():
        label = valid_set.columns[np.argmax(data)]
        filepath = os.path.join(SOURCE, index + ".jpg")
        destination = os.path.join(VALID_DIR, label, index + ".jpg")
        copyfile(filepath, destination)

    print('Valid')
    for subdir in os.listdir(VALID_DIR):
        print(subdir, len(os.listdir(os.path.join(VALID_DIR, subdir))))

    return train_set, valid_set

def DataGenerator(batch_size = 32) :
    data_path = 'datasets/plant-pathology-2020-fgvc7/'
    TRAIN_DIR = data_path+'temp/train/'
    VALID_DIR = data_path+'temp/valid/'

    training_datagen = ImageDataGenerator(rescale = 1./255,
                                        rotation_range=30,
                                        width_shift_range=0.2,
                                        height_shift_range=0.2,
                                        shear_range=0.2,
                                        zoom_range=0.2,
                                        horizontal_flip=True,
                                        fill_mode='nearest')

    validation_datagen = ImageDataGenerator(rescale = 1./255)

    train_generator = training_datagen.flow_from_directory(TRAIN_DIR, target_size=(224,224), class_mode='categorical', batch_size=batch_size)
    validation_generator = validation_datagen.flow_from_directory(VALID_DIR, target_size=(224,224), class_mode='categorical', batch_size=batch_size)

    return train_generator, validation_generator

def get_run_logdir(log_dir='./logs'):
    import time
    run_id = time.strftime("run_%Y_%m_%d-%H_%M_%S")
    return os.path.join(log_dir, run_id) 



