from keras.layers import Conv2D, Conv2DTranspose, Dense, Flatten, Reshape, Activation, Lambda,Concatenate
from keras.models import Sequential, Model
from keras.layers.normalization import BatchNormalization
from keras.utils.vis_utils import plot_model
from keras.layers import *
import numpy as np
import tensorflow as tf


def CAE_3Dprinter(input_shape=(512,512,4)):
    model = Sequential()
    model.add(Conv2D(8, 3, strides= 2, padding='same', activation='relu', name='conv1', input_shape=input_shape))

    model.add(Conv2D(16, 3, strides= 2, padding='same', activation='relu', name='conv2'))

    model.add(Conv2D(32,  3, strides= 2, padding='same', activation='relu', name='conv3'))

    model.add(Conv2D(64, 3, strides=2, padding='same', activation='relu', name='conv4'))

    model.add(Conv2D(128, 3, strides=2, padding='same', activation='relu', name='conv5'))

    model.add(Conv2D(32, 3, strides=2, padding='same', activation='relu', name='conv6'))

    model.add(Flatten())  #2048
    model.add(Dense(units=1024, activation='relu'))

    model.add(BatchNormalization(name='embedding'))

    model.add(Dense(units=16*8*8, activation='relu'))

    model.add(Reshape((8, 8, 16)))

    model.add(Conv2DTranspose(128, 3, strides=2, padding='same', activation='relu',name='deconv6'))

    model.add(Conv2DTranspose(64, 3, strides=2, padding='same', activation='relu', name='deconv5'))

    model.add(Conv2DTranspose(32, 3, strides=2, padding='same', activation='relu',name='deconv4'))

    model.add(Conv2DTranspose(16, 3, strides=2, padding='same', activation='relu', name='deconv3'))

    model.add(Conv2DTranspose(8, 3, strides=2, padding='same', activation='relu',name='deconv2'))

    model.add(Conv2DTranspose(input_shape[2], 3, strides=2, padding='same', name='deconv1'))
    model.summary()
    return model

def CAE_CWRU(input_shape=(512,512,4)):
    model = Sequential()
    model.add(Conv2D(8, 3, strides= 2, padding='same', activation='relu', name='conv1', input_shape=input_shape))

    model.add(Conv2D(16, 3, strides= 2, padding='same', activation='relu', name='conv2'))

    model.add(Conv2D(64,  3, strides= 2, padding='same', activation='relu', name='conv3'))

    model.add(Conv2D(128, 3, strides=2, padding='same', activation='relu', name='conv4'))

    model.add(Conv2D(16, 3, strides=2, padding='same', activation='relu', name='conv7'))

    model.add(Flatten())  #16.16
    model.add(Dense(units=2048, activation='relu'))

    model.add(BatchNormalization(name='embedding'))

    model.add(Dense(units=16*16*16, activation='relu'))

    model.add(Reshape((16, 16, 16)))

    model.add(Conv2DTranspose(128, 3, strides=2, padding='same', activation='relu', name='deconv5'))

    model.add(Conv2DTranspose(64, 3, strides=2, padding='same', activation='relu',name='deconv4'))

    model.add(Conv2DTranspose(16, 3, strides=2, padding='same', activation='relu', name='deconv3'))

    model.add(Conv2DTranspose(8, 3, strides=2, padding='same', activation='relu',name='deconv2'))

    model.add(Conv2DTranspose(input_shape[2], 3, strides=2, padding='same', name='deconv1'))
    model.summary()
    return model

