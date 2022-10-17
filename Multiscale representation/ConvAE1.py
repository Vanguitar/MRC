from keras.layers import Conv2D, Conv2DTranspose, Dense, Flatten, Reshape, Activation, Lambda,Concatenate,Conv1D,LeakyReLU
from keras.models import Sequential, Model
from keras.layers.normalization import BatchNormalization
from keras.utils.vis_utils import plot_model
from keras.layers import *
import numpy as np
import tensorflow as tf


def C1_3Dprinter(input_shape=(1024,1)):
    model = Sequential()
    model.add(Conv1D(filters=512, kernel_size=5, strides=2, name='conv1', input_shape=input_shape))
    model.add(BatchNormalization())
    model.add(LeakyReLU())

    model.add(Conv1D(filters=512, kernel_size=5, strides=2,  name='conv2'))
    model.add(BatchNormalization())
    model.add(LeakyReLU())

    model.add(Conv1D(filters=512, kernel_size=5, strides=2,  name='conv3'))
    model.add(BatchNormalization())
    model.add(LeakyReLU())

    model.add(Conv1D(filters=512, kernel_size=5, strides=2,  name='conv4'))
    model.add(BatchNormalization())
    model.add(LeakyReLU())
    model.add(Flatten())

    model.add(Dense(units=256))
    model.add(BatchNormalization())
    model.add(LeakyReLU())

    model.add(Dense(units=16 * int(16) ,  name='embedding'))
    model.add(LeakyReLU())
    model.add(Reshape((int(16),16)))

    model.add(Conv1D(filters=512, kernel_size=5, strides=1, padding='same', name='deconv3'))
    model.add(BatchNormalization())
    model.add(LeakyReLU())
    model.add(UpSampling1D(length=4))
    model.add(Conv1D(filters=512, kernel_size=5, strides=1, padding='same',   name='deconv2'))
    model.add(BatchNormalization())
    model.add(LeakyReLU())
    model.add(UpSampling1D(length=4))
    model.add(Conv1D(filters=512, kernel_size=5, strides=1, padding='same',  name='deconv1'))
    model.add(BatchNormalization())
    model.add(LeakyReLU())
    model.add(UpSampling1D(length=4))
    model.add(Conv1D(input_shape[1], kernel_size=5, strides=1, padding='same',  name='deconv0'))
    model.add(BatchNormalization())
    model.add(LeakyReLU())
    model.summary()
    return model


def C1_cwru(input_shape=(1024,1)):
    model = Sequential()
    model.add(Conv1D(filters=512, kernel_size=3, strides=2, activation='linear', name='conv1', input_shape=input_shape))
    model.add(BatchNormalization())
    model.add(LeakyReLU())

    model.add(Conv1D(filters=512, kernel_size=3, strides=2, activation='linear', name='conv2'))
    model.add(BatchNormalization())
    model.add(LeakyReLU())

    model.add(Conv1D(filters=512, kernel_size=3, strides=2, activation='linear', name='conv3'))
    model.add(BatchNormalization())
    model.add(LeakyReLU())

    model.add(Conv1D(filters=512, kernel_size=3, strides=2, activation='linear', name='conv4'))
    model.add(BatchNormalization())
    model.add(LeakyReLU())
    model.add(Flatten())

    model.add(Dense(units=256, activation='linear'))
    model.add(BatchNormalization())
    model.add(LeakyReLU())

    model.add(Dense(units=16 * int(40), activation='linear', name='embedding'))
    model.add(Reshape((int(40), 16)))

    model.add(Conv1D(filters=512, kernel_size=3, strides=1, padding='valid', activation='linear', name='deconv4'))
    model.add(UpSampling1D(length=3))
    model.add(Conv1D(filters=512, kernel_size=3, strides=1, padding='same', activation='linear', name='deconv3'))
    model.add(UpSampling1D(length=3))
    model.add(Conv1D(filters=512, kernel_size=3, strides=1, padding='same', activation='linear', name='deconv2'))
    model.add(UpSampling1D(length=3))
    model.add(Conv1D(input_shape[1], kernel_size=3, strides=1, padding='valid', activation='linear', name='deconv1'))
    model.add(Conv1D(input_shape[1], kernel_size=3, strides=1, padding='same', activation='linear', name='deconv0'))
    model.summary()
    return model


