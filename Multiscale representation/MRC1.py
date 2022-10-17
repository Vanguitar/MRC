from time import time
import numpy as np
import keras.backend as K
from keras.engine.topology import Layer, InputSpec
from keras.models import Model
from keras.utils.vis_utils import plot_model
from sklearn.cluster import KMeans,SpectralClustering,AgglomerativeClustering

from .ConvAE1 import C1_3Dprinter,C1_cwru

import tensorflow as tf
import matplotlib
import pickle
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from tqdm import tqdm

from sklearn.manifold import TSNE
from sklearn import mixture
from sklearn.neighbors import kneighbors_graph

from itertools import cycle, islice
from sklearn import cluster, datasets, mixture,manifold
from sklearn.preprocessing import MinMaxScaler, StandardScaler

import cv2

from IPython import embed
import tensorflow as tf

matplotlib.use('Agg')


class MRC1(object):
    def __init__(self,
                 input_shape=(1024,1),       #  参数设置
                 n_clusters=4,
                 datasets = 'cwru',
                 x = None,
                 y = None,
                 pretrained = None,
                 session = None,):

        super(MRC1,self).__init__()

        self.n_clusters = n_clusters
        self.input_shape = input_shape

        self.pretrained = pretrained

        self.datasets = datasets
        self.x = x
        self.y = y

        self.sess = session

        self.cae, self.encoder,self.exlayer1, self.exlayer2,self.exlayer3,self.exlayer4= self.get_model(self.datasets) #  AE模型


        self.learning_rate = tf.Variable(0.000001, trainable=False, name='learning_rate')


        if self.pretrained is not None:
            print("Load pretrained model...")
            self.load_weights(self.pretrained)
            print("Model %s load ok"%self.pretrained)
        else:
            self.sess.run(tf.global_variables_initializer())

    def pretrain(self, x, batch_size=256, epochs=100, optimizer='adam'):
        print('...Pretraining...')
        self.cae.compile(optimizer=optimizer, loss='mse')

        # begin training
        t0 = time()
        self.cae.fit(x, x, batch_size=batch_size, epochs=epochs)  #   训练
        print('Pretraining time: ', time() - t0)

         ##########
        exlayerout1=self.exlayer1.predict(x)
        exlayerout2 = self.exlayer2.predict(x)
        exlayerout3 = self.exlayer3.predict(x)
        exlayerout4 = self.exlayer4.predict(x)

        with open('F:\\exlayerout1.pkl', 'wb') as file:
            pickle.dump(exlayerout1, file,protocol = 4)
        with open('F:\\exlayerout2.pkl', 'wb') as file:
            pickle.dump(exlayerout2, file,protocol = 4)
        with open('F:\\exlayerout3.pkl', 'wb') as file:
            pickle.dump(exlayerout3, file,protocol = 4)
        with open('F:\\exlayerout4.pkl', 'wb') as file:
            pickle.dump(exlayerout4, file,protocol = 4)
        print(exlayerout1.shape, exlayerout2.shape, exlayerout3.shape, exlayerout4.shape)

    def load_weights(self,weights_path):
        self.cae.load_weights(weights_path)

    def get_model(self, datasets ='cwru'):

        if 'C1_3Dprinter' in datasets:
            cae = C1_3Dprinter(self.input_shape)
        elif 'C1_cwru' in datasets:
            cae = C1_cwru(self.input_shape)

        embedding = cae.get_layer(name='embedding').output
        encoder = Model(inputs = cae.input, outputs=embedding)
        exlayer1 = Model(inputs = cae.input, outputs=cae.layers[2].output)
        exlayer2 = Model(inputs=cae.input, outputs=cae.layers[5].output)
        exlayer3 = Model(inputs=cae.input, outputs=cae.layers[8].output)
        exlayer4 = Model(inputs=cae.input, outputs=cae.layers[11].output)

        return cae, encoder,exlayer1,exlayer2,exlayer3,exlayer4

