from time import time
import numpy as np
import keras.backend as K
from keras.engine.topology import Layer, InputSpec
from keras.models import Model
from keras.utils.vis_utils import plot_model
from keras.optimizers import Adam
from sklearn.cluster import KMeans

from .datasets1 import *

import tensorflow as tf
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from tqdm import tqdm

from sklearn.manifold import TSNE

import cv2
from .MRC1 import MRC1
import argparse
import os
matplotlib.use('Agg')



def GpuInit():

    os.environ['CUDA_VISIBLE_DEVICES'] = '1'
    config = tf.ConfigProto(gpu_options = tf.GPUOptions(allow_growth=True))
    session = tf.Session(config=config)
    K.set_session(session)

    return session

def parse_args():
    parser = argparse.ArgumentParser(description='train')

    parser.add_argument('--datasets',default='C1_cwru',choices=['C1_3Dprinter','C1_cwru'])
    parser.add_argument('--n_clusters',default=4,type=int)
    parser.add_argument('--batch_size',default=16,type=int) #16
    parser.add_argument('--epochs',default=1,type=int)  #500改为1
    parser.add_argument('--cae_weights',
                        help = 'This is argument must be given')
    parser.add_argument('--save_dir',default='results2/JNU')

    args = parser.parse_args()


    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)

    return args


if __name__ == "__main__":

    args = parse_args()

    print(args)

    sess = GpuInit()

    x, y = load_dataset(data_path='E:\\JNU\\all4')

    dc = MRC1(input_shape=(1024,1),n_clusters =4,datasets='cwru1D',x = x,y= y,
            pretrained = args.cae_weights,
            session = sess)
    # 将x放入AE训练
    dc.pretrain(x,
                batch_size = args.batch_size,
                epochs = args.epochs)
    print('end')



















