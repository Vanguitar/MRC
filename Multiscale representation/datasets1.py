import numpy as np
import os
import cv2
import random
from PIL import Image


def load_dataset(data_path = './datasets'):

    import h5py

    with h5py.File(data_path + '.h5','r') as hf:
        x = hf.get('data')[:]
        y = hf.get('labels')[:]

    a = set(y)
    u = 0

    for id in a:
        y[y==id] = u
        u = u + 1

    print(x.shape)
    return x, y




