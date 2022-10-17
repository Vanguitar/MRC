
import keras.backend as K
from src.datasets import *
import tensorflow as tf
import matplotlib

matplotlib.use('Agg')

from src.MRC import MRC
import argparse
import os
import warnings
warnings.filterwarnings("ignore")
warnings.simplefilter(action='ignore', category=FutureWarning)
# GPU setting
def GpuInit():

    os.environ['CUDA_VISIBLE_DEVICES'] = '1'
    config = tf.ConfigProto(gpu_options = tf.GPUOptions(allow_growth=True))
    session = tf.Session(config=config)
    K.set_session(session)

    return session
# parameter
def parse_args():
    parser = argparse.ArgumentParser(description='train')

    parser.add_argument('--dataset',default='CAE_CWRU',choices=['CAE_3Dprinter','CAE_CWRU'])
    parser.add_argument('--n_clusters',default=4,type=int)
    parser.add_argument('--batch_size',default=16,type=int)
    parser.add_argument('--epochs',default=5,type=int)
    parser.add_argument('--cae_weights',
                        help = 'This is argument must be given')
    parser.add_argument('--save_dir',default='F:/MRC/result/JNU4')

    args = parser.parse_args()

    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)

    return args


if __name__ == "__main__":

    args = parse_args()
    print(args)

    sess = GpuInit()

    x, y = load_dataset(data_path = "F:/MRC/data/paddingJNU4")

    a = x.mean(axis=(1, 2, 3))
    b = x.std(axis=(1,2,3))
    x = (x.T - a).T
    x = (x.T / b).T    # 标准化

    dc = MRC(input_shape=(512,512,4),n_clusters =4,dataset='CAE_CWRU',x = x,y= y,
            pretrained = args.cae_weights,
            session = sess,
            lamda = 0,
            alpha = 1)
    dc.visulization(args.save_dir + '/embedding_1.svg', save_dir=args.save_dir)

    dc.pretrain(x,
                batch_size = args.batch_size,
                epochs = args.epochs,
                save_dir=args.save_dir)
    dc.evaluate(flag_all=True)
    dc.visulization(args.save_dir + '/embedding_2.svg',save_dir = args.save_dir)

    dc.refineTrain(x,
                   batch_size = args.batch_size,
                   epochs = 1,
                   save_dir = args.save_dir,
                   second = True
                   )

    dc.evaluate(flag_all=True)
    dc.visulization(args.save_dir + '/embedding_3.svg',save_dir = args.save_dir,flag = 1)
    print('end')





