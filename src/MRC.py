from time import time
import numpy as np
import keras.backend as K

from keras.models import Model

from . import metrics

from .ConvAE import CAE_3Dprinter,CAE_CWRU


import matplotlib

matplotlib.use('Agg')

import matplotlib.pyplot as plt


from sklearn.manifold import TSNE


from itertools import cycle, islice
from sklearn import cluster, datasets, mixture,manifold


import tensorflow as tf
from sklearn.metrics import davies_bouldin_score,silhouette_score,calinski_harabasz_score


class MRC(object):
    def __init__(self,
                 input_shape = (512,512,4),       #  参数设置
                 n_clusters=4,
                 dataset = 'CAE_CWRU',
                 x = None,
                 y = None,
                 pretrained = None,
                 session = None,
                 lamda = 0.1,
                 alpha = 100):

        super(MRC,self).__init__()

        self.n_clusters = n_clusters
        self.input_shape = input_shape

        self.pretrained = pretrained

        self.dataset = dataset
        self.x = x
        self.y = y

        self.lamda = lamda
        self.alpha = alpha

        self.gamma_tr = 1.0
        self.margin = 0.2

        self.sess = session

        self.cae, self.encoder = self.get_model(self.dataset)  #

        self.gmm = mixture.GaussianMixture(n_components=n_clusters, covariance_type='full', max_iter = 200000)  #  GMM


        self.learning_rate = tf.Variable(0.000001, trainable=False, name='learning_rate')



        if self.pretrained is not None:
            print("Load pretrained model...")
            self.load_weights(self.pretrained)
            print("Model %s load ok"%self.pretrained)
        else:
            self.sess.run(tf.global_variables_initializer())

    def pretrain(self, x, batch_size=256, epochs=100, optimizer='adam', save_dir='results/temp'):
        print('...Pretraining...')
        self.cae.compile(optimizer=optimizer, loss='mse')
        from keras.callbacks import CSVLogger
        csv_logger = CSVLogger(save_dir + '/pretrain_log.csv')

        # begin training
        t0 = time()
        self.cae.fit(x, x, batch_size=batch_size, epochs=epochs, callbacks=[csv_logger])
        print('Pretraining time: ', time() - t0)
        self.cae.save_weights(save_dir + '/pretrain_model.h5')
        print('Pretrained weights are saved to %s/pretrain_model.h5' % save_dir)
        self.pretrained = save_dir + '/pretrain_model.h5'

    def refineTrain(self, x, batch_size=256, epochs=100, save_dir='results/temp', second=True):

        self.optimizer = tf.train.AdamOptimizer(self.learning_rate)

        self.cluster(x)

        self.gmm.fit(self.z_reduced)
        y_pred = self.gmm.predict(self.z_reduced)


        self.z_label = y_pred

        print("Refine Network...")
        self.proba = self.gmm.predict_proba(self.z_reduced)
        self.p = self.proba

        # restruct loss
        self.input_x = tf.placeholder(tf.float32,
                                      shape=[None, self.input_shape[0], self.input_shape[1], self.input_shape[2]])

        restruct = self.cae(self.input_x)

        self.loss_restruct = K.mean(K.square(restruct - self.input_x))

        # NEW_loss
        self.idx_1 = tf.placeholder(tf.int32, shape=[None])
        self.idx_2 = tf.placeholder(tf.int32, shape=[None])
        self.idx_3 = tf.placeholder(tf.int32, shape=[None])

        self.propos = tf.placeholder(tf.float32, shape=[None])

        z = self.encoder(self.input_x)

        z_anc = tf.gather(z, self.idx_1)
        z_pos = tf.gather(z, self.idx_2)
        z_neg = tf.gather(z, self.idx_3)

        d_pos = tf.reduce_sum(tf.square(z_anc - z_pos), -1)
        d_neg = tf.reduce_sum(tf.square(z_anc - z_neg), -1)

        d_pos = tf.multiply(d_pos, self.propos)

        self.loss_dis = tf.reduce_mean(tf.maximum(0., self.margin + self.gamma_tr * d_pos - d_neg))

        # all_loss
        self.loss = self.loss_restruct + self.loss_dis * self.alpha

        self.train_step = self.optimizer.minimize(self.loss)

        # initial
        if second == False:
            self.sess.run(tf.global_variables_initializer())

        if self.pretrained is not None:
            self.cae.load_weights(self.pretrained)

        # train
        sz = x.shape[0] // batch_size - 1

        for epoch in range(epochs):

            index_rand = np.random.permutation(x.shape[0])

            all_loss = 0
            for i in range(sz):
                tx = x[index_rand[i * batch_size:(i + 1) * batch_size]]

                idx_1, idx_2, idx_3, propos = self.getSample(
                    self.z_label[index_rand[i * batch_size:(i + 1) * batch_size]],
                    self.p[index_rand[i * batch_size:(i + 1) * batch_size]])


                _, loss, loss_re, loss_dis = self.sess.run([self.train_step, self.loss, self.loss_restruct,
                                                            self.loss_dis],
                                                           feed_dict={self.input_x: tx,
                                                                      self.idx_1: idx_1,
                                                                      self.idx_2: idx_2,
                                                                      self.idx_3: idx_3,
                                                                      self.propos: propos,
                                                                      })

                all_loss += loss

            print('Epoch:%4d Loss: %.4f %.4f re: %.4f dis: %.4f' % (epoch, all_loss / sz, loss, loss_re, loss_dis))

        self.cae.save_weights(save_dir + '/refine_model.h5')

        return

    def cluster(self, x, reduce = True):

        z = self.encoder.predict(x)

        if reduce:
            z_reduced = TSNE(n_components=2, random_state=0).fit_transform(z)
        else:
            z_reduced = z

        self.gmm.fit(z_reduced)
        z_label = self.gmm.predict(z_reduced)

        self.z_reduced = z_reduced
        self.z_label = z_label

        return z_label,z_reduced

    def evaluate(self,flag_all = False):

        index = np.random.randint(self.x.shape[0],size = 1000)

        x = self.x[index]
        y = self.y[index]

        if flag_all:
            x = self.x
            y = self.y

        y_pred,x_reduced = self.cluster(x)

        acc = metrics.acc(y, y_pred)
        nmi = metrics.nmi(y, y_pred)
        ari = metrics.ari(y, y_pred)
        sc = silhouette_score(x_reduced, y_pred)
        cn=calinski_harabasz_score(x_reduced, y_pred)
        dbi=davies_bouldin_score(x_reduced, y_pred)

        print('Clustering after reduction: acc = %.4f, nmi = %.4f, ari = %.4f,sc = %.4f,cn= %.4f,dbi= %.4f' % (acc,nmi,ari,sc,cn,dbi))

        y_pred,x_reduced = self.cluster(x, False)

        acc = metrics.acc(y, y_pred)
        nmi = metrics.nmi(y, y_pred)
        ari = metrics.ari(y, y_pred)
        sc = silhouette_score(x_reduced, y_pred)
        cn=calinski_harabasz_score(x_reduced, y_pred)
        dbi=davies_bouldin_score(x_reduced, y_pred)

        print('Clustering without reduction: acc = %.4f, nmi = %.4f, ari = %.4f,sc = %.4f,cn= %.4f,dbi= %.4f' % (acc,nmi,ari,sc,cn,dbi))

        return acc,nmi,ari,sc,cn,dbi

    def load_weights(self,weights_path):
        self.cae.load_weights(weights_path)

    def getSample(self,label,prob):
        st,num = np.unique(label,return_counts=True)
        num_triplet = 0

        for i in range(st.shape[0]):
            num_triplet = num_triplet + (num[i] + 1) * num[i] // 2

        anc = np.zeros(num_triplet, np.int32)
        pos = np.zeros(num_triplet, np.int32)
        neg = np.zeros(num_triplet, np.int32)

        propos = np.ones(num_triplet,np.float32)

        index = 0
        for i in range(st.shape[0]):

            temp = np.argwhere(label == st[i])
            netemp = np.argwhere(label != st[i])

            for t in range(temp.shape[0]):
                for k in range(t+1,temp.shape[0]):
                    anc[index] = temp[t]
                    pos[index] = temp[k]

                    if prob[temp[t]].max() < 0.7:
                        break
                    elif prob[temp[k]].max() < 0.7:
                        continue

                    propos[index] = propos[index] - np.abs(prob[temp[t]].max() - prob[temp[k]].max())

                    c = np.random.randint(netemp.shape[0])
                    neg[index] = netemp[c]

                    index += 1

        return anc,pos,neg,propos

    def getPQ(self,q):
        p = q ** 2 / np.sum(q,axis=0)
        p = (p.T / np.sum(p,axis=1)).T
        return q, p

    def visulization(self,name = 'temp.svg',save_dir = None,flag=0):


        x = self.x
        y = self.y

        z = self.encoder.predict(x)

        print('Enbedding shape:',z.shape)
        z_reduced = TSNE(n_components=2, random_state=0).fit_transform(z)

        if flag == 0:
            np.save(save_dir + '/x.npy',z)
            np.save(save_dir + '/x_reduced.npy',z_reduced)
            np.save(save_dir + '/y_label.npy',y)
        else:
            np.save(save_dir + '/rx.npy',z)
            np.save(save_dir + '/rx_reduced.npy',z_reduced)
            np.save(save_dir + '/ry_label.npy',y)


        y = y.astype(int)

        colors = np.array(list(islice(cycle(['#377eb8', '#ff7f00', '#4daf4a',
                                             '#f781bf', '#a65628', '#984ea3',
                                             '#999999', '#e41a1c', '#dede00','#171717']),
                                      int(max(y) + 1))))

        colors = np.append(colors, ["#000000"])

        plt.scatter(z_reduced[:, 0], z_reduced[:, 1], s=10, color=colors[y],alpha=0.5)

        plt.axis('on')
        plt.savefig(name, dpi=600, format="svg", bbox_inches='tight')
        plt.close('all')


    def get_model(self, dataset ='cwru2D'):

        if 'CAE_3Dprinter' in dataset:
            cae = CAE_3Dprinter(self.input_shape)
        elif 'CAE_CWRU' in dataset:
            cae = CAE_CWRU(self.input_shape)

        embedding = cae.get_layer(name='embedding').output
        encoder = Model(inputs = cae.input, outputs=embedding)

        return cae, encoder