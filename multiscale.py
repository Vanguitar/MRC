'''(511,512) (255,512)...padding(n,512,512) zero'''
import pickle
import numpy as np
import h5py

with open('F:/exlayerout1.pkl', 'rb') as handle:
    h1=pickle.load(handle)
with open('F:/exlayerout2.pkl', 'rb') as handle:
    h2=pickle.load(handle)
with open('F:/exlayerout3.pkl', 'rb') as handle:
    h3=pickle.load(handle)
with open('F:/exlayerout4.pkl', 'rb') as handle:
    h4=pickle.load(handle)

    h1_1d = h1.reshape(-1, 511 * 512)
    h2_1d = h2.reshape(-1, 255 * 512)
    h3_1d = h3.reshape(-1, 127 * 512)
    h4_1d = h4.reshape(-1, 63 * 512)

#  (-1,512*512)

    x1 = np.zeros([1440, 512])
    new_x1=np.hstack((h1_1d,x1))

    x2=np.zeros([1440,131584])
    new_x2=np.hstack((h2_1d,x2))

    x3=np.zeros([1440,197120])
    new_x3=np.hstack((h3_1d,x3))

    x4 = np.zeros([1440, 229888])
    new_x4 = np.hstack((h4_1d, x4))


h_2dd = np.ones([1,4,262144])
for j in range(0,1440):
    x11 = new_x1[j]
    x11=x11[np.newaxis, :]
    x11 = x11[np.newaxis, :,:]
    x22 = new_x2[j]
    x22 = x22[np.newaxis, :]
    x22 = x22[np.newaxis, :, :]
    x33 = new_x3[j]
    x33 = x33[np.newaxis, :]
    x33 = x33[np.newaxis, :, :]
    x44 = new_x4[j]
    x44 = x44[np.newaxis, :]
    x44 = x44[np.newaxis, :, :]
    xx = np.concatenate((x11, x22,x33,x44),axis=1)
    h_2dd = np.concatenate((h_2dd,xx),axis=0)
print(h_2dd.shape)
x_2dd = h_2dd[1:]
metho2 = x_2dd.reshape(-1, 4,512,512)
print(metho2.shape)


#   （1150,）
range_all2 = 0*np.ones(775)

for jj in range(1,4):
    rangee2=jj*np.ones(775)
    range_all2=np.concatenate((range_all2, rangee2), axis=0)
print(range_all2.shape)


with h5py.File('paddingJNU4.h5', 'w') as f:
    f.create_dataset('data', data=metho2)
    f.create_dataset('labels', data=range_all2)

    f.close()