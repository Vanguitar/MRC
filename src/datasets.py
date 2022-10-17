
def load_dataset(data_path = './data/CWRU'):

    import h5py

    with h5py.File(data_path + '.h5','r') as hf:
        x = hf.get('data')[:]
        y = hf.get('labels')[:]

    x = x.transpose(0,2,3,1)
    a = set(y)
    u = 0

    for id in a:
        y[y==id] = u
        u = u + 1

    print(x.shape)
    return x, y
