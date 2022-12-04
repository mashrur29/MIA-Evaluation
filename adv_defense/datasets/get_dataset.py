import os
import tarfile
import urllib
import numpy as np


def iter_loadtxt(filename, delimiter=',', skiprows=0, dtype=float):
    def iter_func():
        with open(filename, 'r') as infile:
            for _ in range(skiprows):
                next(infile)
            for line in infile:
                line = line.rstrip().split(delimiter)
                for item in line:
                    yield dtype(item)
        iter_loadtxt.rowlength = len(line)

    data = np.fromiter(iter_func(), dtype=dtype)
    data = data.reshape((-1, iter_loadtxt.rowlength))
    return data


def purchase_data():
    DATASET_PATH = 'datasets/purchase'
    DATASET_NAME = 'dataset_purchase'

    if not os.path.isdir(DATASET_PATH):
        os.makedirs(DATASET_PATH)

    DATASET_FILE = os.path.join(DATASET_PATH, DATASET_NAME)

    if not os.path.isfile(DATASET_FILE):
        urllib.request.urlretrieve("https://www.comp.nus.edu.sg/~reza/files/dataset_purchase.tgz",
                                   os.path.join(DATASET_PATH, 'tmp.tgz'))

        tar = tarfile.open(os.path.join(DATASET_PATH, 'tmp.tgz'))
        tar.extractall(path=DATASET_PATH)

    #data_set = np.genfromtxt(DATASET_FILE, delimiter=',')
    data_set = iter_loadtxt(DATASET_FILE)
    X = data_set[:, 1:].astype(np.float64)
    Y = (data_set[:, 0]).astype(np.int32) - 1

    return X, Y


def texas_data():
    DATASET_PATH = 'datasets'

    if not os.path.isdir(DATASET_PATH):
        os.makedirs(DATASET_PATH)

    DATASET_X = os.path.join(DATASET_PATH, 'texas/100/feats')
    DATASET_Y = os.path.join(DATASET_PATH, 'texas/100/labels')

    if not os.path.isfile(DATASET_X):
        urllib.request.urlretrieve("https://www.comp.nus.edu.sg/~reza/files/dataset_texas.tgz",
                                   os.path.join(DATASET_PATH, 'tmp.tgz'))
        tar = tarfile.open(os.path.join(DATASET_PATH, 'tmp.tgz'))
        tar.extractall(path=DATASET_PATH)

    data_set_features = iter_loadtxt(DATASET_X)
    data_set_label = iter_loadtxt(DATASET_Y)


    X = data_set_features.astype(np.float64)
    Y = data_set_label.astype(np.int32) - 1

    return X, Y
