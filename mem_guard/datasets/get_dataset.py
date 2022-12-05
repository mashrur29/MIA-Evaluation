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

    # data_set = np.genfromtxt(DATASET_FILE, delimiter=',')
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


def purchase_data_shadow():
    purchaseX, purchaseY = purchase_data()
    purchase_len_train = len(purchaseX)
    purchase_train_classifier_ratio, purchase_train_attack_ratio = 0.1, 0.15

    train_classifier_x = purchaseX[:int(purchase_train_classifier_ratio * purchase_len_train)]
    test_x = purchaseX[int((purchase_train_classifier_ratio + purchase_train_attack_ratio) * purchase_len_train):]
    train_classifier_y = purchaseY[:int(purchase_train_classifier_ratio * purchase_len_train)]
    test_y = purchaseY[int((purchase_train_classifier_ratio + purchase_train_attack_ratio) * purchase_len_train):]

    index1 = np.arange(len(train_classifier_x))
    np.random.shuffle(index1)
    shadow_train_ind = index1[:(len(train_classifier_x) // 2)]
    target_train_ind = index1[(len(train_classifier_x) // 2):]

    index2 = np.arange(len(test_x))
    np.random.shuffle(index2)
    shadow_test_ind = index2[:(len(test_x) // 2)]
    target_test_ind = index2[(len(test_x) // 2):]


    #print(train_classifier_y[shadow_train_ind].shape)

    # shadow_train, target_train, shadow_test, target_test
    return (train_classifier_x[shadow_train_ind], train_classifier_y[shadow_train_ind]), (
        train_classifier_x[target_train_ind], train_classifier_y[target_train_ind]), (
               test_x[shadow_test_ind], test_y[shadow_test_ind]), (
               test_x[target_test_ind], test_y[target_test_ind])


def texas_data_shadow():
    texasX, texasY = texas_data()

    texas_len_train = len(texasX)
    texas_train_classifier_ratio, texas_train_attack_ratio = 0.2, 0.3
    train_classifier_x = texasX[:int(texas_train_classifier_ratio * texas_len_train)]
    test_x = texasX[int((texas_train_classifier_ratio + texas_train_attack_ratio) * texas_len_train):]
    train_classifier_y = texasY[:int(texas_train_classifier_ratio * texas_len_train)]
    test_y = texasY[int((texas_train_classifier_ratio + texas_train_attack_ratio) * texas_len_train):]

    index1 = np.arange(len(train_classifier_x))
    np.random.shuffle(index1)
    shadow_train_ind = index1[:(len(train_classifier_x) // 2)]
    target_train_ind = index1[(len(train_classifier_x) // 2):]

    index2 = np.arange(len(test_x))
    np.random.shuffle(index2)
    shadow_test_ind = index2[:(len(test_x) // 2)]
    target_test_ind = index2[(len(test_x) // 2):]



    # shadow_train, target_train, shadow_test, target_test
    return (train_classifier_x[shadow_train_ind], train_classifier_y[shadow_train_ind].reshape(-1)), (
        train_classifier_x[target_train_ind], train_classifier_y[target_train_ind].reshape(-1)), (
               test_x[shadow_test_ind], test_y[shadow_test_ind].reshape(-1)), (
               test_x[target_test_ind], test_y[target_test_ind].reshape(-1))
