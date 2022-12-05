import torch
from torch import nn, optim
import numpy as np
from train_utils import *
from models.classifier import Classifier, ReLUClassifier
from models.rnn_classifier import RNNClassifier
from models.attack_model import AttackModel
from datasets.get_dataset import *
from termcolor import colored

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


def write2file(filename, lst):
    with open(filename, 'w+') as fp:
        for item in lst:
            fp.write("%s\n" % item)


def train_no_defense(ModelClass, model_name, X, Y, test_X, test_Y, dataset, defended, epochs=50, batch_size=128,
                     learning_rate=0.0001):
    accs = []
    train_accs = []
    checkpoint = 'trained_models/{}/{}_{}'.format(model_name, dataset, defended)
    best_acc = -1
    num_classes = 100
    inp_size = 600
    if dataset == 'texas':
        inp_size = 6169

    if model_name == 'rnn_classifier':
        model = ModelClass(inp_size, num_classes, batch_size).to(device)
    else:
        model = ModelClass(inp_size, num_classes).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    for ep in range(epochs):
        print(colored('Epoch {}/{}:'.format(ep + 1, epochs), 'blue'))
        ind = np.arange(len(X))
        np.random.shuffle(ind)
        train_x = X[ind]
        train_y = Y[ind]

        train_x = torch.Tensor(train_x).to(device)
        train_y = torch.Tensor(train_y).to(device)
        test_X = torch.Tensor(test_X).to(device)
        test_Y = torch.Tensor(test_Y).to(device)

        print('==> Starting to train')
        train_loss, train_acc = train_regular(model_name, train_x, train_y, model, criterion, optimizer, device,
                                              batch_size=batch_size)
        train_accs.append(train_acc)
        print('==> Training complete')

        print(colored('Train loss {} and Train acc {}'.format(train_loss, train_acc), 'yellow'))
        test_loss, test_acc = test_regular(model_name, test_X, test_Y, model, criterion, device, batch_size=batch_size)
        accs.append(test_acc)
        print(colored('Test loss {} and Test acc {}'.format(test_loss, test_acc), 'green'))
        print('')

        isBest = best_acc < test_acc
        best_acc = max(best_acc, test_acc)

        save_checkpoint(
            {
                'epoch': ep + 1,
                'state_dict': model.state_dict(),
                'acc': test_acc,
                'best_acc': best_acc,
                'optimizer': optimizer.state_dict()
            },
            isBest,
            checkpoint=checkpoint,
            filename='epoch{}.pth.tar'.format(ep + 1)
        )

    print(colored('Best Acc {}'.format(best_acc), 'green'))
    write2file(os.path.join(checkpoint, 'test_acc_no_defense'), accs)
    write2file(os.path.join(checkpoint, 'train_acc_no_defense'), train_accs)


def train_with_defense(ModelClass, model_name, X, Y, attack_X, attack_Y, test_X, test_Y, dataset, defended, epochs=50,
                       batch_size=128,
                       learning_rate=0.001, attack_learning_rate=0.0001):
    accs = []
    train_accs = []
    attack_accs = []
    checkpoint = 'trained_models/{}/{}_{}'.format(model_name, dataset, defended)
    best_acc = -1
    num_classes = 100
    inp_size = 600
    if dataset == 'texas':
        inp_size = 6169

    if model_name == 'rnn_classifier':
        model = ModelClass(inp_size, num_classes, batch_size).to(device)
    else:
        model = ModelClass(inp_size, num_classes).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    attack_model = AttackModel(num_classes).to(device)
    attack_criterion = nn.MSELoss()
    attack_optimizer = optim.Adam(attack_model.parameters(), lr=attack_learning_rate)

    for ep in range(epochs):
        print(colored('Epoch {}/{}:'.format(ep + 1, epochs), 'blue'))
        ind = np.arange(len(X))
        np.random.shuffle(ind)
        train_x = X[ind]
        train_y = Y[ind]
        attack_x = attack_X[ind]
        attack_y = attack_Y[ind]

        train_x = torch.Tensor(train_x).to(device)
        train_y = torch.Tensor(train_y).to(device)
        attack_x = torch.Tensor(attack_x).to(device)
        attack_y = torch.Tensor(attack_y).to(device)
        test_x = torch.Tensor(test_X).to(device)
        test_y = torch.Tensor(test_Y).to(device)

        print('==> Starting to train')

        if ep == 0:
            train_loss, train_acc = train_regular(model_name, train_x, train_y, model, criterion, optimizer, device,
                                                  batch_size=batch_size)
            attack_loss, attack_acc = train_attack(model_name, train_x, train_y, model, attack_x, attack_y,
                                                   attack_model,
                                                   attack_criterion, attack_optimizer, device, batch_size=batch_size)
        else:
            attack_loss, attack_acc = train_attack(model_name, train_x, train_y, model, attack_x, attack_y,
                                                   attack_model,
                                                   attack_criterion, attack_optimizer, device, batch_size=batch_size)
            train_loss, train_acc = train_classifier(model_name, train_x, train_y, model, attack_model, criterion,
                                                     optimizer,
                                                     device,
                                                     batch_size=batch_size)

        print('==> Training complete')
        test_loss, test_acc = test_regular(model_name, test_x, test_y, model, criterion, device, batch_size=batch_size)
        accs.append(test_acc)
        attack_accs.append(attack_acc)
        train_accs.append(train_acc)

        print(colored('Train loss {} and Train acc {}'.format(train_loss, train_acc), 'yellow'))
        print(colored('Attack loss {} and Attack acc {}'.format(attack_loss, attack_acc), 'yellow'))
        print(colored('Test loss {} and Test acc {}'.format(test_loss, test_acc), 'green'))
        print('')

        isBest = best_acc < test_acc
        best_acc = max(best_acc, test_acc)

        save_checkpoint(
            {
                'epoch': ep + 1,
                'state_dict': model.state_dict(),
                'acc': test_acc,
                'best_acc': best_acc,
                'optimizer': optimizer.state_dict()
            },
            isBest,
            checkpoint=checkpoint,
            filename='epoch{}.pth.tar'.format(ep + 1)
        )

    print(colored('Best Acc {}'.format(best_acc), 'green'))
    write2file(os.path.join(checkpoint, 'test_acc_defense'), accs)
    write2file(os.path.join(checkpoint, 'train_acc_defense'), train_accs)
    write2file(os.path.join(checkpoint, 'attack_acc_defense'), attack_accs)


def train_purchase_no_defense(ModelClass, model_name, X, Y, test_X, test_Y):
    lr = 0.003
    if model_name == 'rnn_classifier':
        lr = 0.001
    train_no_defense(ModelClass, model_name, X, Y, test_X, test_Y, dataset='purchase', defended='no_defense', epochs=20,
                     learning_rate=lr)


def train_purchase_defense(ModelClass, model_name, X, Y, attack_X, attack_Y, test_X, test_Y):
    lr = 0.0001
    if model_name == 'rnn_classifier':
        lr = 0.001
    train_with_defense(ModelClass, model_name, X, Y, attack_X, attack_Y, test_X, test_Y, dataset='purchase',
                       defended='with_defense', epochs=20,
                       learning_rate=lr, attack_learning_rate=0.001)


def train_texas_defense(ModelClass, model_name, X, Y, attack_X, attack_Y, test_X, test_Y):
    Y = Y.reshape(-1)
    test_Y = test_Y.reshape(-1)
    attack_Y = attack_Y.reshape(-1)
    lr = 0.001
    if model_name == 'rnn_classifier':
        lr = 0.001
    train_with_defense(ModelClass, model_name, X, Y, attack_X, attack_Y, test_X, test_Y, dataset='texas',
                       defended='with_defense',
                       epochs=20, learning_rate=lr, attack_learning_rate=0.001)


def train_texas_no_defense(ModelClass, model_name, X, Y, test_X, test_Y):
    Y = Y.reshape(-1)
    test_Y = test_Y.reshape(-1)
    lr = 0.0001
    if model_name == 'rnn_classifier':
        lr = 0.001
    train_no_defense(ModelClass, model_name, X, Y, test_X, test_Y, dataset='texas', defended='no_defense', epochs=20,
                     learning_rate=lr)


def execute_purchase(ModelClass, model_name, isDefense=False):
    purchaseX, purchaseY = purchase_data()
    purchase_len_train = len(purchaseX)
    purchase_train_classifier_ratio, purchase_train_attack_ratio = 0.1, 0.15
    train_classifier_x = purchaseX[:int(purchase_train_classifier_ratio * purchase_len_train)]
    train_attack_x = purchaseX[int(purchase_train_classifier_ratio * purchase_len_train):int(
        (purchase_train_classifier_ratio + purchase_train_attack_ratio) * purchase_len_train)]
    test_x = purchaseX[int((purchase_train_classifier_ratio + purchase_train_attack_ratio) * purchase_len_train):]
    train_classifier_y = purchaseY[:int(purchase_train_classifier_ratio * purchase_len_train)]
    train_attack_y = purchaseY[int(purchase_train_classifier_ratio * purchase_len_train):int(
        (purchase_train_classifier_ratio + purchase_train_attack_ratio) * purchase_len_train)]
    test_y = purchaseY[int((purchase_train_classifier_ratio + purchase_train_attack_ratio) * purchase_len_train):]

    if isDefense:
        train_purchase_defense(ModelClass, model_name, train_classifier_x, train_classifier_y, train_attack_x,
                               train_attack_y,
                               test_x, test_y)
    else:
        train_purchase_no_defense(ModelClass, model_name, train_classifier_x, train_classifier_y, test_x, test_y)


def execute_texas(ModelClass, model_name, isDefense=False):
    texasX, texasY = texas_data()

    texas_len_train = len(texasX)
    texas_train_classifier_ratio, texas_train_attack_ratio = 0.2, 0.3
    train_classifier_x = texasX[:int(texas_train_classifier_ratio * texas_len_train)]
    train_attack_x = texasX[int(texas_train_classifier_ratio * texas_len_train):int(
        (texas_train_classifier_ratio + texas_train_attack_ratio) * texas_len_train)]
    test_x = texasX[int((texas_train_classifier_ratio + texas_train_attack_ratio) * texas_len_train):]
    train_classifier_y = texasY[:int(texas_train_classifier_ratio * texas_len_train)]
    train_attack_y = texasY[int(texas_train_classifier_ratio * texas_len_train):int(
        (texas_train_classifier_ratio + texas_train_attack_ratio) * texas_len_train)]
    test_y = texasY[int((texas_train_classifier_ratio + texas_train_attack_ratio) * texas_len_train):]

    if isDefense:
        train_texas_defense(ModelClass, model_name, train_classifier_x, train_classifier_y, train_attack_x,
                            train_attack_y, test_x,
                            test_y)
    else:
        train_texas_no_defense(ModelClass, model_name, train_classifier_x, train_classifier_y, test_x, test_y)


if __name__ == '__main__':
    isDefense = True

    models = [Classifier, RNNClassifier, ReLUClassifier]
    model_names = ['tanh_classifier', 'rnn_classifier', 'relu_classifier']

    for id in range(3):
        execute_purchase(models[id], model_names[id], isDefense=True)
        execute_purchase(models[id], model_names[id], isDefense=False)

    for id in range(3):
        execute_texas(models[id], model_names[id], isDefense=True)
        execute_texas(models[id], model_names[id], isDefense=False)
