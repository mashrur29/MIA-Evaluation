import torch
from torch import nn
import torch.nn.functional as F

from adv_defense.models.rnn_classifier import RNNClassifier
from datasets.get_dataset import texas_data_shadow, purchase_data_shadow
from trained_models.utils import get_mode_dir
from models.classifier import Classifier, ReLUClassifier
import numpy as np
from metric_benchmarks import *

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


def get_prediction_model(model, x):
    logits, _ = model.forward(x)
    logits = logits.to(device='cpu')
    logits = logits.detach().numpy()
    mx = np.max(logits, axis=-1, keepdims=True)
    exp = np.exp((logits - mx))
    denominator = np.sum(exp, axis=-1, keepdims=True)

    return exp / denominator


def iter_data(model, data, batch_size):
    outs = []
    labels = []
    num_batch = len(data[0]) // batch_size
    num_batch -= 1
    for i in range(num_batch):
        x = torch.Tensor(data[0][(i * batch_size): ((i + 1) * batch_size)]).to(device)
        y = data[1][(i * batch_size): ((i + 1) * batch_size)]
        out = get_prediction_model(model, x)
        outs.append(np.array(out))
        labels.append(np.array(y))

    outs = np.concatenate(outs)
    labels = np.concatenate(labels)
    return (outs, labels)


def run_models(ModelClass, model_name, dataset='purchase', isDefense=False, batch_size=128):
    if dataset == 'texas':
        shadow_train, target_train, shadow_test, target_test = texas_data_shadow()

        if model_name == 'rnn_classifier':
            model = ModelClass(6169, 100, batch_size).to(device)
        else:
            model = ModelClass(6169, 100).to(device)
    else:
        shadow_train, target_train, shadow_test, target_test = purchase_data_shadow()

        if model_name == 'rnn_classifier':
            model = ModelClass(600, 100, batch_size).to(device)
        else:
            model = ModelClass(600, 100).to(device)

    if isDefense:
        model_dir = 'trained_models/{}/{}_with_defense/model_best.pth.tar'.format(model_name, dataset)
    else:
        model_dir = 'trained_models/{}/{}_no_defense/model_best.pth.tar'.format(model_name, dataset)


    chk = torch.load(model_dir)
    model.load_state_dict(chk['state_dict'])
    model.to(device)
    model.eval()
    # print(shadow_train[0].shape, shadow_train[1].shape)

    shadow_train_performance = iter_data(model, shadow_train, batch_size)
    shadow_test_performance = iter_data(model, shadow_test, batch_size)
    target_train_performance = iter_data(model, target_train, batch_size)
    target_test_performance = iter_data(model, target_test, batch_size)

    return shadow_train_performance, target_train_performance, shadow_test_performance, target_test_performance


def run_benchmarks(shadow_train_performance, target_train_performance, shadow_test_performance,
                   target_test_performance):
    get_correctness(shadow_train_performance, target_train_performance, shadow_test_performance,
                    target_test_performance)
    get_confidence(shadow_train_performance, target_train_performance, shadow_test_performance, target_test_performance)
    get_entropy(shadow_train_performance, target_train_performance, shadow_test_performance, target_test_performance)
    get_modified_entropy(shadow_train_performance, target_train_performance, shadow_test_performance,
                         target_test_performance)


def run(ModelClass, model_name, tp):
    if tp == 0:
        print('==> Running Models')
        p_shadow_train_performance, p_target_train_performance, p_shadow_test_performance, p_target_test_performance = run_models(
            ModelClass, model_name, 'purchase', False)
        print('==> Purchase no defense benchmarks')
        run_benchmarks(p_shadow_train_performance, p_target_train_performance, p_shadow_test_performance,
                       p_target_test_performance)
    elif tp == 1:
        pd_shadow_train_performance, pd_target_train_performance, pd_shadow_test_performance, pd_target_test_performance = run_models(
            ModelClass, model_name, 'purchase', True)
        print('==> Purchase with defense benchmarks')
        run_benchmarks(pd_shadow_train_performance, pd_target_train_performance, pd_shadow_test_performance,
                       pd_target_test_performance)
    elif tp == 2:
        t_shadow_train_performance, t_target_train_performance, t_shadow_test_performance, t_target_test_performance = run_models(
            ModelClass, model_name, 'texas', False)
        print('==> Texas no defense benchmarks')
        run_benchmarks(t_shadow_train_performance, t_target_train_performance, t_shadow_test_performance,
                       t_target_test_performance)
    else:
        td_shadow_train_performance, td_target_train_performance, td_shadow_test_performance, td_target_test_performance = run_models(
            ModelClass, model_name, 'texas', True)
        print('==> Texas with defense benchmarks')
        run_benchmarks(td_shadow_train_performance, td_target_train_performance, td_shadow_test_performance,
                       td_target_test_performance)


if __name__ == '__main__':
    models = [Classifier, RNNClassifier, ReLUClassifier]
    model_names = ['tanh_classifier', 'rnn_classifier', 'relu_classifier']
    id = 1

    run(models[id], model_names[id], 0)
