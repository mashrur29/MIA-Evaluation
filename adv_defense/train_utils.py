import os.path
import shutil
from statistics import mean
import torch
from torch import nn
import numpy as np


def accuracy(output, target, topk=(1,)):
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.reshape(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].reshape(-1).float().sum(0)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res


def attack_accuracy(out, expected_out, threshold=0.5):
    res = 0.0

    for i in range(len(out)):
        if expected_out[i] == 1:
            if out[i] > threshold:
                res += 1
        else:
            if out[i] <= threshold:
                res += 1
    return float(res / float(len(out)))*100.0


def save_checkpoint(statedict, best, checkpoint, filename):
    if not os.path.isdir(checkpoint):
        try:
            os.makedirs(checkpoint)
        except Exception as e:
            print(str(e))

    path = os.path.join(checkpoint, filename)

    torch.save(statedict, path)
    if best:
        shutil.copyfile(path, os.path.join(checkpoint, 'model_best.pth.tar'))


def train_regular(model_name, X, Y, model, criterion, optimizer, device, batch_size, early_stop=10 ** 9):
    model.train()
    losses = []
    accs = []

    batch_nums = len(X) // batch_size
    batch_nums -= 1

    for i in range(batch_nums):
        if i > early_stop:
            break
        x = X[(batch_size * i):(batch_size * (i + 1))].to(device)
        y = Y[(batch_size * i):(batch_size * (i + 1))].to(device)
        y = y.type(torch.LongTensor).to(device)

        out, _ = model(x)

        # print(out.shape, y.shape)
        loss = criterion(out, y)

        acc1, acc2 = accuracy(out, y, topk=(1, 5))

        losses.append(loss.item())
        accs.append(acc1.item())

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if i % 50 == 0:
            print('Batch {}/{}: loss {}, accuracy {}'.format(i + 1, batch_nums, loss.item(), acc1.item()))

    return mean(losses), mean(accs)


def test_regular(model_name, X, Y, model, criterion, device, batch_size, early_stop=10 ** 9):
    model.eval()
    losses = []
    accs = []

    batch_nums = len(X) // batch_size
    batch_nums -= 1

    with torch.no_grad():
        for i in range(batch_nums):
            if i > early_stop:
                break
            x = X[(batch_size * i):(batch_size * (i + 1))].to(device)
            y = Y[(batch_size * i):(batch_size * (i + 1))].to(device)
            x = torch.autograd.Variable(x)
            y = torch.autograd.Variable(y)
            y = y.type(torch.LongTensor).to(device)

            out, _ = model(x)

            loss = criterion(out, y)
            acc1, acc2 = accuracy(out, y, topk=(1, 5))

            losses.append(loss.item())
            accs.append(acc1.item())

    return mean(losses), mean(accs)


def train_attack(model_name, X, Y, model, attack_x, attack_y, attack_model, attack_criterion, attack_optimizer, device,
                 batch_size,
                 num_classes=100, skip_batch=[], early_stop=10 ** 9):
    model.eval()
    for param in model.parameters():
        param.requires_grad = False
    attack_model.train()

    losses = []
    accs = []

    batch_num = min(len(X) // batch_size, len(attack_x) // batch_size)
    batch_num -= 1

    for i in range(batch_num):
        if i > early_stop:
            break
        if i in skip_batch:
            continue

        train_x = X[(i * batch_size):((i + 1) * batch_size)]
        train_y = Y[(i * batch_size):((i + 1) * batch_size)]

        train_attack_x = attack_x[(i * batch_size):((i + 1) * batch_size)]
        train_attack_y = attack_y[(i * batch_size):((i + 1) * batch_size)]

        out, _ = model(train_x)
        attack_out, _ = model(train_attack_x)
        attack_inp = torch.cat((out, attack_out), dim=0)
        index_tn = torch.cat((train_y, train_attack_y), dim=0).squeeze(0)
        attack_label = nn.functional.one_hot(index_tn.long(), num_classes=num_classes).to(attack_inp.dtype)
        attack_prob = attack_model(attack_inp, attack_label).reshape(-1)
        expected_prob = torch.cat((torch.ones(len(train_x)), torch.zeros(len(train_attack_x))), dim=0).reshape(-1).type(
            torch.FloatTensor).to(device)

        loss = attack_criterion(attack_prob, expected_prob)
        acc = attack_accuracy(attack_prob, expected_prob)

        losses.append(loss.item())
        accs.append(acc)

        attack_optimizer.zero_grad()
        loss.backward()
        attack_optimizer.step()

        if i % 50 == 0:
            print('Batch {}/{}: loss {}, accuracy {}'.format(i + 1, batch_num, loss.item(), acc))

    return mean(losses), mean(accs)


def train_classifier(model_name, X, Y, model, attack_model, criterion, optimizer, device, batch_size, num_classes=100,
                     alpha=0.5, skip_batch=[], early_stop=10 ** 9):
    model.train()
    for param in model.parameters():
        param.requires_grad = True
    attack_model.eval()

    losses = []
    accs = []

    batch_num = len(X) // batch_size
    batch_num -= 1

    for i in range(batch_num):
        if i > early_stop:
            break
        if i in skip_batch:
            continue

        train_x = X[(i * batch_size):((i + 1) * batch_size)]
        train_y = Y[(i * batch_size):((i + 1) * batch_size)].type(torch.LongTensor).to(device)

        out, _ = model(train_x)

        inp_label_enc = nn.functional.one_hot(train_y.reshape(-1), num_classes=num_classes).to(out.dtype)
        attack_out = attack_model(out, inp_label_enc)

        loss = criterion(out, train_y) + alpha * (torch.mean(attack_out) - alpha)
        acc1, acc2 = accuracy(out, train_y, topk=(1, 5))

        losses.append(loss.item())
        accs.append(acc1.item())

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if i % 50 == 0:
            print('Batch {}/{}: loss {}, accuracy {}'.format(i + 1, batch_num, loss.item(), acc1.item()))

    return mean(losses), mean(accs)
