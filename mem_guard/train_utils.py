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


def train_regular(X, Y, model, criterion, optimizer, device, batch_size, early_stop=10 ** 9):
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


def test_regular(X, Y, model, criterion, device, batch_size, noise=None, early_stop=10 ** 9):
    model.eval()
    losses = []
    accs = []

    if noise is not None:
        noise = noise.to(device)

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
            if noise is not None:
                for j in range(out.shape[0]):

                    out[j] = torch.add(out[j], noise)

            loss = criterion(out, y)
            acc1, acc2 = accuracy(out, y, topk=(1, 5))

            losses.append(loss.item())
            accs.append(acc1.item())

    return mean(losses), mean(accs)


def train_w_noise(model_name, X, Y, model, criterion, optimizer, device, batch_size, num_classes=100, early_stop=10 ** 9):
    model.train()
    losses = []
    accs = []

    noise = torch.empty(num_classes,)
    nn.init.normal_(noise)
    noise.requires_grad_()
    #noise.requires_grad = True



    noise = noise.to(device)

    batch_nums = len(X) // batch_size
    batch_nums -= 1

    with torch.autograd.set_detect_anomaly(True):
        for i in range(batch_nums):
            if i > early_stop:
                break
            x = X[(batch_size * i):(batch_size * (i + 1))].to(device)
            y = Y[(batch_size * i):(batch_size * (i + 1))].to(device)
            y = y.type(torch.LongTensor).to(device)

            out, _ = model(x)
            out_clone = out.clone()

            for j in range(out_clone.shape[0]):
                #noise = noise.to(device)
                #noise_clone = noise.to(device).clone()
                out_clone[j] = torch.add(out_clone[j], noise)


            loss = criterion(out_clone, y)

            acc1, acc2 = accuracy(out_clone, y, topk=(1, 5))


            losses.append(loss.item())
            accs.append(acc1.item())

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if i % 50 == 0:
                print('Batch {}/{}: loss {}, accuracy {}'.format(i + 1, batch_nums, loss.item(), acc1.item()))

    return mean(losses), mean(accs), noise
