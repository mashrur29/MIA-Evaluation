import os.path
import shutil
from statistics import mean
import torch


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
        #x = torch.autograd.Variable(x)
        #y = torch.autograd.Variable(y)
        #y = y.reshape(-1)
        y = y.type(torch.LongTensor).to(device)

        out, h = model(x)

        #print(out.shape, y.shape)
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


def test_regular(X, Y, model, criterion, device, batch_size, early_stop=10 ** 9):
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

            out, h = model(x)

            loss = criterion(out, y)
            acc1, acc2 = accuracy(out, y, topk=(1, 5))

            losses.append(loss.item())
            accs.append(acc1.item())


    return mean(losses), mean(accs)