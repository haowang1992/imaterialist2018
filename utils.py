import torch
from torch.autograd import Variable
from tqdm import tqdm

import numpy as np

use_gpu = torch.cuda.is_available()


class RunningMean:
    def __init__(self, value=0, count=0):
        self.total_value = value
        self.count = count

    def update(self, value, count=1):
        self.total_value += value
        self.count += count

    @property
    def value(self):
        if self.count:
            return self.total_value / self.count
        else:
            return float("inf")

    def __str__(self):
        return str(self.value)


def predict(model, dataloader):
    all_labels = []
    all_outputs = []
    model.eval()

    pbar = tqdm(dataloader, total=len(dataloader))
    #print(len(dataloader))
    for inputs, labels in pbar:
        if len(inputs.size()) == 5:
            labels_np = labels.numpy()
            #print(labels_np.repeat(5))
            all_labels.append(torch.from_numpy(labels_np.repeat(5)))
        else:
            all_labels.append(labels)

        inputs = Variable(inputs, volatile=True)
        if use_gpu:
            inputs = inputs.cuda()

        if len(inputs.size()) == 5:
            bs, ncrops, c, h, w = inputs.size()
            outputs = model(inputs.view(-1, c, h, w))
            #print(outputs.size())
        else:
            outputs = model(inputs)

        all_outputs.append(outputs.data.cpu())

    all_outputs = torch.cat(all_outputs)
    all_labels = torch.cat(all_labels)

    #print(all_outputs.size())

    if use_gpu:
        all_labels = all_labels.cuda()
        all_outputs = all_outputs.cuda()

    return all_labels, all_outputs


def safe_stack_2array(a, b, dim=0):
    if a is None:
        return b
    return torch.stack((a, b), dim=dim)


def predict_tta(model, dataloaders):
    prediction = None
    lx = None
    for dataloader in dataloaders:
        lx, px = predict(model, dataloader)
        prediction = safe_stack_2array(prediction, px, dim=-1)

    return lx, prediction



def mixup_data(x, y, alpha=1.0, use_cuda=True):

    '''Compute the mixup data. Return mixed inputs, pairs of targets, and lambda'''
    if alpha > 0.:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1.
    batch_size = x.size()[0]
    if use_cuda:
        index = torch.randperm(batch_size).cuda()
    else:
        index = torch.randperm(batch_size)

    mixed_x = lam * x + (1 - lam) * x[index,:]
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lam


def mixup_criterion(y_a, y_b, lam):
    return lambda criterion, pred: lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)