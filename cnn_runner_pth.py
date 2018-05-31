import argparse

from tqdm import tqdm
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.autograd import Variable
import torch.backends.cudnn as cudnn

import models
import utils
from utils import RunningMean, use_gpu, mixup_data, mixup_criterion
from misc import FurnitureDataset, preprocess, preprocess_with_augmentation, NB_CLASSES, preprocess_hflip, preprocess_five_crop, preprocess_five_crop_hflip

from FocalLoss import FocalLoss
import numpy as np

BATCH_SIZE = 10


def get_model(name):
    print('[+] loading model... ', end='', flush=True)
    if name == 'densenet201':
        model = models.densenet201_finetune(NB_CLASSES)
    elif name == 'inceptionresnetv2':
        model = models.inceptionresnetv2_finetune(NB_CLASSES)
    if name == 'senet154':
        model = models.senet154_finetune(NB_CLASSES)
    if name == 'nasnetlarge':
        model = models.nasnetlarge_finetune(NB_CLASSES)
    if name == 'inceptionv4':
        model = models.inceptionv4_finetune(NB_CLASSES)
    if name == 'se_resnext101_32x4d':
        model = models.se_resnext101_32x4d_finetune(NB_CLASSES)

    if use_gpu:
        model.cuda()
    print('done')
    return model


def predict(args):
    model = get_model(args.name)
    model.load_state_dict(torch.load('models_trained/{}_{}_{}/best_val_weight_{}.pth'.format(args.name, args.aug, args.alpha, args.name)))
    model.eval()

    #tta_preprocess = [preprocess_five_crop, preprocess_five_crop_hflip]
    tta_preprocess = [preprocess, preprocess_hflip]

    data_loaders = []
    for transform in tta_preprocess:
        test_dataset = FurnitureDataset('test', transform=transform)
        data_loader = DataLoader(dataset=test_dataset, num_workers=1,
                                 batch_size=BATCH_SIZE,
                                 shuffle=False)
        data_loaders.append(data_loader)

    lx, px = utils.predict_tta(model, data_loaders)
    data = {
        'lx': lx.cpu(),
        'px': px.cpu(),
    }
    torch.save(data, 'models_trained/{}_{}_{}/test_prediction_{}.pth'.format(args.name, args.aug, args.alpha, args.name))

    data_loaders = []
    for transform in tta_preprocess:
        test_dataset = FurnitureDataset('val', transform=transform)
        data_loader = DataLoader(dataset=test_dataset, num_workers=1,
                                 batch_size=BATCH_SIZE,
                                 shuffle=False)
        data_loaders.append(data_loader)

    lx, px = utils.predict_tta(model, data_loaders)
    data = {
        'lx': lx.cpu(),
        'px': px.cpu(),
    }
    torch.save(data, 'models_trained/{}_{}_{}/val_prediction_{}.pth'.format(args.name, args.aug, args.alpha, args.name))


def train(args):

    train_dataset = FurnitureDataset('train', transform=preprocess_with_augmentation)
    val_dataset = FurnitureDataset('val', transform=preprocess)
    training_data_loader = DataLoader(dataset=train_dataset, num_workers=8,
                                      batch_size=BATCH_SIZE,
                                      shuffle=True)
    validation_data_loader = DataLoader(dataset=val_dataset, num_workers=1,
                                        batch_size=BATCH_SIZE,
                                        shuffle=False)

    model = get_model(args.name)

    class_weight = np.load('./class_weight.npy')

    #criterion = nn.CrossEntropyLoss(weight=torch.FloatTensor(class_weight)).cuda()
    criterion = nn.CrossEntropyLoss().cuda()
    #criterion = FocalLoss(alpha=alpha, gamma=0).cuda()

    nb_learnable_params = sum(p.numel() for p in model.fresh_params())
    print(f'[+] nb learnable params {nb_learnable_params}')

    min_loss = float("inf")
    lr = 0
    patience = 0
    for epoch in range(30):
        print(f'epoch {epoch}')
        if epoch == 1:
            lr = 0.00003
            print(f'[+] set lr={lr}')
        if patience == 2:
            patience = 0
            model.load_state_dict(torch.load('models_trained/{}_{}_{}/best_val_weight_{}.pth'.format(args.name, args.aug, args.alpha, args.name)))
            lr = lr / 10
            if lr < 3e-6:
                lr = 3e-6
            print(f'[+] set lr={lr}')
        if epoch == 0:
            lr = 0.001
            print(f'[+] set lr={lr}')
            optimizer = torch.optim.Adam(model.fresh_params(), lr=lr)
        else:
            optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=0.0001)

        running_loss = RunningMean()
        running_score = RunningMean()

        model.train()
        pbar = tqdm(training_data_loader, total=len(training_data_loader))
        for inputs, labels in pbar:
            batch_size = inputs.size(0)

            inputs = Variable(inputs)
            labels = Variable(labels)
            if use_gpu:
                inputs = inputs.cuda()
                labels = labels.cuda()

            optimizer.zero_grad()

            if args.aug:
                inputs, targets_a, targets_b, lam = mixup_data(inputs, labels, args.alpha, use_gpu)

            outputs = model(inputs)

            if args.aug:
                loss_func = mixup_criterion(targets_a, targets_b, lam)
                loss = loss_func(criterion, outputs)
            else:
                loss = criterion(outputs, labels)

            _, preds = torch.max(outputs.data, dim=1)
            running_loss.update(loss.data[0], 1)

            if args.aug:
                running_score.update(batch_size - lam * preds.eq(targets_a.data).cpu().sum() - (1 - lam) * preds.eq(targets_b.data).cpu().sum(), batch_size)
            else:
                running_score.update(torch.sum(preds != labels.data), batch_size)

            loss.backward()
            optimizer.step()

            pbar.set_description(f'{running_loss.value:.5f} {running_score.value:.3f}')
        print(f'[+] epoch {epoch} {running_loss.value:.5f} {running_score.value:.3f}')

        lx, px = utils.predict(model, validation_data_loader)
        log_loss = criterion(Variable(px), Variable(lx))
        log_loss = log_loss.data[0]
        _, preds = torch.max(px, dim=1)
        accuracy = torch.mean((preds != lx).float())
        print(f'[+] val {log_loss:.5f} {accuracy:.3f}')

        if log_loss < min_loss:
            torch.save(model.state_dict(), 'models_trained/{}_{}_{}/best_val_weight_{}.pth'.format(args.name, args.aug, args.alpha, args.name))
            print(f'[+] val score improved from {min_loss:.5f} to {log_loss:.5f}. Saved!')
            min_loss = log_loss
            patience = 0
        else:
            patience += 1


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('mode', choices=['train', 'predict'])
    parser.add_argument('name', choices=['densenet201', 'inceptionresnetv2', 'senet154', 'nasnetlarge', 'inceptionv4', 'se_resnext101_32x4d'])
    parser.add_argument('aug', type=bool)
    parser.add_argument('alpha', type=float)
    args = parser.parse_args()
    print(f'[+] start `{args.mode}` using `{args.name}` augmentation `{args.aug}` alpha `{args.alpha}`')

    cudnn.benchmark = True
    print(f'[+] cudnn `{cudnn.benchmark}`')

    if args.mode == 'train':
        train(args)
    elif args.mode == 'predict':
        predict(args)
