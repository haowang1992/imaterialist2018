import os

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import itertools

import torch
import torch.nn.functional as F
from torch.autograd import Variable
from sklearn.metrics import accuracy_score, confusion_matrix
from time import time


def plot_confusion_matrix(cm, classes,
                          normalize=True,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')

def get_distribute(path):
    firstdir = os.listdir(path)
    countTable = np.zeros((len(firstdir),1))
    for i in range(len(firstdir)):
        secondfile = os.listdir(path+'/'+firstdir[i])
        count = len(secondfile)
        countTable[int(firstdir[i])-1] = count
    return countTable

if __name__ == "__main__":

    five_crop = False

    countTable = get_distribute('./data/train')
    df = pd.DataFrame(countTable, index=[str(i+1) for i in range(len(countTable))], columns=['FreqCount'])
    df.plot(kind='bar', title='Train Statistics')

    countTable = get_distribute('./data/validation')
    df = pd.DataFrame(countTable, index=[str(i + 1) for i in range(len(countTable))], columns=['FreqCount'])
    df.plot(kind='bar', title='Validation Statistics')
    #plt.show()

    val_pred = torch.load('val_prediction_densenet201.pth')
    val_prob = F.softmax(Variable(val_pred['px']), dim=1).data.numpy()



    if five_crop:
        nsample, nclass, naug = val_prob.shape[0], val_prob.shape[1], val_prob.shape[2]
        #print(nsample, nclass, naug)
        val_prob = val_prob.transpose(1,2,0)
        val_prob = val_prob.reshape(nclass, naug, int(nsample/5), -1)
        #print(val_prob.shape)
        val_prob = val_prob.mean(axis=-1)
        #print(val_prob.shape)
        val_prob = val_prob.transpose(2,0,1)
        val_prob = val_prob.mean(axis=2)
        print(val_prob, val_prob.shape)
    else:
        val_prob = val_prob.mean(axis=2)

    if five_crop:
        val_gt = val_pred['lx'].numpy()
        val_gt = val_gt.reshape(int(val_gt.shape[0]/5), -1)
        val_gt = val_gt.mean(axis=-1)
    else:
        val_gt = val_pred['lx'].numpy()

    val_gt = val_pred['lx'].numpy()
    val_predicted = np.argmax(val_prob, axis=1)

    #acc = accuracy_score(val_gt, val_predicted)
    print(val_gt, val_gt.shape)
    cf = confusion_matrix(val_gt, val_predicted).astype(float)
    plot_confusion_matrix(cf, classes=[str(i+1) for i in range(len(set(val_gt)))])
    cls_cnt = cf.sum(axis=1)
    cls_hit = np.diag(cf)
    cls_acc = cls_hit / cls_cnt

    print(np.mean(cls_acc))
    df = pd.DataFrame(cls_acc)
    df.plot(kind='bar', title='Validation Acc')

    print([[1,4][i] for i in cls_acc <= 0.8])

    #print(countTable.shape, cls_acc.shape)
    #combined = np.concatenate((countTable/max(countTable), cls_acc.reshape(cls_acc.shape[0], 1)), axis=1)
    #df = pd.DataFrame(combined, columns=['freq', 'acc'])
    #df.plot(kind='bar', title='Validation freq & Acc')

    plt.show()
    #print(cls_acc)



