import os

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import itertools
import multiprocessing

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

    # countTable = get_distribute('./data/train')
    # df = pd.DataFrame(countTable, index=[str(i+1) for i in range(len(countTable))], columns=['FreqCount'])
    # #df.plot(kind='bar', title='Train Statistics')
    #
    # countTable = get_distribute('./data/validation')
    # df = pd.DataFrame(countTable, index=[str(i + 1) for i in range(len(countTable))], columns=['FreqCount'])
    # #df.plot(kind='bar', title='Validation Statistics')
    # #plt.show()

    val_pred = torch.load('models_trained/densenet201_True_0.2/val_prediction_densenet201.pth')
    val_prob = F.softmax(Variable(val_pred['px']), dim=1).data.numpy()


    val_pred_2 = torch.load('models_trained/inceptionresnetv2_True_0.2/val_prediction_inceptionresnetv2.pth')
    val_prob_2 = F.softmax(Variable(val_pred_2['px']), dim=1).data.numpy()

    val_pred_3 = torch.load('models_trained/senet154_True_0.2/val_prediction_senet154.pth')
    val_prob_3 = F.softmax(Variable(val_pred_3['px']), dim=1).data.numpy()

    val_pred_4 = torch.load('models_trained/nasnetlarge_True_0.2/val_prediction_nasnetlarge.pth')
    val_prob_4 = F.softmax(Variable(val_pred_4['px']), dim=1).data.numpy()

    val_pred_5 = torch.load('models_trained/inceptionv4_True_0.2/val_prediction_inceptionv4.pth')
    val_prob_5 = F.softmax(Variable(val_pred_5['px']), dim=1).data.numpy()

    val_pred_6 = torch.load('models_trained/se_resnext101_32x4d_True_0.2/val_prediction_se_resnext101_32x4d.pth')
    val_prob_6 = F.softmax(Variable(val_pred_6['px']), dim=1).data.numpy()

    val_gt = val_pred['lx'].numpy()

    iterlen = 11
    a = range(iterlen)
    b = range(iterlen)
    c = range(iterlen)
    d = range(iterlen)
    e = range(iterlen)
    f = range(iterlen)

    paramlist = list(itertools.product(a,b,c,d,e,f))

    def cal_acc(params):
        i = params[0]
        j = params[1]
        k = params[2]
        l = params[3]
        m = params[4]
        n = params[5]

        val_com = i * val_prob + j * val_prob_2 + k * val_prob_3 + l * val_prob_4 + m * val_prob_5 + n * val_prob_6
        val_com = val_com.mean(axis=2)
        val_predicted = np.argmax(val_com, axis=1)
        cls_acc = accuracy_score(val_gt, val_predicted)
        print(i,j,k,l,m,n)
        return (i,j,k,l,m,n), cls_acc

    print('calculate start ...')
    pool = multiprocessing.Pool()

    res = pool.map(cal_acc, paramlist)

    print('calculate done!')

    index = [data[0] for data in res]
    acc = [data[1] for data in res]

    # 86.5825,87.3328, 87.1134, 87.4767, 86.6812, 86.0650, 86.0043
    #val_prob = 1.0 *val_prob + 1.0 *val_prob_2 + 1.0 * val_prob_3 + 1.0 * val_prob_4 + 1.0 * val_prob_5 + 1.0 * val_prob_6
    # print(val_prob.shape, val_prob_2.shape, val_prob_3.shape)
    # start = time()
    # val_com = [i * val_prob + j * val_prob_2 + k * val_prob_3 + l * val_prob_4 + m * val_prob_5 + n * val_prob_6 + o * val_prob_7 for i in range(11) for j in range(11) for k in range(11) for l in range(11) for m in range(11) for n in range(11) for o in range(11)]
    # print(time() - start)
    # val_com = [data.mean(axis=2) for data in val_com]
    # val_predicted = [np.argmax(data) for data in val_com]
    #
    # index = [(i,j,k,l,m,n,o) for i in range(11) for j in range(11) for k in range(11) for l in range(11) for m in range(11) for n in range(11) for o in range(11)]
    # val_gt = val_pred['lx'].numpy()
    # acc = [accuracy_score(val_gt, data) for data in val_com]

    # index, acc = [], []
    # val_gt = val_pred['lx'].numpy()
    # for i in range(0,11):
    #     for j in range(0,11):
    #         for k in range(0,11):
    #             for l in range(0,11):
    #                 for m in range(0,11):
    #                     for n in range(0,11):
    #                         for o in range(0,11):
    #                             #start = time()
    #                             val_com = i * val_prob + j * val_prob_2 + k * val_prob_3 + l * val_prob_4 + m * val_prob_5 + n * val_prob_6 + o * val_prob_7
    #                             val_com = val_com.mean(axis=2)
    #                             val_predicted = np.argmax(val_com, axis=1)
    #                             cls_acc = accuracy_score(val_gt, val_predicted)
    #                             # cf = confusion_matrix(val_gt, val_predicted).astype(float)
    #                             # cls_cnt = cf.sum(axis=1)
    #                             # cls_hit = np.diag(cf)
    #                             # cls_acc = cls_hit / cls_cnt
    #                             # cls_acc = np.mean(cls_acc)
    #                             index.append((i,j,k,l,m,n,o))
    #                             acc.append(cls_acc)
    #                             maxind = np.argmax(acc)
    #                             print(i, j, k, l, m, n, o,  cls_acc, '   ', acc[maxind], index[maxind], 'done')
    #                             #print(time() - start)
    maxind = np.argmax(acc)
    print(acc[maxind], index[maxind])
    #
    val_prob = index[maxind][0] * val_prob + index[maxind][1] * val_prob_2 + index[maxind][2] * val_prob_3 + index[maxind][3] * val_prob_4 + index[maxind][4] * val_prob_5 + index[maxind][5] * val_prob_6
    # print(val_prob.shape)

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
    val_gt_2 = val_pred_2['lx'].numpy()
    print(sum(val_gt==val_gt_2)/len(val_gt))

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
    #df.plot(kind='bar', title='Validation Acc')

    print([[1,4][i] for i in cls_acc <= 0.8])

    #print(countTable.shape, cls_acc.shape)
    #combined = np.concatenate((countTable/max(countTable), cls_acc.reshape(cls_acc.shape[0], 1)), axis=1)
    #df = pd.DataFrame(combined, columns=['freq', 'acc'])
    #df.plot(kind='bar', title='Validation freq & Acc')

    #plt.show()
    #print(cls_acc)



