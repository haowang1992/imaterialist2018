import torch
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np
import pandas as pd
from misc import FurnitureDataset, preprocess

five_crop = False

test_dataset = FurnitureDataset('test', transform=preprocess)

test_pred = torch.load('test_prediction_densenet201.pth')
test_prob = F.softmax(Variable(test_pred['px']), dim=1).data.numpy()

if five_crop:
    nsamples, nclass, naug = test_prob.shape
    test_prob = test_prob.transpose(1,2,0)
    #print(test_prob.shape)
    test_prob = test_prob.reshape(nclass, naug, -1, 5)
    test_prob = test_prob.mean(axis=-1)
    test_prob = test_prob.transpose(2,0,1)
#print(test_prob.shape)
test_prob = test_prob.mean(axis=2)

test_id_ord = test_pred['lx'].numpy()
if five_crop:
    test_id_ord = test_id_ord.reshape(int(test_id_ord.shape[0]/5), -1)
    test_id_ord = test_id_ord.mean(axis=-1)
#print(test_id_ord)


test_predicted = np.argmax(test_prob, axis=1)
test_predicted += 1
result = test_predicted

sx = pd.read_csv('data/sample_submission_randomlabel.csv')
# sx.loc[sx.id.isin(test_dataset.data.image_id), 'predicted'] = result
sx.loc[test_id_ord - 1, 'predicted'] = result
sx.to_csv('sx.csv', index=False)
