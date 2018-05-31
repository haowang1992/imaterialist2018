import torch
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np
import pandas as pd
from misc import FurnitureDataset, preprocess

five_crop = False

test_dataset = FurnitureDataset('test', transform=preprocess)

test_pred = torch.load('models_trained/densenet201_True_0.2/test_prediction_densenet201.pth')
#test_pred = torch.load('test_prediction_densenet201.pth')
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

test_pred_2 = torch.load('models_trained/inceptionresnetv2_True_0.2/test_prediction_inceptionresnetv2.pth')
test_prob_2 = F.softmax(Variable(test_pred_2['px']), dim=1).data.numpy()
test_prob_2 = test_prob_2.mean(axis=2)
test_id_ord_2 = test_pred_2['lx'].numpy()

test_pred_3 = torch.load('models_trained/senet154_True_0.2/test_prediction_senet154.pth')
test_prob_3 = F.softmax(Variable(test_pred_3['px']), dim=1).data.numpy()
test_prob_3 = test_prob_3.mean(axis=2)
test_id_ord_3 = test_pred_3['lx'].numpy()

test_pred_4 = torch.load('models_trained/nasnetlarge_True_0.2/test_prediction_nasnetlarge.pth')
test_prob_4 = F.softmax(Variable(test_pred_4['px']), dim=1).data.numpy()
test_prob_4 = test_prob_4.mean(axis=2)
test_id_ord_4 = test_pred_4['lx'].numpy()

test_pred_5 = torch.load('models_trained/inceptionv4_True_0.2/test_prediction_inceptionv4.pth')
test_prob_5 = F.softmax(Variable(test_pred_5['px']), dim=1).data.numpy()
test_prob_5 = test_prob_5.mean(axis=2)
test_id_ord_5 = test_pred_5['lx'].numpy()

test_pred_6 = torch.load('models_trained/se_resnext101_32x4d_True_0.2/test_prediction_se_resnext101_32x4d.pth')
test_prob_6 = F.softmax(Variable(test_pred_6['px']), dim=1).data.numpy()
test_prob_6 = test_prob_6.mean(axis=2)
test_id_ord_6 = test_pred_6['lx'].numpy()


print(sum(test_id_ord == test_id_ord_2)/ len(test_id_ord))
print(sum(test_id_ord == test_id_ord_3)/ len(test_id_ord))
print(sum(test_id_ord_2 == test_id_ord_3)/ len(test_id_ord))

# 7,5,9 0.12630
# 15,8,15 0.12656
# 5, 3, 5 0.12630
# 8, 5, 9
# 5, 6, 4, 2
# 6,8,7,7 88.7637
# 2,2,5,4 88.7932 12.421
# 2,2,5,4 88.7932 12.421 (86.5825, 87.3328, 87.1134, 87.4767)
# 0,3,4,3 88.6311 12.604 (86.5825, 87.3328, 87.1134, 87.4767)
# 0,8,7,8,9,0 88.69772 12.421
# 8,7,2,9,8,0 88.8093 12.317
# 0, 2, 10, 5, 4, 1 88.8544 12.187
# 0, 2, 10, 5,4, 1, 4 88.8589
test_prob = (0.0 * test_prob + 2.0 * test_prob_2 + 10.0 * test_prob_3 + 5.0 * test_prob_4 + 4.0 * test_prob_5 + 1.0 * test_prob_6)

test_predicted = np.argmax(test_prob, axis=1)
test_predicted += 1
result = test_predicted

sx = pd.read_csv('data/sample_submission_randomlabel.csv')
# sx.loc[sx.id.isin(test_dataset.data.image_id), 'predicted'] = result
sx.loc[test_id_ord - 1, 'predicted'] = result
sx.to_csv('sx.csv', index=False)
