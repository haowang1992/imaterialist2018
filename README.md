## Kaggle: imaterialist-challenge-furniture-2018

Competition: image classification with 128 classes
Link: https://www.kaggle.com/c/imaterialist-challenge-furniture-2018

# How to run
1. Download data from kaggle to `./data/`
2. Train model `python cnn_runner_pth.py train model_name augmentation_mixup mixup_alpha`
3. Predict `python cnn_runner_pth.py predict model_name augmentation_mixup mixup_alpha`
4. Calculate weights for finetuned models `python statistics.py`
5. Generate submission `python submit.py`

# Ensemble
* ensemble denset201, inceptionv4, inceptionresnetv2, senet154, se_resnext101, naslarge
* hflip at test

# Result
* Public Borard: 9/436 Private Board: 13/436
* models trained at [model](https://pan.baidu.com/s/1bA353cQcfm2jrv40G4n0aA)

# Related Projects
- [skrypka/imaterialist-furniture-2018] : winner of imaterialist furniture 2018


[skrypka/imaterialist-furniture-2018](https://github.com/skrypka/imaterialist-furniture-2018)