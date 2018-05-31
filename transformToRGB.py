import numpy as np
from PIL import Image
import os

dir = 'data/test/'

a = os.listdir(dir)
list.sort(a)

for i in range(len(a)):
    with Image.open(dir+a[i]) as image:
        img = np.array(image)
        if len(img.shape) !=3 or img.shape[2] != 3:
            # rgb_image = image.convert('RGB')
            # rgb_image.save(dir+a[i])
            print(img.shape, a[i])
    # print(i, ' is RGB')