import cv2
from glob import glob 
import os

root = 'Full/'

img_dirs = glob(os.path.join(root, '*.jpg'))

for im in img_dirs: 
    name = root + os.path.basename(im).split('.')[0] + '.txt'
    if not os.path.exists(name):
        print(name) 
        # os.remove(im)