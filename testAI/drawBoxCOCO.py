import cv2
from glob import glob 
import os

root = 'train/'

img_dirs = glob(os.path.join(root, '*.jpg'))

list_class = ['motorcycle','bicycle','car','person','bus','truck']

for im in img_dirs:
    name = root + os.path.basename(im).split('.')[0]
    
    f = open(name + '.txt','r')
    img = cv2.imread(name + '.jpg')

    hei, wid, _ = img.shape
    datas = f.read().splitlines()

    for data in datas:
        array = []
        array.append([x for x in data.split()])
        x_C = float(array[0][1])*wid
        y_C = float(array[0][2])*hei
        wi_C = float(array[0][3])*wid
        he_C = float(array[0][4])*hei
        x1 = int(x_C - wi_C//2)
        y1 = int(y_C - he_C//2)
        x2 = int(x_C + wi_C//2)
        y2 = int(y_C + he_C//2)
        cv2.rectangle(img, (x1,y1), (x2,y2), (0,255,0), 2)
        cv2.putText(img, list_class[int(array[0][0])], (x1, y1+30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (36, 255, 12), 1)
    cv2.imshow('hihi', img)
    cv2.waitKey(0)
