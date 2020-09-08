from xml.dom import minidom
from glob import glob
import os
import cv2

root = r'C:\Users\nguye\Desktop\Trung\dataset\train'
img_dirs = glob(os.path.join(root, 'imgs', '*.JPG'))
for img_dir in img_dirs:
    name = os.path.basename(img_dir).split('.')[0]
    xml_dir = os.path.join(root, 'lbls', name + '.xml')

    anno = minidom.parse(xml_dir)
    xmin_items = anno.getElementsByTagName('xmin')
    ymin_items = anno.getElementsByTagName('ymin')
    xmax_items = anno.getElementsByTagName('xmax')
    ymax_items = anno.getElementsByTagName('ymax')


    img = cv2.imread(img_dir)
    print(xml_dir)
    for i in range(len(xmin_items)):
        xmin = xmin_items[i].childNodes[0].data
        ymin = ymin_items[i].childNodes[0].data
        xmax = xmax_items[i].childNodes[0].data
        ymax = ymax_items[i].childNodes[0].data
        print(xmin, ymin, xmax, ymax)
        cv2.rectangle(img, (int(xmin), int(ymin)), (int(xmax), int(ymax)), (0,255,0), 8)

    img = cv2.resize(img, (480,640))
    cv2.imshow('img',img)
    cv2.waitKey(0)



