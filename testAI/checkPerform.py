from ctypes import *
import cv2
import numpy as np
import argparse
import os
import threading
import time
import live555

class IMAGE(Structure):
    _fields_ = [("w", c_int),
                ("h", c_int),
                ("c", c_int),
                ("data", POINTER(c_float))]

class BOX(Structure):
    _fields_ = [("x", c_float),
                ("y", c_float),
                ("w", c_float),
                ("h", c_float)]

class DETECTION(Structure):
    _fields_ = [("cl", c_int),
                ("bbox", BOX),
                ("prob", c_float),
                ("name", c_char*20),
                ]

lib = CDLL("build/libdarknetTR.so", RTLD_GLOBAL)
load_network = lib.load_network
load_network.argtypes = [c_char_p, c_int, c_int]
load_network.restype = c_void_p

copy_image_from_bytes = lib.copy_image_from_bytes
copy_image_from_bytes.argtypes = [IMAGE,c_char_p]

make_image = lib.make_image
make_image.argtypes = [c_int, c_int, c_int]
make_image.restype = IMAGE

do_inference = lib.do_inference
do_inference.argtypes = [c_void_p, IMAGE]

get_network_boxes = lib.get_network_boxes
get_network_boxes.argtypes = [c_void_p, c_float, c_int, POINTER(c_int)]
get_network_boxes.restype = POINTER(DETECTION) 

def resizePadding(image, height, width):
    desized_size = height, width
    old_size = image.shape[:2]
    max_size_idx = old_size.index(max(old_size))
    ratio = float(desized_size[max_size_idx]) / max(old_size)
    new_size = tuple([int(x * ratio) for x in old_size])

    if new_size > desized_size:
        min_size_idx = old_size.index(min(old_size))
        ratio = float(desized_size[min_size_idx]) / min(old_size)
        new_size = tuple([int(x * ratio) for x in old_size])

    image = cv2.resize(image, (new_size[1], new_size[0]))
    delta_w = desized_size[1] - new_size[1]
    delta_h = desized_size[0] - new_size[0]
    top, bottom = delta_h // 2, delta_h - (delta_h // 2)
    left, right = delta_w // 2, delta_w - (delta_w // 2)

    image = cv2.copyMakeBorder(image, top, bottom, left, right, cv2.BORDER_CONSTANT)
    return image

def detect_image(net, meta, darknet_image, thresh=.5):
    num = c_int(0)

    pnum = pointer(num)
    do_inference(net, darknet_image)
    dets = get_network_boxes(net, 0.5, 0, pnum)
    res = []
    for i in range(pnum[0]):
        b = dets[i].bbox
        res.append((dets[i].name.decode("ascii"), dets[i].prob, (b.x, b.y, b.w, b.h)))

    return res


def loop_detect(detect_m, video_path):
    stream = cv2.VideoCapture(video_path)
    start = time.time()
    cnt = 0
    while stream.isOpened():
        ret, image = stream.read()
        if ret is False:
            break
        # image = resizePadding(image, 512, 512)
        # frame_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = cv2.resize(image,
                           (512, 512),
                           interpolation=cv2.INTER_LINEAR)
        detections = detect_m.detect(image, need_resize=False)
        cnt += 1
        for det in detections:
            print(det)
    end = time.time()
    print("frame:{},time:{:.3f},FPS:{:.2f}".format(cnt, end-start, cnt/(end-start)))
    stream.release() 


class YOLO4RT(object):
    def __init__(self,
                 input_size=512,
                 weight_file='build/yolo4_fp16.rt', 
                 nms=0.2,
                 conf_thres=0.3,
                 device='cuda'):
        self.input_size = input_size
        self.metaMain =None
        self.model = load_network(weight_file.encode("ascii"), 80, 1)
        self.darknet_image = make_image(input_size, input_size, 3)
        self.thresh = conf_thres
        # self.resize_fn = ResizePadding(input_size, input_size)
        # self.transf_fn = transforms.ToTensor()

    def detect(self, image, need_resize=True, expand_bb=5):
        try:
            if need_resize:
                frame_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                image = cv2.resize(frame_rgb,
                                   (self.input_size, self.input_size),
                                   interpolation=cv2.INTER_LINEAR)
            frame_data = image.ctypes.data_as(c_char_p)
            copy_image_from_bytes(self.darknet_image, frame_data)

            detections = detect_image(self.model, self.metaMain, self.darknet_image, thresh=self.thresh)

            # cvDrawBoxes(detections, image)
            # cv2.imshow("1", image)
            # cv2.waitKey(1)
            # detections = self.filter_results(detections, "person")
            return detections
        except Exception as e_s:
            print(e_s)

def parse_args():
    parser = argparse.ArgumentParser(description='tkDNN detect')
    parser.add_argument('--weight', help='rt file path', default = 'build/yolo4_fp16.rt')
    parser.add_argument('--video',  type=str, help='video part', default = './demo/yolo_test.mp4')
    args = parser.parse_args()

    return args
    
def oneFrame(codecName, bytes, sec, streamURL, hihi):      
    if hihi != "first":
       img1 = cv2.imread('croped_left.jpg') # cv2.imdecode(np.frombuffer(bytes, dtype='uint8'), cv2.IMREAD_COLOR)
       height = img1.shape[0]
       width = img1.shape[1]
       img = img1[1:height, 1: width]
        
       detections = detect_m.detect(img, need_resize=True)
       for det in detections:
           print('====', det[2])           
           x1 = int(det[2][0]*width/512)
           y1 = int(det[2][1]*height/512)
           x2 = int((det[2][0] + det[2][2])*width/512)
           y2 = int((det[2][1] + det[2][3])*height/512)
           color = (255, 0, 0)
           img = cv2.rectangle(img, (int(x1),int(y1)), (int(x2),int(y2)), color, 2)
           
       cv2.resize(img, (width, height), interpolation = cv2.INTER_AREA) 
       cv2.imwrite('hihi.jpg', img)
       # cv2.waitKey(1)

def hihi(vcl, dcm, hhihi):
    print('STATUS: ', hhihi)

if __name__ == '__main__':
    args = parse_args()
    url1 = 'rtsp://admin:SCC_gundam@siliconcube.asuscomm.com:554/Streaming/Channels/102/?transportmode=unicast'
    url = 'rtsp://siliconcube.asuscomm.com:8555/test'
    detect_m = YOLO4RT(weight_file=args.weight)
    
#    cap = cv2.VideoCapture(url1)
    
#    while True:
#      ret, frame = cap.read()
#      gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
#      cv2.imshow('frame',frame)
#      detections = detect_m.detect(frame, need_resize=True)
#      print('====', detections)
#      if cv2.waitKey(1) & 0xFF == ord('q'):
#          break
          
          
    live555.startRTSP(url1, oneFrame, hihi, False, "CAM1")
    t = threading.Thread(target=live555.runEventLoop, args=())
    t.setDaemon(True)
    t.start()
    endTime = time.time() + 100
    while time.time() < endTime:
        time.sleep(0.5)
      
    # t = Thread(target=loop_detect, args=(detect_m, args.video), daemon=True)

    # thread1 = myThread(loop_detect, [detect_m])

    # Start new Threads
    # t.start()
    # t.join()
