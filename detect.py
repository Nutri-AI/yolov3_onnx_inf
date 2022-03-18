# YOLOv3 🚀 by Ultralytics, GPL-3.0 license
# edited by hyeniii
"""
Run inference on images, videos, directories, streams, etc.

Usage:
    $ python path/to/detect.py --weights yolov3.pt --source 0  # webcam
                                                             img.jpg  # image
                                                             vid.mp4  # video
                                                             path/  # directory
                                                             path/*.jpg  # glob
                                                             'https://youtu.be/Zgi9g1ksQHc'  # YouTube
                                                             'rtsp://example.com/media.mp4'  # RTSP, RTMP, HTTP stream
"""

import argparse
import os
import sys
from pathlib import Path
from webbrowser import get
import numpy as np
from PIL import Image
import time
from io import BytesIO

import cv2
import onnxruntime

FILE = Path(__file__).resolve() # file = path to train.py
ROOT = FILE.parents[0]  # root directory = file train.py is contained in [0]=1 folder [1] = 2 folder ...etc
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # get relative path

# from models.common import DetectMultiBackend
from utils.datasets import IMG_FORMATS, VID_FORMATS, LoadImages 
from utils.general import (LOGGER, check_img_size, check_requirements, colorstr, check_suffix,
                           non_max_suppression, scale_coords, xyxy2xywh)
from utils.plots import Annotator, save_one_box, colors
# from utils.torch_utils import select_device
from utils.s3 import get_img

import os

class DetectMultiBackend():
    #  MultiBackend class for python inference on various backends
    def __init__(self, weights='yolov3.pt', device=None, dnn=True):
        # Usage:
        #   ONNX Runtime:           *.onnx
        #   OpenCV DNN:             *.onnx with dnn=True
        super().__init__()
        w = str(weights[0] if isinstance(weights, list) else weights)
        suffix, suffixes = Path(w).suffix.lower(), ['.pt', '.onnx']
        check_suffix(w, suffixes)  # check weights have acceptable suffix
        pt, onnx = (suffix == x for x in suffixes)  # backend booleans
        stride, names = 64, [f'class{i}' for i in range(1000)]  # assign defaults

        if onnx:  # ONNX Runtime
            LOGGER.info(f'Loading {w} for ONNX Runtime inference...')
            check_requirements(('onnx', 'onnxruntime-gpu' if onnxruntime.get_device()=='GPU' else 'onnxruntime'))
            #2022.03.02 edited by nhwh
            names = ["pork_belly","ramen","bibimbap","champon","cold_noodle","cutlassfish","egg_custard","egg_soup","jajangmyeon","kimchi_stew","multigrain_rice",
                     "oxtail_soup","pickled spianch","pizza","pork_feet","quail_egg_stew","seasoned_chicken","seaweed_soup","soy_bean_paste_soup","stewed_bean","stewed_lotus_stew",
                     "stir_fried_anchovy","sitr_fried_pork","salad","ice_americano","Bottled_Beer","Canned_Beer","Draft_Beer","Fried_Chicken","Tteokbokki","Cabbage_Kimchi","Radish_Kimchi"]
            #2022.03.02 edited by nhwh
            session = onnxruntime.InferenceSession(w, providers=['CUDAExecutionProvider','TensorrtExecutionProvider', 'CPUExecutionProvider'])
        self.__dict__.update(locals())  # assign all variables to self

    def forward(self, im, augment=False, visualize=False, val=False):
        #  MultiBackend inference
        b, ch, h, w = im.shape  # batch, channel, height, width
        if self.onnx:  # ONNX
            y = self.session.run([self.session.get_outputs()[0].name], {self.session.get_inputs()[0].name: im})[0]
#        y = torch.tensor(y)
        return (y, []) if val else y

   
# @torch.no_grad()
def run(image,
        weights=ROOT / 'best.onnx',  # model.pt path(s)
        imgsz=[640,640],  # inference size (pixels)
        conf_thres=0.25,  # confidence threshold
        iou_thres=0.45,  # NMS IOU threshold
        max_det=1000,  # maximum detections per image
        device='',  # cuda device, i.e. 0 or 0,1,2,3 or cpu
        view_img=False,  # show results
        save_crop=False,  # save cropped prediction boxes
        nosave=False,  # do not save images/videos
        classes=None,  # filter by class: --class 0, or --class 0 2 3
        agnostic_nms=False,  # class-agnostic NMS
        augment=False,  # augmented inference
        visualize=False,  # visualize features
        line_thickness=3,  # bounding box thickness (pixels)
        hide_labels=False,  # hide labels
        hide_conf=False,  # hide confidences
        half=False,  # use FP16 half-precision inference
        dnn=False,  # use OpenCV DNN for ONNX inference
        save_txt=False,  # save results to *.txt
        save_conf=False,  # save confidences in --save-txt labels
        update=False,  # update all models
        exist_ok=False,  # existing project/name ok, do not increment
        ):
    
    img = Image.open(BytesIO(image))
    source = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
    save_img = not nosave 

    # Load model
    device = 'cuda:0' if onnxruntime.get_device()== 'GPU' else 'cpu'
    model = DetectMultiBackend(weights, device=device, dnn=dnn)
    stride, names, pt, onnx = model.stride, model.names, model.pt, model.onnx
    imgsz = check_img_size(imgsz, s=stride)  # check image size
    
    # Dataloader
    im, im0s , s = LoadImages(source, img_size=imgsz, stride=stride, auto=pt)
    bs = 1  # batch_size

    dt, seen = [0.0, 0.0, 0.0], 0

    t1 = time.time()
#        im = torch.from_numpy(im).to(device) ########
    im = im.half() if half else np.float32(im)  # uint8 to fp16/32
    im /= 255  # 0 - 255 to 0.0 - 1.0
    if len(im.shape) == 3:
        im = im[None]  # expand for batch dim
    t2 = time.time()
    dt[0] += t2 - t1

    # Inference
    pred = model.forward(im, augment=augment, visualize=visualize) #forward
    t3 = time.time()
    dt[1] += t3 - t2

    # NMS
    pred = non_max_suppression(pred, conf_thres, iou_thres, classes, agnostic_nms, max_det=max_det)
    dt[2] += time.time() - t3

    # Process predictions
    for i, det in enumerate(pred):  # per image
        seen += 1
        im0 = im0s.copy()

        s += '%gx%g ' % im.shape[2:]  # print string
#       gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]  # normalization gain whwh
        gn = np.array(im0.shape)[[1, 0, 1, 0]]  # normalization gain whwh
        annotator = Annotator(im0, line_width=line_thickness, example=str(names))
        if len(det):
            # Rescale boxes from img_size to im0 size
            det[:, :4] = scale_coords(im.shape[2:], det[:, :4], im0.shape).round()
            # Print results
            for c in np.unique(det[:, -1]):
                n = (det[:, -1] == c).sum()  # detections per class
                s += f"{n} {names[int(c)]}{'s' * (n > 1)}, "  # add to string
                

            # Write results
            for *xyxy, conf, cls in reversed(det):

                if save_img or save_crop or view_img:  # Add bbox to image
                    c = int(cls)  # integer class
                    label = None if hide_labels else (names[c] if hide_conf else f'{names[c]} {conf:.2f}')
                    annotator.box_label(xyxy, label, color=colors(c, True))

        # Print time (inference-only)
        LOGGER.info(f'{s}Done. ({t3 - t2:.3f}s)')

        # Stream results
        im0 = annotator.result()     
        _, im0_byte = cv2.imencode('.jpg', im0)
        
        # im0_re = Image.open(BytesIO(im0_byte))
        # im0_re_ndarr = cv2.cvtColor(np.array(im0_re), cv2.COLOR_RGB2BGR)
        
        # test = cv2.resize(im0_re_ndarr, (1000, 1000))
        # cv2.imshow(f'{names[c]} {conf:.2f}', test)
        # cv2.waitKey(6000)

    # Print results
    t = tuple(x / seen * 1E3 for x in dt)  # speeds per image
    LOGGER.info(f'Speed: %.1fms pre-process, %.1fms inference, %.1fms NMS per image at shape {(1, 3, *imgsz)}' % t)
    return im0_byte, c

def main(img_b):
    check_requirements(exclude=('tensorboard', 'thop'))
    run(image = img_b)

# if __name__ == '__main__':
#     main()



# def from_s3(self, bucket, key):
#     file_byte_string = self.s3.get_object(Bucket=bucket, Key=key)['Body'].read()
#     return Image.open(BytesIO(file_byte_string))