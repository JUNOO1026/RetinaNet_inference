import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import time

import torch
from models.retinaface import RetinaFace
from data import cfg_mnet, cfg_re50

from layers.functions.prior_box import PriorBox
from utils.nms.py_cpu_nms import py_cpu_nms
from utils.box_utils import decode, decode_landm

weight_path = "D:/DL_models/Pytorch_Retinaface/model_save/face_detection.pth"
cfg = cfg_mnet # mobile0.25 (cfg_mnet) or resnet50 (cfg_re50)
resize = 1
confidence_threshold = 0.02
top_k = 5000
nms_threshold = 0.4
keep_top_k = 750
vis_thres = 0.6

device = "cuda"
model = RetinaFace(cfg, phase='test').to(device)
model.load_state_dict(torch.load(weight_path, map_location=device))
model.eval()
print("Model Loaded!")

def retinaface_inf(test_img, model):
    img = np.float32(test_img)
    print(img.shape)
    im_height, im_width, _ = img.shape


    scale = torch.Tensor([img.shape[1], img.shape[0], img.shape[1], img.shape[0]])

    print("#########################")
    img -= (104, 117, 123)
    print(img.shape)
    img = img.transpose(2, 0, 1) # 채널 높이 폭
    img = torch.from_numpy(img).unsqueeze(0) # 1, 채널, 높이, 폭
    print(img.shape)
    img = img.to(device)
    scale = scale.to(device)


    loc, conf, landms = model(img)  # box_regression의 loc, softmax의 confidence, landmark_regression

    priorbox = PriorBox(cfg, image_size=(im_height, im_width))
    priors = priorbox.vectorized_forward()
    print(priors.shape)
    priors = priors.to(device)
    prior_data = priors.data
    boxes = decode(loc.data.squeeze(0), prior_data, cfg['variance'])
    boxes = boxes * scale / resize
    boxes = boxes.cpu().numpy()
    scores = conf.squeeze(0).data.cpu().numpy()[:, 1]

    # ignore low scores
    inds = np.where(scores > confidence_threshold)[0]
    boxes = boxes[inds]
    scores = scores[inds]

    # keep top-K before NMS
    order = scores.argsort()[::-1][:top_k]
    boxes = boxes[order]
    scores = scores[order]

    # do NMS
    dets = np.hstack((boxes, scores[:, np.newaxis])).astype(np.float32, copy=False)
    keep = py_cpu_nms(dets, nms_threshold)
    dets = dets[keep, :]

    # keep top-K faster NMS
    dets = dets[:keep_top_k, :]

    fps_ = round(1/(time.time() - tic), 2)
    for b in dets:
        if b[4] < vis_thres:
            continue
        b = list(map(int, b))
        cv2.rectangle(test_img, (b[0], b[1]), (b[2], b[3]), (0, 0, 255), 4)
    cv2.putText(test_img, "retinaface", (410,70),cv2.FONT_HERSHEY_DUPLEX, 1.5,(255,0,0), thickness=3, lineType=cv2.LINE_AA)
    cv2.putText(test_img, "fps : "+str(fps_), (5,70),cv2.FONT_HERSHEY_DUPLEX, 1.5,(0,0,255), thickness=3, lineType=cv2.LINE_AA)
    return test_img

test_path = "D:/DL_models/Pytorch_Retinaface/dataset/pr_me.jpg"
test_img = cv2.imread(test_path)
tic = time.time()
retina_img = retinaface_inf(test_img, model)
toc = time.time()

print("time :", toc - tic )
plt.imshow(cv2.cvtColor(test_img, cv2.COLOR_BGR2RGB))
plt.show()