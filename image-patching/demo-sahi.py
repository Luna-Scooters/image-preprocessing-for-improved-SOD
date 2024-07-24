import math
import os
from PIL import Image, ImageDraw
import cv2
from ultralytics import YOLO
import numpy as np
from collections import defaultdict
from collections import Counter
import torch
import csv 
import yaml
import argparse
import tqdm
import time 
import sahi 
from sahi import AutoDetectionModel
from sahi.utils.cv import read_image
from sahi.utils.file import download_from_url
from sahi.predict import get_prediction, get_sliced_prediction, predict
from torchmetrics.detection import MeanAveragePrecision
from pprint import pprint


# yolov8_model_path = "/Users/chinya07/Downloads/best (7).pt"
yolov8_model_path = "/Users/chinya07/Downloads/nuImages_solo_320_nano.pt"

detection_model = AutoDetectionModel.from_pretrained(
    model_type='yolov8',
    model_path=yolov8_model_path,
    confidence_threshold=0.3,
    device="cpu", # or 'cuda:0'
)
image_path = "/Users/chinya07/Downloads/n003-2018-01-02-11-48-43+0800__CAM_FRONT__1514865148310836.jpg"
org_image = Image.open(image_path)
frame = org_image.resize((640,640))

result = get_sliced_prediction(
                frame,
                detection_model,
                slice_height=320,
                slice_width=320,
                overlap_height_ratio=0,
                overlap_width_ratio=0,
                postprocess_match_threshold=0.5,
                
            ).object_prediction_list

# print(result.object_prediction_list)

# result.export_visuals(export_dir="/Users/chinya07/Desktop/PROJECTS/PHD/image-preprocessing-for-improved-SOD/sahi/demo-data/")

# im = Image.open("/Users/chinya07/Desktop/PROJECTS/PHD/image-preprocessing-for-improved-SOD/sahi/demo-data/prediction_visual.png")     

# im.show()

# detections = [
#                 [dict(boxes=torch.tensor(i.bbox.to_xyxy()), scores=torch.tensor(i.score.value), labels=torch.tensor(i.category.id))] for i in result
#             ]
            
# print(detections)

preds = [
                dict(boxes=torch.tensor([i.bbox.to_xyxy() for i in result]), scores=torch.tensor([i.score.value for i in result]), labels=torch.tensor([i.category.id for i in result]))
            ]

# print(preds)

target = [
                dict(boxes=torch.tensor([i.bbox.to_xyxy() for i in result]), labels=torch.tensor([i.category.id for i in result]))
            ]


metric = MeanAveragePrecision(iou_type="bbox", backend='pycocotools')
metric.update(preds, target)
pprint(metric.compute())