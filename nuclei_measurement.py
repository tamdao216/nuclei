#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May 21 21:32:17 2024

@author: mysns
"""
import detectron2
import cv2
import torch
import json
import os
import shutil
import yaml
import sys
import csv
from skimage.measure import regionprops, label
from PIL import Image
import matplotlib.pyplot as plt
print(sys.path)
sys.path.append("/home/mysns/detectron2")
from detectron2.utils.logger import setup_logger

from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg

cfg = get_cfg()
cfg.merge_from_file("config-10k_iter.yaml")
cfg.MODEL.WEIGHTS = "model_final.pth"
cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5  
cfg.MODEL.DEVICE = "cuda" if torch.cuda.is_available() else "cpu"  
predictor = DefaultPredictor(cfg)

input_folder = ""  
zoomed_folder = ""  
output_csv_path = ""

os.makedirs(output_folder, exist_ok=True)

output_dir_name = os.path.dirname(output_csv_path)

valid_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.gif']


x, y, w, h = 350, 350, 256, 256  

with open(output_csv_path, 'w', newline='') as csvfile:
    csvwriter = csv.writer(csvfile)

    
    csvwriter.writerow(["File Name", "Class Name", "Object Number", "Area", "Centroid", "BoundingBox"])

    
    for filename in os.listdir(input_folder):
        if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff')):
            
            image_path = os.path.join(input_folder, filename)
            
            image = cv2.imread(image_path)
            if image is None:
                print(f"Error reading image {image_path}")
                continue
            
            
            roi = image[y:y+h, x:x+w]
            
            
            zoomed_image = cv2.resize(roi, (image.shape[1], image.shape[0]), interpolation=cv2.INTER_LINEAR)
            
            zoomed_image_path = os.path.join(zoomed_folder, filename)
            cv2.imwrite(zoomed_image_path, zoomed_image)
            print(f"Zoomed image saved: {zoomed_image_path}")

        
        outputs = predictor(zoomed_image)

        
        mask = outputs["instances"].pred_masks.to("cpu").numpy().astype(bool)

        
        class_labels = outputs["instances"].pred_classes.to("cpu").numpy()

        
        labeled_mask = label(mask)
        props = regionprops(labeled_mask)

        
        for i, prop in enumerate(props):
            object_number = i + 1
            area = prop.area
            centroid = prop.centroid
            bounding_box = prop.bbox

            

            csvwriter.writerow([image_filename, class_name, object_number, area, centroid, bounding_box])

print("Object-level information saved to CSV file.")


