#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Aug 17 22:10:01 2018

@author: Caroline Pacheco do E. Silva
"""

import os
import cv2
from xml.dom.minidom import parseString
from lxml.etree import Element, SubElement, tostring
import numpy as np
from os.path import join

## yolo classes
YOLO_CLASSES = ('Model 0', 'Model 1', 'Model 2', 'Model 3', 'Model 4', 'Model 5',
                'Model 6', 'Model 7', 'Model 8', 'Model 9', 'Model 10', 'Model 11',
                'Model 12', 'Model 13', 'Model 14', 'Model 15', 'Model 16', 'Model 17',
                'Model 18', 'Model 19', 'Model 20', 'Model 21', 'Model 22', 'Model 23',
                'Model 24', 'Model 25', 'Model 26', 'Model 27', 'Model 28', 'Model 29', 'Model 30'
                )


## converts the normalized positions  into integer positions
def unconvert(class_id, width, height, x, y, w, h):

    xmax = int((x*width) + (w * width)/2.0)
    xmin = int((x*width) - (w * width)/2.0)
    ymax = int((y*height) + (h * height)/2.0)
    ymin = int((y*height) - (h * height)/2.0)
    class_id = int(class_id)
    return (class_id, xmin, xmax, ymin, ymax)


## path root folder
ROOT = '/scratch/project_2005695/Transfer-Learning-Library/examples/domain_adaptation/object_detection/datasets/TLESS_real_dataset'


## converts coco into xml 
def xml_transform(root, classes):  
    class_path  = join(root, 'YoloAnnotations')
    ids = list()
    l=os.listdir(class_path)
    
    check = '.DS_Store' in l
    if check == True:
        l.remove('.DS_Store')
        
    ids=[x.split('.')[0] for x in l]   

    annopath = join(root, 'YoloAnnotations', '%s.txt')
    imgpath = join(root, 'JPEGImages', '%s.png')
    
    os.makedirs(join(root, 'Annotations'), exist_ok=True)
    outpath = join(root, 'Annotations', '%s.xml')

    for i in range(len(ids)):
        img_id = ids[i] 
        img= cv2.imread(imgpath % img_id)
        height, width, channels = img.shape # pega tamanhos e canais das images

        node_root = Element('annotation')
        node_folder = SubElement(node_root, 'folder')
        node_folder.text = 'TLESS_real_dataset'
        img_name = img_id + '.png'
    
        node_filename = SubElement(node_root, 'filename')
        node_filename.text = img_name
        
        node_source= SubElement(node_root, 'source')
        node_database = SubElement(node_source, 'database')
        node_database.text = 'None'
        
        node_size = SubElement(node_root, 'size')
        node_width = SubElement(node_size, 'width')
        node_width.text = str(width)
    
        node_height = SubElement(node_size, 'height')
        node_height.text = str(height)

        node_depth = SubElement(node_size, 'depth')
        node_depth.text = str(channels)

        node_segmented = SubElement(node_root, 'segmented')
        node_segmented.text = '0'

        target = (annopath % img_id)
        if os.path.exists(target):
            label_norm= np.loadtxt(target).reshape(-1, 5)

            for i in range(len(label_norm)):
                labels_conv = label_norm[i]
                new_label = unconvert(labels_conv[0], width, height, labels_conv[1], labels_conv[2], labels_conv[3], labels_conv[4])
                node_object = SubElement(node_root, 'object')
                node_name = SubElement(node_object, 'name')
                node_name.text = classes[new_label[0]]
                
                node_pose = SubElement(node_object, 'pose')
                node_pose.text = 'Unspecified'
                
                
                node_truncated = SubElement(node_object, 'truncated')
                node_truncated.text = '0'
                node_difficult = SubElement(node_object, 'difficult')
                node_difficult.text = '0'
                node_bndbox = SubElement(node_object, 'bndbox')
                node_xmin = SubElement(node_bndbox, 'xmin')
                node_xmin.text = str(new_label[1])
                node_ymin = SubElement(node_bndbox, 'ymin')
                node_ymin.text = str(new_label[3])
                node_xmax = SubElement(node_bndbox, 'xmax')
                node_xmax.text =  str(new_label[2])
                node_ymax = SubElement(node_bndbox, 'ymax')
                node_ymax.text = str(new_label[4])
                xml = tostring(node_root, pretty_print=True)  
                dom = parseString(xml)
        print(target)  
        f =  open(outpath % img_id, "wb")
        #f = open(os.path.join(outpath, img_id), "w")
        #os.remove(target)
        f.write(xml)
        f.close()     
       

xml_transform(ROOT, YOLO_CLASSES)
print("Done")