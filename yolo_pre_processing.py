# -*- coding: utf-8 -*-
"""
Created on Wed Apr  3 20:41:38 2019

@author: hemaxi
"""
import os
import cv2
from skimage.measure import regionprops, label
import numpy as np
from xmls_yolo import createXML, beginXML
from skimage.segmentation import clear_border
from lxml import etree

saveDir = r'D:\Ambiente_de_Trabalho\Tese\FUCCI\dataset\annotations_val'  #directory where you want to save the annotations (XML files)
masks_path = r'D:\Ambiente_de_Trabalho\Tese\FUCCI\dataset\masks_val' #directory with the binary segmentation masks
images_path = r'D:\Ambiente_de_Trabalho\Tese\FUCCI\dataset\color_jpg_val' #directory with the corresponding images of the binary masks

for img, msk in zip(os.listdir(images_path), os.listdir(masks_path)):
    mask = cv2.imread(os.path.join(masks_path, msk), cv2.IMREAD_GRAYSCALE)
    mask_labelled = label(mask)
    
    #remove nuclei at borders
    #mask_labelled = clear_border(mask_labelled, buffer_size=1)
    
    properties = regionprops(mask_labelled)

    #for each nuclei, save the corresponding features 
    annotation = beginXML(img, np.shape(mask)[1], np.shape(mask)[0] )

    
    for region in properties:
        bbox = region.bbox
        annotation = createXML(bbox[1], bbox[3], bbox[0], bbox[2], img, np.shape(mask)[1], 
                               np.shape(mask)[0], annotation)
    
    aux_img_name = img.replace('.jpg', '')
    save_path = os.path.join(saveDir, aux_img_name + ".xml")
    with open(save_path, 'wb') as file:
        aux = etree.tostring(annotation, pretty_print=True)
        aux.decode("utf-8")
        file.write(aux)
        
        
        
        
        
        
        