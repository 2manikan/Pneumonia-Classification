# -*- coding: utf-8 -*-
"""
Created on Thu May  1 22:37:59 2025

@author: manid
"""

from PIL import Image
import os
#import matplotlib.pyplot as plt
import torch
import torchvision.transforms as trf
import torchvision
from cnn_detection import Model
from cnn_detection import CustomDataset



#load model
model_state_dict=torch.load("C:/Users/manid/OneDrive/Documents/sd.pt")
model=Model()
model.load_state_dict(model_state_dict)
model.eval()

#create testing dataset
root_folder="C:/Users/manid/OneDrive/Documents/dataset/chest_xray_low_res/test/NORMAL/"
root_folder2="C:/Users/manid/OneDrive/Documents/dataset/chest_xray_low_res/test/PNEUMONIA/"
transform=trf.Compose([trf.Resize(128), trf.CenterCrop(128), trf.ToTensor()])
ds=CustomDataset(root_folder,root_folder2, transform)



#iterate over testing dataset
correct=0
total=4500
for i in range(total):
    x, target=ds.get_item()
    f=model(x)
    
    if(torch.all(target==f.round()).item()):
        correct+=1
print("Accuracy: ", correct/total)


