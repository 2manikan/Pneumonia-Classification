# -*- coding: utf-8 -*-
"""
Created on Mon Apr 28 20:25:14 2025

@author: manid
"""

from PIL import Image
import os
import matplotlib.pyplot as plt
import torch
import torchvision.transforms as trf
import torchvision


class Model(torch.nn.Module):
    def __init__(self):
        super(Model,self).__init__()
        #model creation
        self.model=torch.nn.Sequential(
        
        torch.nn.BatchNorm2d(num_features=1),
        
        torch.nn.Conv2d(1, 16, 3, padding=1), 
        #padding: adding zeros around the image. can add 1 layer of zeroes, 2, etc.
        #added one layer to preserve shape
        
        torch.nn.MaxPool2d(kernel_size=2,stride=2), #16x64x64
        
        torch.nn.BatchNorm2d(num_features=16),
        
        torch.nn.Conv2d(16, 32, 3, padding=1), 
        torch.nn.MaxPool2d(kernel_size=2,stride=2), #32x32x32
        
        torch.nn.BatchNorm2d(num_features=32),
        
        torch.nn.Conv2d(32, 64, 3, padding=1),
        torch.nn.MaxPool2d(kernel_size=2,stride=2), #64x16x16
        
        torch.nn.BatchNorm2d(num_features=64),
        
        torch.nn.Conv2d(64, 128, 3, padding=1),
        torch.nn.MaxPool2d(kernel_size=2,stride=2), #128x8x8
        
        torch.nn.BatchNorm2d(num_features=128),
        
        torch.nn.Conv2d(128, 256, 3, padding=1),
        torch.nn.MaxPool2d(kernel_size=2,stride=2), #256x4x4
        
        torch.nn.BatchNorm2d(num_features=256),
        
        torch.nn.Conv2d(256, 512, 3, padding=1),
        torch.nn.MaxPool2d(kernel_size=2,stride=2), #512x2x2
        
        torch.nn.BatchNorm2d(num_features=512),
        
        torch.nn.Conv2d(512, 1024, 3, padding=1),
        torch.nn.MaxPool2d(kernel_size=2,stride=2), #1024x1x1
        
        
        
        torch.nn.Flatten(), 
        #parameters for flatten are start_dim and end_dim to flatten. these are 1 and -1 respectively, meaning all dimensions are flattened except batch dimension
        
        torch.nn.Linear(1024*1*1, 256), #fully connected layer
        torch.nn.Linear(256,2), #fully connected layer
        torch.nn.Sigmoid()
        )
        
        
    def forward(self, x):
        return self.model(x)

#create dataset
class CustomDataset(torch.utils.data.Dataset):
    def __init__(self, root_folder, root_folder2, transform=None):
        super(CustomDataset,self).__init__()
        self.root_folder=root_folder
        self.root_folder2=root_folder2
        self.transform=transform
        self.index=0
        self.index2=0
        self.labels=os.listdir(self.root_folder)
        self.labels2=os.listdir(self.root_folder2)
    def __len__(self):
        return len(self.labels) + len(self.labels2)
    def get_item(self): #__getitem__ alters indexing capability but we're not using now
        if self.index==self.index2 and self.index>=len(self.labels): #reset indices to beginning
            self.index=0
            self.index2=0
        
        if self.index==self.index2:
           x=Image.open(self.root_folder+self.labels[self.index]).convert('L') #torch.load doesn't work with image format
           self.index+=1
           return (self.transform(x).unsqueeze(dim=0), torch.tensor([[1.,0.]]).float()) #[1,0] means is normal, not pneumonia
        else:
           x=Image.open(self.root_folder2+self.labels2[self.index2]).convert('L') #torch.load doesn't work with image format
           self.index2+=1
           return (self.transform(x).unsqueeze(dim=0), torch.tensor([[0.,1.]]).float())
        
#create dataset object
root_folder="C:/Users/manid/OneDrive/Documents/dataset/chest_xray_low_res/train/NORMAL/"
root_folder2="C:/Users/manid/OneDrive/Documents/dataset/chest_xray_low_res/train/PNEUMONIA/"
transform=trf.Compose([trf.Resize(128), trf.CenterCrop(128), trf.ToTensor()])
ds=CustomDataset(root_folder,root_folder2, transform)

#create model object
model=Model()

#optimizer creation
optimizer=torch.optim.Adam(model.parameters(), lr=1e-5)

losses=[]
#TRAINING
#number of training steps
for i in range(1000):
    x,label=ds.get_item()
    x=x.float()
    
    prediction=model(x)
    loss=torch.nn.functional.binary_cross_entropy(prediction, label)#binary crossentropy is used for binary classification problems
    losses.append(loss.item())
    
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    

plt.plot(losses)
plt.show()


torch.save(model.state_dict(), "C:/Users/manid/OneDrive/Documents/sd.pt")




