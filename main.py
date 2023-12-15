#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Aug  9 04:38:18 2020

@author: mhemsley
"""

from __future__ import print_function 
from __future__ import division
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import torchvision
import matplotlib.pyplot as plt
import os
from PIL import Image

from model import initialize_model
from train import train_model
from utils import data_load
from utils import create_optimizer
from utils import plot_roc
from utils import check_image
from utils import plot_training_curve

print("PyTorch Version: ",torch.__version__)
print("Torchvision Version: ",torchvision.__version__)


#This code is an example on how to train a deep nerual network for classification, as well as implement a
#basic transfer learning approch
#I'll comment with instructions and links to the textbook "Deep Learning with PyTorch" to explain concepts when relevant
# https://pytorch.org/assets/deep-learning/Deep-Learning-with-PyTorch.pdf
#I'd recommend reading the book sequentially, since the narrative is pretty good and easy to follow,
#but probably still useful to link to sections when I use them


######################################################################
# Setup
# --------------------------------
# 


##Path to data

data_dir = "/home/mhemsley/hymenoptera_data"
#data_dir = "/home/mhemsley/BRATS18_Slices_combined"

# Models to choose from [resnet, alexnet, vgg, squeezenet, densenet, inception]
# These are all examples of Convolutional Neural Networks, covered in chapter 8 of the text
# An more comprehensive overview of how CNNs work can be found here: https://cs231n.github.io/convolutional-networks
# this papge also specifically descrbes some of the different "flavours" of CNNs here https://cs231n.github.io/convolutional-networks/#case
# Note:Use VGG for best results with saliency map since it uses raw scores, no final softmax
model_name = "vgg"

# Number of classes in the dataset
num_classes = 2

# Batch size for training
# Batches are tensors which contain several samples. Larger the batch size typically leads to more staple convergence, but requires more memory.
batch_size = 16

# Number of epochs to train for. An "Epoch" is one iteration of the training process.
# Its better practice to define some stopping criteria and then train until that is reached, but hard defining the epoch number is fine for now 
num_epochs = 15

# Setup the loss function. Textbook Chapter 5.3 describes the basic idea of a loss function. 
criterion = nn.CrossEntropyLoss()

#Flag for pretained model. If true will train on top of imagenet parameters
#If pretrained is flase, feature_extract MUST BE FALSE

# "pretraining" is an example of transfer learning. Chapter 2 describes the idea behind, and other examples, of pretraining networks

pretrained=True

# Flag for feature extracting. When False, we finetune the whole model, 
# when True we only update the reshaped layer params
# Feature extracting is when you only update the final classification weights when pretraining, rather than the entire network. Also covered in chap 2
feature_extract = True

#if verbose is true, then print the model
#Turn this off after you've seen it once or twice, then its just clutter
verbose=False

# If you want to save the model, set to True
# It will be in a folder called models in the project dir
save_model=True
save_model_name='ft_whole'

if not os.path.exists('./models/'):
    os.makedirs('./models/')

# Initialize the model for this run
model_ft, input_size = initialize_model(model_name, num_classes, feature_extract, use_pretrained=pretrained)

if verbose==True:   
    # Print the model 
    print(model_ft)


#Dataloaders are covered in chapter 7.2.6
image_datasets,dataloaders_dict = data_load(input_size,data_dir,batch_size)

# Detect if we have a GPU available, otherwise use the cpu to train
# training on GPU is much faster see, for example, - https://azure.microsoft.com/en-us/blog/gpus-vs-cpus-for-deployment-of-deep-learning-models/
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Send the model to GPU
model_ft = model_ft.to(device)


#Optimizers are the lifeblood behind training a model. Covered in Chapter 5.5
optimizer_ft=create_optimizer(model_ft,feature_extract,verbose)

######################################################################
# Run Training and Validation 
# --------------------------------
# 


#Train the model. see train.py for more info
model_ft, hist = train_model(model_ft, dataloaders_dict, criterion, optimizer_ft, device, num_epochs=num_epochs, is_inception=(model_name=="inception"))

if save_model==True:
   torch.save(model_ft.state_dict(), './models/'+save_model_name) 

   ohist = [h.cpu().numpy().item() for h in hist]
   with open('./models/'+save_model_name+'hist', 'w') as file:
        for row in ohist:
            file.write(str(row)+'\n')

######################################################################
# Generate Figures
# --------------------------------
# 


#First, lets load a new instance of the model from the saved weights   
model,_ = initialize_model(model_name, num_classes, feature_extract, use_pretrained=pretrained)
model.load_state_dict(torch.load('./models/'+save_model_name))
model = model.to(device)

#Next we plot the ROC curve
plot_roc(model,dataloaders_dict)

#Now we can use the check_imge function to observe some individual examples
#Lets load this specific image from the validation folder
#img1=Image.open('/home/mhemsley/BRATS18_Slices_combined/val/tumor/Brats18_CBICA_AXJ_1_171_117.jpg')
img1=Image.open("/home/mhemsley/hymenoptera_data/val/bees/1032546534_06907fe3b3.jpg")

#now we run check image to get the prediction, plot the images, create the saliency map
check_image(img1,model)

#Plot the learning curve from the saved history file
history=[]
file_in = open('./models/'+save_model_name+'hist')
for line in file_in.readlines():
  history.append(float(line))
  
plot_training_curve(history)



