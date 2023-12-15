#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Aug  9 04:30:02 2020

@author: mhemsley
"""
import time
import os
import torch
import torch.optim as optim
import numpy as np
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter
from sklearn.metrics import roc_curve
from sklearn.metrics import auc

def set_parameter_requires_grad(model, feature_extracting):
    if feature_extracting:
        for param in model.parameters():
            param.requires_grad = False
            
def create_optimizer(model,feature_extract,verbose):

    # Sometimes we want an optimizer which only updates some parameters
    # If feature_extract=True, then when we initiate the model we set the
    # paramerts requires_grad to False. So we manually pick which layers
    # we want to change (i.e change requires_grad=True) in the bit below
    # that list gets passed to the optimizer
    
    params_to_update = model.parameters()
    if verbose==True: print("Params to learn:")
    if feature_extract==True:
        params_to_update = []
        for name,param in model.named_parameters():
            if param.requires_grad == True:
                params_to_update.append(param)
                if verbose==True: print("\t",name)
    else:
        for name,param in model.named_parameters():
            if param.requires_grad == True:
                if verbose==True: print("\t",name)
    
    optimizer_ft = optim.SGD(params_to_update, lr=0.001, momentum=0.9)
    return optimizer_ft
            
            
            
def data_load(input_size,data_dir,batch_size):
    
    # Data augmentation and normalization for training
    # Just normalization for validation
    data_transforms = {
        'train': transforms.Compose([
            #transforms.RandomResizedCrop(input_size),
            transforms.Resize(input_size),
            transforms.CenterCrop(input_size),
            #transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
        'val': transforms.Compose([
            transforms.Resize(input_size),
            transforms.CenterCrop(input_size),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
    }
    
    print("Initializing Datasets and Dataloaders...")
    
    # Create training and validation datasets using ImageFolder
    image_datasets = {x: datasets.ImageFolder(os.path.join(data_dir, x), data_transforms[x]) for x in ['train', 'val']}
    # Create training and validation dataloaders using ImageFolder
    dataloaders_dict = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=batch_size, shuffle=True, num_workers=4) for x in ['train', 'val']}
    return image_datasets,dataloaders_dict
            
def check_image(image,model):
    
    preprocess = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(
    mean=[0.485, 0.456, 0.406],
    std=[0.229, 0.224, 0.225]
    )])    
    
    since = time.time()
    image_t=preprocess(image)
    batch_t=torch.unsqueeze(image_t,0)
    batch_t=batch_t.to('cuda')
    batch_t.requires_grad_()
    out=model(batch_t)
    
    # I should make a dict of labels to easily match output with labels
    _, preds = torch.max(out, 1)
    print(preds)
    
    #Create Saliency Map by backprop the class score and accumulating the grad
    #over the input data
    score_max_index = out.argmax()
    score_max = out[0,score_max_index]
    print(out)
    score_max.backward()
    saliency, _ = torch.max(batch_t.grad.data.abs(),dim=1)
    
    #Send to cpu to plot
    saliency=saliency.to('cpu')
    
    time_elapsed = time.time() - since
    
    #print(time_elapsed)
    print('Test complete in {:.0f}ms'.format(time_elapsed*1000))
    
    #Need titles and better organization for plots
    plt.subplot(1,3,1)
    plt.imshow(image_t[0,:,:],cmap='gray')
    plt.title('T1w')
    plt.axis('off')
    plt.subplot(1,3,2)
    plt.imshow(image_t[1,:,:],cmap='gray')
    plt.title('T2w')
    plt.axis('off')
    plt.subplot(1,3,3)
    plt.imshow(image_t[2,:,:],cmap='gray')
    plt.title('FLAIR')
    plt.axis('off')
    plt.suptitle('Test Images')
    plt.show()
    
    plt.subplot(1,2,1)
    plt.imshow(saliency[0], cmap=plt.cm.hot)
    plt.title('Raw Saliency Map')
    plt.axis('off')
    
    smoothed_map=gaussian_filter(saliency[0].numpy(), sigma=3)
    
    #thresh=0.15 #arbitrary threshold to make things look cleaner
    #smoothed_map_indices = smoothed_map < thresh
    #smoothed_map[smoothed_map_indices] = 0
   
    plt.subplot(1,2,2)
    plt.imshow(smoothed_map, cmap=plt.cm.hot)
    plt.title('Smoothed Saliency Map')
    plt.axis('off')
    plt.suptitle('Saliency Maps')
    plt.show()
    
    
    plt.subplot(1,3,1)
    plt.imshow(image_t[0,:,:], cmap='gray')
    plt.imshow(smoothed_map, cmap=plt.cm.hot, alpha=0.5)
    plt.title('T1w')
    plt.axis('off')
    plt.subplot(1,3,2)
    plt.imshow(image_t[1,:,:], cmap='gray')
    plt.imshow(smoothed_map, cmap=plt.cm.hot, alpha=0.5)
    plt.title('T2w')
    plt.axis('off')
    plt.subplot(1,3,3)
    plt.imshow(image_t[2,:,:], cmap='gray')
    plt.imshow(smoothed_map, cmap=plt.cm.hot, alpha=0.5)
    plt.title('FLAIR')
    plt.axis('off')
    plt.suptitle('Input Overlayed with Saliency')
    plt.show()
        
    
def test_class_probabilities(model, device, test_loader):
    model.eval()
    actuals = []
    probabilities = []
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            prediction = output.argmax(dim=1, keepdim=True)
            actuals.extend(target.view_as(prediction) == 1)
            probabilities.extend(np.exp(output[:, 1].cpu()))
    return [i.item() for i in actuals], [i.item() for i in probabilities]



def plot_roc(model, dataloaders_dict):
    
    actuals, class_probabilities=test_class_probabilities(model,'cuda:0',dataloaders_dict['val'])
    
    fpr, tpr, _ = roc_curve(actuals, class_probabilities)
    roc_auc = auc(fpr, tpr)
    plt.figure()
    lw = 2
    plt.plot(fpr, tpr, color='darkorange',
             lw=lw, label='ROC curve (area = %0.2f)' % roc_auc)
    plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC')
    plt.legend(loc="lower right")
    plt.show()

def plot_training_curve(history):
    plt.title("Validation Accuracy vs. Number of Training Epochs")
    plt.xlabel("Training Epochs")
    plt.ylabel("Validation Accuracy")
    plt.plot(range(1,len(history)+1),history,label="Training Accuracy")
    plt.xticks(np.arange(1, len(history)+1, 1.0))
    plt.ylim((0,1.))
    plt.legend()
    plt.show()
