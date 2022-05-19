# -*- coding: utf-8 -*-
"""
Created on Wed May 11 11:22:59 2022

@author: albu
"""

# -*- coding: utf-8 -*-
"""
Function to test the Baseline model

Input: car image

Output: clean vs dirty class | probability of dirty | heatmap where the model focus the decision
"""

#imports
from pytorch_grad_cam import GradCAM, ScoreCAM, GradCAMPlusPlus, AblationCAM, XGradCAM, EigenCAM, FullGrad
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from pytorch_grad_cam.utils.image import show_cam_on_image

import torch
from torchvision import transforms
import torch.nn.functional as F

import numpy as np
from PIL import Image
from matplotlib.pyplot import imshow
import matplotlib.pyplot as plt
import cv2
import os

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


#Function to test the model
def test_classifier_maps(idx,image_array ,model_path='baseline.pth', verbose='True',path_save_heatmaps='heatmaps'):
    
    im_rgb= cv2.cvtColor(image_array, cv2.COLOR_BGR2RGB)
    im = Image.fromarray(im_rgb)
    
    newsize = (224, 224)
    im = im.resize(newsize)
    
    #To plot
    pic = np.asarray(im)
    pic = pic.astype('float32')
    pic /= 255.0

    #transformations and normalization 
    test_transforms = transforms.Compose([
        
        transforms.ToTensor(), 
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
    ])
    
    X = test_transforms(im)
    
    #load model
    model = torch.load(model_path)
    model.eval()
    
    #get layers name where we want to get the activation maps
    named_layers = dict(model.named_modules())
    
    #uncomment to get the layer name and position
    # print(named_layers)
    
    target_layers = [model.model[0][0][18][0]]

    
    #Apply GRADCAM
    input_tensor = torch.unsqueeze(X, 0).to(device)  # Create an input tensor image for your model..

    # Construct the CAM object once, and then re-use it on many images:
    cam = GradCAM(model=model, target_layers=target_layers, use_cuda=device)

    #predict class
    phat = F.softmax(model(torch.unsqueeze(X, 0).to(device)),1)
    
    y_pred = phat.argmax(1)
    
    y_pred = y_pred.cpu().numpy()
    phat = phat.detach().cpu().numpy()[0][1]

    if y_pred==0:
        y_pred_='clean'
        target_=0
    else:
        y_pred_='dirty'
        target_=1

    targets = [ClassifierOutputTarget(target_)]

    # You can also pass aug_smooth=True and eigen_smooth=True, to apply smoothing.
    grayscale_cam = cam(input_tensor=input_tensor, targets=targets)

    # In this example grayscale_cam has only one image in the batch:
    grayscale_cam = grayscale_cam[0, :]
    visualization = show_cam_on_image(pic, grayscale_cam, use_rgb=True)

    title_={'pred_class': y_pred_, 'probability_dirty': phat}
    f = plt.figure()
    plt.axis('off')
    plt.title(title_)
    f.add_subplot(1,2, 1)
    plt.axis('off')
    plt.imshow(pic)
    f.add_subplot(1,2, 2)
    plt.axis('off')
    plt.imshow(np.asarray(visualization))
    # cbar = plt.colorbar(orientation='horizontal')
    plt.show(block=True)
    
    
    isExist = os.path.exists(path_save_heatmaps)

    if not isExist:
      
      # Create a new directory because it does not exist 
      os.makedirs(path_save_heatmaps)
      
      print("The new directory is created!")
    if verbose:
        f.savefig(os.path.join(path_save_heatmaps,'%s_heatmap.png'%(idx)))
    
        
    return {'pred_class': y_pred_, 'probability_dirty': phat, 'activation map':np.asarray(visualization)}


def videotoheatmap(video_file='taxi_video.mp4', number_of_frames=50, save_heatmaps='True'):
    vidcap = cv2.VideoCapture(video_file)
    success,image = vidcap.read()
    count = 0
    images=[]
    
    while success:
      images.append(image)
      success,image = vidcap.read()
      count += 1
      
    import glob
    for idx, image_arr in enumerate(images):
        if idx<number_of_frames:
            predictions=test_classifier_maps(idx,image_arr, verbose=save_heatmaps)
        else:
            pass
        
        

videotoheatmap()