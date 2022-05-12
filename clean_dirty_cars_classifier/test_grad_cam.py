# -*- coding: utf-8 -*-
"""
Function to test the Baseline model

Input: car image

Output: clean vs dirty class | probability of dirty | heatmap where the model focus the decision
"""

#imports
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from pytorch_grad_cam.utils.image import show_cam_on_image

import torch
from torchvision import transforms
import torch.nn.functional as F
import json

import numpy as np
from PIL import Image
# from matplotlib.pyplot import imshow
# import matplotlib.pyplot as plt

device = torch.device('cpu')

#Function to test the model
def test_classifier_maps(im, model_path):
    
    # im = Image.open(image_name)
    
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
    model = torch.load(model_path, map_location = torch.device('cpu'))
    model.eval()
    
    #get layers name where we want to get the activation maps
    named_layers = dict(model.named_modules())
    
    #uncomment to get the layer name and position
    # print(named_layers)
    
    target_layers = [model.model[0][0][18][0]]

    
    #Apply GRADCAM
    input_tensor = torch.unsqueeze(X, 0).to(device)  # Create an input tensor image for your model..

    # Construct the CAM object once, and then re-use it on many images:
    cam = GradCAM(model=model, target_layers=target_layers, use_cuda=False)

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

    """
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
    cbar = plt.colorbar(orientation='horizontal')
    plt.show(block=True)
    """

    activations = json.dumps(np.asarray(visualization).tolist())

    # https://stackoverflow.com/questions/71595635/render-numpy-array-in-fastapi
    # TODO: Alternative ways of passing image (e.g. base64)
    # TODO: Save activations in an image (image filename = random hash)
    # Api returns image filename

    return {'pred_class': y_pred_, 'probability_dirty': float(phat) , 'activation map': activations}


# predictions=test_classifier_maps()
