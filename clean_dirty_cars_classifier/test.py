"""
Function to test the Baseline model

Input: car image

Output: clean vs dirty
"""


import torch
from PIL import Image
from torchvision import transforms
import torch.nn.functional as F
import numpy as np

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


#Function to test the model
def test_classifier(image_name="3efcdc3507.jpg",model_path='baseline.pth'):
    
    im = Image.open(image_name)
    
    newsize = (224, 224)
    im = im.resize(newsize)
    
    test_transforms = transforms.Compose([
        
        transforms.ToTensor(), 
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
    ])
    
    X = test_transforms(im)
    
    model = torch.load(model_path)
    phat = F.softmax(model(torch.unsqueeze(X, 0).to(device)),1)
    
    y_pred = phat.argmax(1)
    
    y_pred = y_pred.cpu().numpy()
    phat = phat.detach().cpu().numpy()[0][1]

    if y_pred==0:
        y_pred_='clean'
    else:
        y_pred_='dirty'
        
    return {'pred_class': y_pred_, 'probability_dirty': phat}
    

predictions=test_classifier()