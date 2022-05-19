# Imports
import os
import _pickle as pickle
import numpy as np
from PIL import Image

# PyTorch Imports
import torch
import torchvision

# Fix Random Seeds
random_seed = 42
torch.manual_seed(random_seed)
np.random.seed(random_seed)


# Project Imports
from car_model_classifier.code.model_utilities import VGG16, DenseNet121, ResNet50



# Function: Predict Model
def predict_car_model(image, img_nr_channels=3, img_height=224, img_width=224, backbone="ResNet50", nr_classes=196, model_checkpoint=True, device='cpu'):

    # VGG-16
    if backbone.lower() == "VGG16".lower():
        model = VGG16(channels=img_nr_channels, height=img_height, width=img_width, nr_classes=nr_classes)

    # DenseNet-121
    elif backbone.lower() == "DenseNet121".lower():
        model = DenseNet121(channels=img_nr_channels, height=img_height, width=img_width, nr_classes=nr_classes)

    # ResNet50
    elif backbone.lower() == "ResNet50".lower():
        model = ResNet50(channels=img_nr_channels, height=img_height, width=img_width, nr_classes=nr_classes)
    
    # Move model to device
    model = model.to(device)

    # Load model weights
    if model_checkpoint:    
        model_file = os.path.join("results", f"{backbone.lower()}_stanfordcars_best.pt")
        checkpoint = torch.load(model_file, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'], strict=True)

    # Put model in evaluation mode
    model.eval()


    # Mean and STD to Normalize the inputs into pretrained models
    MEAN = [0.485, 0.456, 0.406]
    STD = [0.229, 0.224, 0.225]

    # Transforms
    transforms = torchvision.transforms.Compose([
        torchvision.transforms.Resize((224, 224)),
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize(mean=MEAN, std=STD)
    ])


    with torch.no_grad():
        # Load data
        # pil_image = Image.open(image).convert("RGB")
        pil_image = transforms(image)
        pil_image.to(device)

        # Get logits
        logits = model(pil_image)

        # Apply Softmax to Logits
        s_logits = torch.nn.Softmax(dim=1)(logits)                        
        s_logits = torch.argmax(s_logits, dim=1)
        
        # Get prediction
        prediction = s_logits[0].item()

        # Map prediction into class name
        with open('idx_to_class_name.pickle', 'rb') as f:
            idx_to_class_name = pickle.load(f)
        
        
        # Get predicted class
        predicted_class = idx_to_class_name[int(prediction)]


    return predicted_class
