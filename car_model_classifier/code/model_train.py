# Imports
import os
import argparse
import numpy as np
from tqdm import tqdm
import datetime
from torchinfo import summary

# Sklearn Imports
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight

# PyTorch Imports
import torch
from torch.utils.data import DataLoader
import torchvision
from torch.utils.tensorboard import SummaryWriter


# Fix Random Seeds
random_seed = 42
torch.manual_seed(random_seed)
np.random.seed(random_seed)


# Project Imports
from model_utilities import VGG16, DenseNet121, ResNet50
from data_utilities import StanfordCarsDataset



# Add the arguments
# Data directory
DATA_DIR = "data"

# Model
MODEL = "ResNet50"

# Batch size
BATCH_SIZE = 32

# Image size
IMG_SIZE = 224

# Number of epochs
EPOCHS = 300

# Learning rate
LEARNING_RATE = 1e-3

# Output directory
RESULTS_DIR = "results"

# Number of workers
NUM_WORKERS = 4

# GPU ID
GPU_ID = 0



# Timestamp (to save results)
timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
outdir = os.path.join(RESULTS_DIR, "stanfordcars", MODEL.lower(), timestamp)
if not os.path.isdir(outdir):
    os.makedirs(outdir)



# Mean and STD to Normalize the inputs into pretrained models
MEAN = [0.485, 0.456, 0.406]
STD = [0.229, 0.224, 0.225]

# Data Transforms
# Train Transforms
train_transforms = torchvision.transforms.Compose([
    torchvision.transforms.Resize((IMG_SIZE, IMG_SIZE)),
    torchvision.transforms.RandomAffine(degrees=(-10, 10), translate=(0.05, 0.1), scale=(0.95, 1.05), shear=0, resample=0, fillcolor=(0, 0, 0)),
    torchvision.transforms.RandomHorizontalFlip(p=0.5),
    torchvision.transforms.ToTensor(),
    torchvision.transforms.Normalize(mean=MEAN, std=STD)
])

# Validation Transforms
val_transforms = torchvision.transforms.Compose([
    torchvision.transforms.Resize((IMG_SIZE, IMG_SIZE)),
    torchvision.transforms.ToTensor(),
    torchvision.transforms.Normalize(mean=MEAN, std=STD)
])


# Datasets
# Train Set
train_set = StanfordCarsDataset(base_data_path=DATA_DIR, data_split="train", resized=True, transform=train_transforms)

# Validation Set
val_set = StanfordCarsDataset(base_data_path=DATA_DIR, data_split="test", resized=True, transform=val_transforms)

# Get number of classes
nr_classes = len(train_set.class_names)
# print(nr_classes)


# Results and Weights
weights_dir = os.path.join(outdir, "weights")
if not os.path.isdir(weights_dir):
    os.makedirs(weights_dir)


# History Files
history_dir = os.path.join(outdir, "history")
if not os.path.isdir(history_dir):
    os.makedirs(history_dir)


# Tensorboard
tbwritter = SummaryWriter(log_dir=os.path.join(outdir, "tensorboard"), flush_secs=30)


# Choose GPU
DEVICE = f"cuda:{GPU_ID}" if torch.cuda.is_available() else "cpu"
print(f"Using device: {DEVICE}")



# Input Data Dimensions
img_nr_channels = 3
img_height = IMG_SIZE
img_width = IMG_SIZE



# VGG-16
if MODEL.lower() == "VGG16".lower():
    model = VGG16(channels=img_nr_channels, height=img_height, width=img_width, nr_classes=nr_classes)

# DenseNet-121
elif MODEL.lower() == "DenseNet121".lower():
    model = DenseNet121(channels=img_nr_channels, height=img_height, width=img_width, nr_classes=nr_classes)

# ResNet50
elif MODEL.lower() == "ResNet50".lower():
    model = ResNet50(channels=img_nr_channels, height=img_height, width=img_width, nr_classes=nr_classes)



# Put model into DEVICE (CPU or GPU)
model = model.to(DEVICE)


# Get model summary
try:
    model_summary = summary(model, (1, 3, IMG_SIZE, IMG_SIZE), device=DEVICE)

except:
    model_summary = str(model)


# Write into file
with open(os.path.join(outdir, "model_summary.txt"), 'w') as f:
    f.write(str(model_summary))


# Hyper-parameters
LOSS = torch.nn.CrossEntropyLoss(reduction="sum")
VAL_LOSS = torch.nn.CrossEntropyLoss(reduction="sum")
OPTIMISER = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)


# Dataloaders
train_loader = DataLoader(dataset=train_set, batch_size=BATCH_SIZE, shuffle=True, pin_memory=False, num_workers=NUM_WORKERS)
val_loader = DataLoader(dataset=val_set, batch_size=BATCH_SIZE, shuffle=True, pin_memory=False, num_workers=NUM_WORKERS)



# Train model and save best weights on validation set
# Initialise min_train and min_val loss trackers
min_train_loss = np.inf
min_val_loss = np.inf

# Initialise losses arrays
train_losses = np.zeros((EPOCHS, ))
val_losses = np.zeros_like(train_losses)

# Initialise metrics arrays
train_metrics = np.zeros((EPOCHS, 5))
val_metrics = np.zeros_like(train_metrics)

# Go through the number of Epochs
for epoch in range(0, EPOCHS):
    # Epoch 
    print(f"Epoch: {epoch+1}")
    
    # Training Loop
    print("Training Phase")
    
    # Initialise lists to compute scores
    y_train_true = np.empty((0), int)
    y_train_pred = torch.empty(0, dtype=torch.int32, device=DEVICE)
    y_train_scores = torch.empty(0, dtype=torch.float, device=DEVICE) # save scores after softmax for roc auc


    # Running train loss
    run_train_loss = torch.tensor(0, dtype=torch.float64, device=DEVICE)


    # Put model in training mode
    model.train()

    # Iterate through dataloader
    for images, labels in tqdm(train_loader):
        # Concatenate lists
        y_train_true = np.append(y_train_true, labels.numpy(), axis=0)

        # Move data and model to GPU (or not)
        images, labels = images.to(DEVICE, non_blocking=True), labels.to(DEVICE, non_blocking=True)

        # Find the loss and update the model parameters accordingly
        # Clear the gradients of all optimized variables
        OPTIMISER.zero_grad(set_to_none=True)


        # Forward pass: compute predicted outputs by passing inputs to the model
        logits = model(images)

        
        # Compute the batch loss
        # Using CrossEntropy w/ Softmax
        loss = LOSS(logits, labels)

        # Update batch losses
        run_train_loss += loss

        # Backward pass: compute gradient of the loss with respect to model parameters
        loss.backward()
        
        # Perform a single optimization step (parameter update)
        OPTIMISER.step()

        # Using Softmax
        # Apply Softmax on Logits and get the argmax to get the predicted labels
        s_logits = torch.nn.Softmax(dim=1)(logits)
        y_train_scores = torch.cat((y_train_scores, s_logits))
        s_logits = torch.argmax(s_logits, dim=1)
        y_train_pred = torch.cat((y_train_pred, s_logits))


    # Compute Average Train Loss
    avg_train_loss = run_train_loss/len(train_loader.dataset)
    

    # Compute Train Metrics
    y_train_pred = y_train_pred.cpu().detach().numpy()
    y_train_scores = y_train_scores.cpu().detach().numpy()
    train_acc = accuracy_score(y_true=y_train_true, y_pred=y_train_pred)
    train_recall = recall_score(y_true=y_train_true, y_pred=y_train_pred, average='micro')
    train_precision = precision_score(y_true=y_train_true, y_pred=y_train_pred, average='micro')
    train_f1 = f1_score(y_true=y_train_true, y_pred=y_train_pred, average='micro')
    train_auc = roc_auc_score(y_true=y_train_true, y_score=y_train_scores[:, 1], average='micro')

    # Print Statistics
    print(f"Train Loss: {avg_train_loss}\tTrain Accuracy: {train_acc}")
    # print(f"Train Loss: {avg_train_loss}\tTrain Accuracy: {train_acc}\tTrain Recall: {train_recall}\tTrain Precision: {train_precision}\tTrain F1-Score: {train_f1}")


    # Append values to the arrays
    # Train Loss
    train_losses[epoch] = avg_train_loss
    # Save it to directory
    fname = os.path.join(history_dir, f"{MODEL.lower()}_tr_losses.npy")
    np.save(file=fname, arr=train_losses, allow_pickle=True)


    # Train Metrics
    # Acc
    train_metrics[epoch, 0] = train_acc
    # Recall
    train_metrics[epoch, 1] = train_recall
    # Precision
    train_metrics[epoch, 2] = train_precision
    # F1-Score
    train_metrics[epoch, 3] = train_f1
    # ROC AUC
    train_metrics[epoch, 4] = train_auc

    # Save it to directory
    fname = os.path.join(history_dir, f"{MODEL.lower()}_tr_metrics.npy")
    np.save(file=fname, arr=train_metrics, allow_pickle=True)

    # Plot to Tensorboard
    tbwritter.add_scalar("loss/train", avg_train_loss, global_step=epoch)
    tbwritter.add_scalar("acc/train", train_acc, global_step=epoch)
    tbwritter.add_scalar("rec/train", train_recall, global_step=epoch)
    tbwritter.add_scalar("prec/train", train_precision, global_step=epoch)
    tbwritter.add_scalar("f1/train", train_f1, global_step=epoch)
    tbwritter.add_scalar("auc/train", train_auc, global_step=epoch)

    # Update Variables
    # Min Training Loss
    if avg_train_loss < min_train_loss:
        print(f"Train loss decreased from {min_train_loss} to {avg_train_loss}.")
        min_train_loss = avg_train_loss


    # Validation Loop
    print("Validation Phase")


    # Initialise lists to compute scores
    y_val_true = np.empty((0), int)
    y_val_pred = torch.empty(0, dtype=torch.int32, device=DEVICE)
    y_val_scores = torch.empty(0, dtype=torch.float, device=DEVICE) # save scores after softmax for roc auc

    # Running train loss
    run_val_loss = 0.0

    # Put model in evaluation mode
    model.eval()

    # Deactivate gradients
    with torch.no_grad():

        # Iterate through dataloader
        for images, labels in tqdm(val_loader):
            y_val_true = np.append(y_val_true, labels.numpy(), axis=0)

            # Move data data anda model to GPU (or not)
            images, labels = images.to(DEVICE, non_blocking=True), labels.to(DEVICE, non_blocking=True)

            # Forward pass: compute predicted outputs by passing inputs to the model
            logits = model(images)
            
            # Compute the batch loss
            # Using CrossEntropy w/ Softmax
            loss = VAL_LOSS(logits, labels)
            
            # Update batch losses
            run_val_loss += loss


            # Using Softmax Activation
            # Apply Softmax on Logits and get the argmax to get the predicted labels
            s_logits = torch.nn.Softmax(dim=1)(logits)                        
            y_val_scores = torch.cat((y_val_scores, s_logits))
            s_logits = torch.argmax(s_logits, dim=1)
            y_val_pred = torch.cat((y_val_pred, s_logits))

        

        # Compute Average Validation Loss
        avg_val_loss = run_val_loss/len(val_loader.dataset)

        # Compute Validation Accuracy
        y_val_pred = y_val_pred.cpu().detach().numpy()
        y_val_scores = y_val_scores.cpu().detach().numpy()
        val_acc = accuracy_score(y_true=y_val_true, y_pred=y_val_pred)
        val_recall = recall_score(y_true=y_val_true, y_pred=y_val_pred, average='micro')
        val_precision = precision_score(y_true=y_val_true, y_pred=y_val_pred, average='micro')
        val_f1 = f1_score(y_true=y_val_true, y_pred=y_val_pred, average='micro')
        val_auc = roc_auc_score(y_true=y_val_true, y_score=y_val_scores[:, 1], average='micro')

        # Print Statistics
        print(f"Validation Loss: {avg_val_loss}\tValidation Accuracy: {val_acc}")
        # print(f"Validation Loss: {avg_val_loss}\tValidation Accuracy: {val_acc}\tValidation Recall: {val_recall}\tValidation Precision: {val_precision}\tValidation F1-Score: {val_f1}")

        # Append values to the arrays
        # Validation Loss
        val_losses[epoch] = avg_val_loss
        # Save it to directory
        fname = os.path.join(history_dir, f"{MODEL.lower()}_val_losses.npy")
        np.save(file=fname, arr=val_losses, allow_pickle=True)


        # Train Metrics
        # Acc
        val_metrics[epoch, 0] = val_acc
        # Recall
        val_metrics[epoch, 1] = val_recall
        # Precision
        val_metrics[epoch, 2] = val_precision
        # F1-Score
        val_metrics[epoch, 3] = val_f1
        # ROC AUC
        val_metrics[epoch, 4] = val_auc

        # Save it to directory
        fname = os.path.join(history_dir, f"{MODEL.lower()}_val_metrics.npy")
        np.save(file=fname, arr=val_metrics, allow_pickle=True)

        # Plot to Tensorboard
        tbwritter.add_scalar("loss/val", avg_val_loss, global_step=epoch)
        tbwritter.add_scalar("acc/val", val_acc, global_step=epoch)
        tbwritter.add_scalar("rec/val", val_recall, global_step=epoch)
        tbwritter.add_scalar("prec/val", val_precision, global_step=epoch)
        tbwritter.add_scalar("f1/val", val_f1, global_step=epoch)
        tbwritter.add_scalar("auc/val", val_auc, global_step=epoch)

        # Update Variables
        # Min validation loss and save if validation loss decreases
        if avg_val_loss < min_val_loss:
            print(f"Validation loss decreased from {min_val_loss} to {avg_val_loss}.")
            min_val_loss = avg_val_loss

            print("Saving best model on validation...")

            # Save checkpoint
            model_path = os.path.join(weights_dir, f"{MODEL.lower()}_stanfordcars_best.pt")
            
            save_dict = {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': OPTIMISER.state_dict(),
                'loss': avg_train_loss,
            }
            torch.save(save_dict, model_path)

            print(f"Successfully saved at: {model_path}")
    


# Finish statement
print("Finished.")
