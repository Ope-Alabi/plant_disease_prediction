import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import torch.nn.functional as F
import torchvision
from torchvision import datasets, models, transforms
from torchinfo import summary

# from IPython.display import Image
# from torchview import draw_graph

import streamlit as st

import time
import os
import copy
import json
from PIL import Image
import zipfile
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# hyper parameters
num_epochs = 35
batch_size = 32
learning_rate = 0.001
momentum = 0.9

# Define the transformation pipeline
mean = np.array([0.485, 0.456, 0.406])
std = np.array([0.229, 0.224, 0.225])

data_transforms = {
    'train': transforms.Compose([
    transforms.RandomHorizontalFlip(),   # Horizontal reflection
    transforms.RandomVerticalFlip(),     # Vertical reflection
    transforms.RandomRotation(30),       # Rotation
    transforms.RandomResizedCrop(224),   # Rescale images and crop to size
    transforms.RandomAffine(degrees=0, translate=(0.2, 0.2)), # Width and height shifts
    transforms.ColorJitter(brightness=(0.9, 1.1)),  # Brightness adjustment
    transforms.ToTensor(),               # Convert images to Tensor
    transforms.Normalize(mean, std)  # Normalize to [0, 1] range
]),
    'val': transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),              
    transforms.Normalize(mean, std)   
]),
    'test': transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),              
    transforms.Normalize(mean, std)  
]),
}



# Import Data
# data_dir = 'Soybean_ML_orig_20'
data_dir = '/Users/oalabi1/Desktop/PhD/Datasets/Soybean_ML_orig'
sets = ['train', 'val', 'test']

image_datasets = {x: datasets.ImageFolder(os.path.join(data_dir, x), data_transforms[x])
                  for x in sets}

dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=batch_size, shuffle=False, num_workers=0)
               for x in sets}

dataset_sizes = {x: len(image_datasets[x]) for x in sets}
class_names = image_datasets['train'].classes

num_classes = len(class_names)


# Setup pretrained model with ImageNet's pretrained weights
weights = torchvision.models.DenseNet201_Weights.DEFAULT
densenet_model = torchvision.models.densenet201(weights=weights).to(device)

print(summary(densenet_model, 
    input_size=(32, 3, 224, 224), 
    verbose=0,
    col_names=["input_size", "output_size", "num_params", "trainable"],
    col_width=20,
    row_settings=["var_names"]))

# Get the length of class_names (one output unit for each class)
output_shape = len(class_names)

# Modify the classifier
densenet_model.classifier = nn.Sequential(
    nn.Linear(in_features=densenet_model.classifier.in_features, out_features=128, bias=True), # in_features = 1920
    nn.ReLU(inplace=True),
    nn.Dropout(p=0.2),
    nn.Linear(in_features=128, out_features=64, bias=True),
    nn.ReLU(inplace=True),
    nn.Dropout(p=0.3),
    nn.Linear(in_features=64, out_features=output_shape, bias=True)  # Number of output classes
)

# # Redefine the forward pass
# class CustomDenseNet201(nn.Module):
#     def __init__(self, base_model):
#         super(CustomDenseNet201, self).__init__()
#         self.features = base_model.features
#         self.pool = nn.AdaptiveAvgPool2d((1, 1))
#         self.flatten = nn.Flatten()
#         self.classifier = base_model.classifier

#     def forward(self, x):
#         x = self.features(x)
#         x = self.pool(x)
#         x = self.flatten(x)
#         x = self.classifier(x)
#         return x

# Instantiate the custom model and move to device
model = densenet_model.to(device)


# Set the manual seeds
torch.manual_seed(42)
torch.cuda.manual_seed(42)


# Define loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, momentum=momentum)

n_total_steps = len(dataloaders['train'])


# print(summary(model, 
#             input_size=(32, 3, 224, 224), 
#             verbose=0,
#             col_names=["input_size", "output_size", "num_params", "trainable"],
#             col_width=20,
#             row_settings=["var_names"]))


print("Training started")

# Checkpoint directory
checkpoint_dir = 'checkpoints'
if not os.path.exists(checkpoint_dir):
    os.makedirs(checkpoint_dir)

best_val_loss = float('inf')

# START LOGGING INTO TEXT FILE

train_hist_dir = 'training_history'
if not os.path.exists(train_hist_dir):
    os.makedirs(train_hist_dir)

training_history_path = os.path.join(train_hist_dir, "train_hist.txt")

with open (training_history_path, 'w') as history:
    history.write("STARTING TO TRAIN \n")

# Training
for epoch in range(num_epochs):
    epoch_start_time = time.time()  # Start time for the epoch
    
    running_loss = 0.0
    n_correct_train = 0
    n_samples_train = 0
    model.train()  # Set model to training mode
    for i, (images, labels) in enumerate(dataloaders['train']):
        images = images.to(device)
        labels = labels.to(device)

        # Forward pass
        outputs = model(images)
        loss = criterion(outputs, labels)

        # Backward pass and optimization
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

        # Calculate training accuracy
        _, predictions = torch.max(outputs, 1)
        n_samples_train += labels.size(0)
        n_correct_train += (predictions == labels).sum().item()

        if (i + 1) % 300 == 0:
            print(f'Epoch [{epoch + 1}/{num_epochs}], Step [{i + 1}/{len(dataloaders["train"])}], Loss: {loss.item():.4f}')

    avg_train_loss = running_loss / len(dataloaders['train'])
    train_accuracy = 100.0 * n_correct_train / n_samples_train
    print(f'Epoch [{epoch + 1}/{num_epochs}], Train Loss: {avg_train_loss:.4f}, Train Accuracy: {train_accuracy:.2f}%')

    # Validation phase
    model.eval()  # Set model to evaluation mode
    val_running_loss = 0.0
    n_correct_val = 0
    n_samples_val = 0
    with torch.no_grad():
        for images, labels in dataloaders['val']:
            images = images.to(device)
            labels = labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            val_running_loss += loss.item()

            _, predictions = torch.max(outputs, 1)
            n_samples_val += labels.size(0)
            n_correct_val += (predictions == labels).sum().item()

    avg_val_loss = val_running_loss / len(dataloaders['val'])
    val_accuracy = 100.0 * n_correct_val / n_samples_val
    # print(f'Epoch [{epoch + 1}/{num_epochs}], Val Loss: {avg_val_loss:.4f}, Val Accuracy: {val_accuracy:.2f}%, Epoch Duration: {epoch_duration:.2f} seconds')

    # Checkpointing
    if avg_val_loss < best_val_loss:
        best_val_loss = avg_val_loss
        torch.save(model.state_dict(), os.path.join(checkpoint_dir, 'best_model.pth'))
        print(f'Saved Best Model with Val Loss: {avg_val_loss:.4f}')

    epoch_end_time = time.time()  # End time for the epoch
    epoch_duration = epoch_end_time - epoch_start_time  # Duration of the epoch

    print(f'Epoch [{epoch + 1}/{num_epochs}], Val Loss: {avg_val_loss:.4f}, Val Accuracy: {val_accuracy:.2f}%, Epoch Duration: {epoch_duration:.2f} seconds')


    # Write the current epoch, training loss, training accuracy, validation loss, and validation accuracy on a single line
    with open('training_history/train_hist.txt', 'a') as history:
        history.write(f"Epoch [{epoch+1}/{num_epochs}], Train Loss: {avg_train_loss:.4f}, Train Accuracy: {train_accuracy:.2f}%, Val Loss: {avg_val_loss:.4f}, Val Accuracy: {val_accuracy:.2f}%, Epoch Duration: {epoch_duration:.2f} seconds\n")

print('Finished Training')





with open (training_history_path, 'a') as history:
    history.write("TRAINING ENDED \n")

### SAVE MODEL #######
model_dir = "saved_models"
if not os.path.exists(model_dir):
    os.makedirs(model_dir)
model_path = os.path.join(model_dir, "model.pth")
torch.save(model.state_dict(), model_path)

print("Model Saved")


# with torch.no_grad():
with torch.inference_mode():
    n_correct = 0
    n_samples = 0
    n_class_correct = [0 for i in range(len(class_names))]
    n_class_samples = [0 for i in range(len(class_names))]

    for images, labels in dataloaders['test']:
        images = images.to(device)
        labels = labels.to(device) # pushes it to GPU if available
        outputs = model(images)

        _, predictions = torch.max(outputs, 1) # returns value and index
        # print(labels.size(0), labels)
        n_samples += labels.size(0)
        n_correct += (predictions == labels).sum().item()

        for i in range(labels.size(0)):
            label = labels[i]
            pred = predictions[i]
            if (label == pred):
                n_class_correct[label] += 1
            n_class_samples[label] += 1

    
    acc = 100.0 * n_correct / n_samples
    print(f'Accuracy of the network = {acc:.2f}%')

    for i in range(len(class_names)):
        if n_class_samples[i] != 0:
            acc = 100.0 * n_class_correct[i] / n_class_samples[i]
            print(f'Accuracy of {class_names[i]}: {acc:.2f}%')