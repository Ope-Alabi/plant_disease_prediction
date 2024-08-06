import torch
from torch import nn
import torch.quantization
import numpy as np
import pandas as pd
import torchvision
import os
# from PIL import Image
import matplotlib.pyplot as plt
# from torch.ao.quantization import QuantStub, DeQuantStub
# from torch.profiler import profile, record_function, ProfilerActivity
# from typing import List, Any
# from torchinfo import summary
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix
# from enum import Enum
# from torchmetrics.classification import MulticlassAccuracy
# from torchvision.datasets import VisionDataset
from typing import Tuple
# from torch.utils.data import ConcatDataset
import time

# Device configuration
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class DiseaseClassifier(nn.Module):
    """Define a custom model that wraps a pre-trained model for classification
    on the Soybean plant diseases dataset.
    """
    def __init__(self, backbone, load_pretrained, pddd, device, rank, output_shape=8, freeze=False):
        super().__init__()
        # assert backbone in backbones
        self.seed = 49
        self.backbone = backbone
        self.pretrained_model = None
        self.classifier_layers = []
        self.new_layers = []
        self.pddd = pddd
        self.load_pretrained = load_pretrained
        self.freeze = freeze
        self.device = device
        self.rank = rank
        # self.output_shape = output_shape
        if backbone == "resnet50":
            if load_pretrained:
                # weights = torchvision.models.DenseNet201_Weights.DEFAULT
                # model = torchvision.models.densenet201(weights=weights).to(device)
                self.pretrained_model = torchvision.models.resnet50(weights=torchvision.models.ResNet50_Weights.DEFAULT)
            elif pddd and not load_pretrained:
                self.pretrained_model = torchvision.models.resnet50(weights=None)
                state_dict = torch.load(f"./INPUT/BASE_MODELS/pddd_{backbone}.pth", map_location=self.device)
            else:
                self.pretrained_model = torchvision.models.resnet50(weights=None)
            # end if

            self.classifier_layers = [self.pretrained_model.fc]
            
            # Replace the final layer with a classifier for 102 classes for the Flowers 102 dataset.
            self.pretrained_model.fc = nn.Sequential(
                    nn.Linear(in_features=self.pretrained_model.fc.in_features, out_features=128, bias=True),
                    nn.ReLU(inplace=True),
                    nn.Dropout(p=0.2),
                    nn.Linear(in_features=128, out_features=64, bias=True),
                    nn.ReLU(inplace=True),
                    nn.Dropout(p=0.3),
                    nn.Linear(in_features=64, out_features=output_shape, bias=True)  # Number of output classes
                    )
            if pddd: # If using PDDD base model
                self.pretrained_model.load_state_dict(state_dict, strict=False)

            if freeze:
                for name, param in self.pretrained_model.named_parameters():
                    if "fc" not in name:
                        param.requires_grad = False
                    # end if
                # end for
            self.new_layers = [self.pretrained_model.fc]
        elif backbone == "densenet201":
            if load_pretrained:
                self.pretrained_model = torchvision.models.densenet201(weights=torchvision.models.DenseNet201_Weights.DEFAULT)
            elif pddd and not load_pretrained:
                self.pretrained_model = torchvision.models.densenet201(weights=None)
                state_dict = torch.load(f"./INPUT/BASE_MODELS/pddd_{backbone}.pth", map_location=self.device)
            else:
                self.pretrained_model = torchvision.models.densenet201(weights=None)
            # end if

            self.classifier_layers = [self.pretrained_model.classifier]
            # Replace the final layer with a classifier for the number of classes in the dataset.
            self.pretrained_model.classifier = nn.Sequential(
                    nn.Linear(in_features=self.pretrained_model.classifier.in_features, out_features=128, bias=True),
                    nn.ReLU(inplace=True),
                    nn.Dropout(p=0.2),
                    nn.Linear(in_features=128, out_features=64, bias=True),
                    nn.ReLU(inplace=True),
                    nn.Dropout(p=0.3),
                    nn.Linear(in_features=64, out_features=output_shape, bias=True)  # Number of output classes
                    )
            if pddd:
                self.pretrained_model.load_state_dict(state_dict, strict=False)

            if freeze:
                for name, param in self.pretrained_model.named_parameters():
                    if "classifier" not in name:
                        param.requires_grad = False
                    # end if
                # end for

            self.new_layers = [self.pretrained_model.classifier]
            
        elif backbone == "vgg16":
            if load_pretrained:
                self.pretrained_model = torchvision.models.vgg16_bn(weights=torchvision.models.VGG16_BN_Weights.DEFAULT)
            elif pddd and not load_pretrained:
                self.pretrained_model = torchvision.models.vgg16_bn(weights=None)
                state_dict = torch.load(f"./INPUT/BASE_MODELS/pddd_{backbone}.pth", map_location=self.device)
            else:
                self.pretrained_model = torchvision.models.vgg16_bn(weights=None)
            # end if

            self.classifier_layers = [self.pretrained_model.classifier]
            # Replace the final layer with a classifier for 102 classes for the Flowers 102 dataset.
            self.pretrained_model.classifier[6] = nn.Sequential(  #nn.Linear(in_features=4096, out_features=38, bias=True)
                    nn.Linear(in_features=self.pretrained_model.classifier[6].in_features, out_features=128, bias=True),
                    nn.ReLU(inplace=True),
                    nn.Dropout(p=0.2),
                    nn.Linear(in_features=128, out_features=64, bias=True),
                    nn.ReLU(inplace=True),
                    nn.Dropout(p=0.3),
                    nn.Linear(in_features=64, out_features=output_shape, bias=True)  # Number of output classes
                    ) 
            if pddd:
                self.pretrained_model.load_state_dict(state_dict, strict=False)
            
            if freeze:
                for name, param in self.pretrained_model.named_parameters():
                    if "classifier" not in name:
                        param.requires_grad = False
                    # end if
                # end for

            self.new_layers = [self.pretrained_model.classifier[6]]

        elif backbone == "efficientnet_b0":
            if load_pretrained:
                self.pretrained_model = torchvision.models.efficientnet_b0(weights=torchvision.models.EfficientNet_B0_Weights.DEFAULT)
            elif pddd and not load_pretrained:
                self.pretrained_model = torchvision.models.efficientnet_b0(weights=None)
                state_dict = torch.load(f"./INPUT/BASE_MODELS/pddd_{backbone}.pth", map_location=self.device)
            else:
                self.pretrained_model = torchvision.models.efficientnet_b0(weights=None)
            # end if

            self.classifier_layers = [self.pretrained_model.classifier]
            # Replace the final layer with a classifier for 102 classes for the Flowers 102 dataset.
            self.pretrained_model.classifier[1] = nn.Sequential(
                    nn.Linear(in_features=self.pretrained_model.classifier[1].in_features, out_features=128, bias=True),
                    nn.ReLU(inplace=True),
                    nn.Dropout(p=0.2),
                    nn.Linear(in_features=128, out_features=64, bias=True),
                    nn.ReLU(inplace=True),
                    nn.Dropout(p=0.3),
                    nn.Linear(in_features=64, out_features=output_shape, bias=True)  # Number of output classes
                    ) 
            if pddd:
                self.pretrained_model.load_state_dict(state_dict, strict=False)
            
            if freeze:
                for name, param in self.pretrained_model.named_parameters():
                    if "classifier" not in name:
                        param.requires_grad = False
                    # end if
                # end for

            self.new_layers = [self.pretrained_model.classifier[1]]

        # Dummy Param to be able to check the device on which this model is.
        # From https://stackoverflow.com/questions/58926054/how-to-get-the-device-type-of-a-pytorch-module-conveniently
        self.dummy_param = nn.Parameter(torch.empty(0))

    def forward(self, x):
        return self.pretrained_model(x)
    
    def summarize_model(self):
        return summary(self.pretrained_model, 
        input_size=(32, 3, 224, 224), 
        verbose=0,
        col_names=["input_size", "output_size", "num_params", "trainable"],
        col_width=20,
        row_settings=["var_names"])
    

    def set_seeds(self):
        torch.manual_seed(self.seed)
        if self.device.type == 'cuda':
            torch.cuda.manual_seed(self.seed)

    def train_model(self, criterion, optimizer, dataloaders, num_epochs):
        checkpoint_dir = './OUTPUT/CHECKPOINT'
        if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)

        best_val_loss = float('inf')
        train_hist_dir = './OUTPUT/LOGS'
        if not os.path.exists(train_hist_dir):
            os.makedirs(train_hist_dir)
        training_history_path = os.path.join(train_hist_dir, f"{self.backbone}_train_hist[pddd={self.pddd},frozen={self.freeze},pretrained={self.load_pretrained}].txt")

        with open(training_history_path, 'w') as history:
            history.write(f"STARTING TO TRAIN [pddd={self.pddd},frozen={self.freeze},pretrained={self.load_pretrained}]\n ")

        print("TRAINING STARTED")

        for epoch in range(num_epochs):
            epoch_start_time = time.time()
            running_loss = 0.0
            n_correct_train = 0
            n_samples_train = 0
            self.train()

            for i, (images, labels) in enumerate(dataloaders['train']):
                images = images.to(self.device)
                labels = labels.to(self.device)

                outputs = self(images)
                loss = criterion(outputs, labels)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                running_loss += loss.item()
                _, predictions = torch.max(outputs, 1)
                n_samples_train += labels.size(0)
                n_correct_train += (predictions == labels).sum().item()

                if (i + 1) % 100 == 0:
                    print(f'Epoch [{epoch + 1}/{num_epochs}], Step [{i + 1}/{len(dataloaders["train"])}], Loss: {loss.item():.4f}')

            avg_train_loss = running_loss / len(dataloaders['train'])
            train_accuracy = 100.0 * n_correct_train / n_samples_train
            print(f'Epoch [{epoch + 1}/{num_epochs}], Train Loss: {avg_train_loss:.4f}, Train Accuracy: {train_accuracy:.2f}%')

            self.eval()
            val_running_loss = 0.0
            n_correct_val = 0
            n_samples_val = 0
            with torch.no_grad():
                for images, labels in dataloaders['val']:
                    images = images.to(self.device)
                    labels = labels.to(self.device)
                    outputs = self(images)
                    loss = criterion(outputs, labels)
                    val_running_loss += loss.item()
                    _, predictions = torch.max(outputs, 1)
                    n_samples_val += labels.size(0)
                    n_correct_val += (predictions == labels).sum().item()

            avg_val_loss = val_running_loss / len(dataloaders['val'])
            val_accuracy = 100.0 * n_correct_val / n_samples_val

            if avg_val_loss < best_val_loss and self.device  == "cuda:0":
                best_val_loss = avg_val_loss
                torch.save(self.state_dict(), os.path.join(checkpoint_dir, f'{self.backbone}_best_model[pddd={self.pddd},frozen={self.freeze},pretrained={self.load_pretrained}].pth'))
                print(f'Saved Best Model with Val Loss: {avg_val_loss:.4f}')

            epoch_end_time = time.time()
            epoch_duration = epoch_end_time - epoch_start_time

            print(f'Epoch [{epoch + 1}/{num_epochs}], Val Loss: {avg_val_loss:.4f}, Val Accuracy: {val_accuracy:.2f}%, Epoch Duration: {epoch_duration:.2f} seconds')

            with open(training_history_path, 'a') as history:
                history.write(f"Epoch [{epoch+1}/{num_epochs}], Train Loss: {avg_train_loss:.4f}, Train Accuracy: {train_accuracy:.2f}%, Val Loss: {avg_val_loss:.4f}, Val Accuracy: {val_accuracy:.2f}%, Epoch Duration: {epoch_duration:.2f} seconds\n")

        print('Finished Training')
        with open(training_history_path, 'a') as history:
            history.write("TRAINING ENDED \n")

    def save_model(self):
        path=f'./OUTPUT/SAVED_MODELS/{self.backbone}_model[pddd={self.pddd},frozen={self.freeze},pretrained={self.load_pretrained}].pth'
        model_dir = os.path.dirname(path)
        if not os.path.exists(model_dir):
            os.makedirs(model_dir)

        torch.save(self.state_dict(), path)
        print("Model Saved")

    def calculate_metrics(self, true_labels, predicted_labels, class_names, average='macro'):
        """
        Calculate precision, recall, F1 score, and confusion matrix.

        Parameters:
        - true_labels (list or numpy array): True class labels
        - predicted_labels (list or numpy array): Predicted class labels by the model
        - average (str): Averaging method for multi-class data ('micro', 'macro', 'weighted', 'samples')

        Returns:
        - precision (float): Precision score
        - recall (float): Recall score
        - f1 (float): F1 score
        - cm (numpy array): Confusion matrix
        """

        precision = precision_score(true_labels, predicted_labels, average=average)
        recall = recall_score(true_labels, predicted_labels, average=average)
        f1 = f1_score(true_labels, predicted_labels, average=average)
        cm = confusion_matrix(true_labels, predicted_labels)

        # Create a DataFrame for the confusion matrix with class names
        cm_df = pd.DataFrame(cm, index=class_names, columns=class_names)

        return precision, recall, f1, cm_df

    def test_model(self, dataloaders, class_names):
        test_evaluation_dir = f'./OUTPUT/EVALUATION'
        if not os.path.exists(test_evaluation_dir):
            os.makedirs(test_evaluation_dir)
        test_evaluation_path = os.path.join(test_evaluation_dir, f"{self.backbone}_test_result[pddd={self.pddd},frozen={self.freeze},pretrained={self.load_pretrained}].txt")

        with open(test_evaluation_path, 'w') as history:
            history.write("EVALUATING MODEL \n")
        self.eval()

        true_labels = []
        predicted_labels = []

        with torch.no_grad():
            n_correct = 0
            n_samples = 0
            n_class_correct = [0 for _ in range(len(class_names))]
            n_class_samples = [0 for _ in range(len(class_names))]
            # true_labels = []
            # predicted_labels = []

            for images, labels in dataloaders['test']:
                images = images.to(self.device)
                labels = labels.to(self.device)
                outputs = self(images)

                _, predictions = torch.max(outputs, 1)
                n_samples += labels.size(0)
                n_correct += (predictions == labels).sum().item()

                true_labels.extend(labels.cpu().numpy())
                predicted_labels.extend(predictions.cpu().numpy())

                for i in range(labels.size(0)):
                    label = labels[i]
                    pred = predictions[i]
                    if label == pred:
                        n_class_correct[label] += 1
                    n_class_samples[label] += 1

            acc = 100.0 * n_correct / n_samples
            precision, recall, f1, cm = self.calculate_metrics(true_labels, predicted_labels, class_names)

            with open(test_evaluation_path, 'a') as history:
                    history.write(f"Accuracy of the network = {acc:.2f}%\n")
            print(f"Accuracy of the network = {acc:.2f}%\n")


            for i in range(len(class_names)):
                if n_class_samples[i] != 0:
                    acc = 100.0 * n_class_correct[i] / n_class_samples[i]
                    with open(test_evaluation_path, 'a') as history:
                        history.write(f"Accuracy of {class_names[i]}: {acc:.2f}%\n")
                    print(f'Accuracy of {class_names[i]}: {acc:.2f}%\n')
            
            with open(test_evaluation_path, 'a') as history:
                history.write('\t'.join([''] + cm.columns.tolist()) + '\n')
                for idx, row in cm.iterrows():
                    history.write('\t'.join([idx] + row.astype(str).tolist()) + '\n')
                # history.write(f"\nPrecision: {precision:.4f}\nRecall: {recall:.4f}\nF1 Score: {f1:.4f}\nConfusion Matrix:\n{cm}")
            print(f'Precision: {precision:.4f}\nRecall: {recall:.4f}\nF1 Score: {f1:.4f}\nConfusion Matrix:\n{cm}')


        
# end class

# Sanity check to see if we can run a single forward pass with this model
# when it is provided an input with the expected shape.
# backbones = ["resnet50", "vgg16", "densenet201"]
# for backbone in backbones:
#     print(f"Backbone: {backbone}")
#     fc_test = DiseaseClassifier(backbone=backbone, load_pretrained=True)
#     x = torch.randn(4, 3, 224, 224)
#     y = fc_test(x)
#     print(x.shape, y.shape)
    # print(fc_test.get_optimizer_params())