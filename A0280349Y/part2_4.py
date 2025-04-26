#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
/*_____________________________EE5907_CA2_A0280349Y_____________________________*\
<part2_4.py> { 
    context         ; 
    purpose         applying CNN for classification; 
    used in         parent folder [A0280349Y]; 
    py version      3.10; 
    os              windows 10;
    ref(s)          ;       
    note            for master branch only; 
} // "<part2_4.py>"
/*_____________________________EE5907_CA2_A0280349Y_____________________________*\
"""


import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from A0280349Y.config import *


# settings for file saving;
assignment_name = os.path.basename(__file__).replace(".py", "")
#- [dir_part1_1] is for this <.py> only;
#-- create the results folder only when needed;
dir_thisPart = dir_part2_4
os.makedirs(dir_thisPart, exist_ok=True)


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class CNNClassifier(nn.Module):
    #- initialise layers; 
    def __init__(self):
        super(CNNClassifier, self).__init__()
        #- 1x input channel, 20 filters of 5x5, Stride1, no padding;
        self.conv1 = nn.Conv2d(1, 20, kernel_size=5, stride=1, padding=0)
        #- max pooling is 2x2 block, keep the largest value + shrink image by half;
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        #- 20x input channels, 50 filters of 5x5; 
        self.conv2 = nn.Conv2d(20, 50, kernel_size=5, stride=1, padding=0)
        #- additional pooling layer to shrink image;  
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        #- flatten output,  (50*5*5) features into 500 output; 
        self.fc1 = nn.Linear(50 * 5 * 5, 500)
        #- ReLU activation; 
        self.relu = nn.ReLU()
        #- convert to 26 neurons because {subjects_total} = 26;
        self.fc2 = nn.Linear(500, 26)

    def forward(self, x):
        x = self.pool1(self.conv1(x))
        x = self.pool2(self.conv2(x))
        x = x.view(x.size(0), -1)
        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        return x
    
    
def load_data():
    X_train = np.load(os.path.join(dir_part1_1, "X_train.npy"))
    y_train = np.load(os.path.join(dir_part1_1, "y_train.npy"))
    X_test = np.load(os.path.join(dir_part1_1, "X_test.npy"))
    y_test = np.load(os.path.join(dir_part1_1, "y_test.npy"))
    
    # reshape (N, 32, 32) and add channel dimension (N, 1, 32, 32)
    X_train = X_train.reshape(-1, 1, 32, 32)
    X_test = X_test.reshape(-1, 1, 32, 32)

    # convert to tensors
    X_train = torch.tensor(X_train, dtype=torch.float32)
    y_train = torch.tensor(y_train, dtype=torch.long)
    X_test = torch.tensor(X_test, dtype=torch.float32)
    y_test = torch.tensor(y_test, dtype=torch.long)

    return X_train, y_train, X_test, y_test


def train(model, train_loader, criterion, optimizer):
    model.train()
    #- initialise the accumulated loss;
    running_loss = 0.0
    #- initialise num of correct predictions;
    correct = 0
    #- initialise total num of samples processed; 
    total = 0
    for inputs, labels in train_loader:
        inputs, labels = inputs.to(device), labels.to(device)
        #-- reset gradient buffer before backpropagation; 
        optimizer.zero_grad()
        #-- get predicted outputs; 
        outputs = model(inputs)
        #-- computes how wrong the output is compared to the labels;
        loss = criterion(outputs, labels)
        #-- compute backpropagation;
        loss.backward()
        optimizer.step()
        #-- compute the num of wrong predictions;
        running_loss += loss.item()
        #-- compute the predicted label for each input; 
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        #-- compute the num of correct predictions;
        correct += (predicted == labels).sum().item()
        
    #- compute the average loss per batch;
    epoch_loss = running_loss / len(train_loader)
    epoch_acc = correct / total
    return epoch_loss, epoch_acc


def evaluate(model, test_loader, criterion):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)

            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    epoch_loss = running_loss / len(test_loader)
    epoch_acc = correct / total
    return epoch_loss, epoch_acc


def main():
    X_train, y_train, X_test, y_test = load_data()

    train_dataset = TensorDataset(X_train, y_train)
    test_dataset = TensorDataset(X_test, y_test)
    
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

    model = CNNClassifier().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    num_epochs = 20
    for epoch in range(num_epochs):
        train_loss, train_acc = train(model, train_loader, criterion, optimizer)
        print(f"Epoch {epoch+1}/{num_epochs}, Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}")

    test_loss, test_acc = evaluate(model, test_loader, criterion)
    print(f"\nFinal Test Accuracy: {test_acc:.4f}")

if __name__ == "__main__":
    main()
    
    