# coding=utf-8
import matplotlib.pyplot as plt

from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.neural_network import MLPClassifier
import seaborn as sb
from sklearn.svm import LinearSVC
from sklearn.decomposition import PCA
from scipy.special import expit
import time
import cv2

import pandas as pd
import numpy as np

import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms

########################################################################################################################

# read data
# data_train2
# data_train2 = pd.read_csv("DataLabel2_train.csv")
data_train2 = pd.read_csv("fe_DataLabel1_train.csv")
print(data_train2.shape)
# print(data)

# get numpy matrix which only contains data (do not contain the title)
data_train2 = np.array(data_train2)
print(data_train2.shape)
print(len(data_train2))
print(data_train2.shape[1])

# data_test2
print('  ')
# data_test2 = pd.read_csv("DataLabel2_test.csv")
data_test2 = pd.read_csv("fe_DataLabel1_test.csv")
print(data_test2.shape)
# print(data)

# get numpy matrix which only contains data (do not contain the title)
data_test2 = np.array(data_test2)
print(data_test2.shape)
print(len(data_test2))
print(data_test2.shape[1])

# data_train4
print('  ')
# data_train4 = pd.read_csv("DataLabel4_train.csv")
data_train4 = pd.read_csv("fe_DataLabel10_train.csv")
print(data_train4.shape)
# print(data)

# get numpy matrix which only contains data (do not contain the title)
data_train4 = np.array(data_train4)
print(data_train4.shape)
print(len(data_train4))
print(data_train4.shape[1])

# data_test4
print('  ')
# data_test4 = pd.read_csv("DataLabel4_test.csv")
data_test4 = pd.read_csv("fe_DataLabel10_test.csv")
print(data_test4.shape)
# print(data)

# get numpy matrix which only contains data (do not contain the title)
data_test4 = np.array(data_test4)
print(data_test4.shape)
print(len(data_test4))
print(data_test4.shape[1])

########################################################################################################################
print('  ')
# get x_train
x_train_row = len(data_train2) + len(data_train4)
x_train_col = data_train2.shape[1] - 4
# print(x_train_row)
# print(x_train_col)
x_train = np.zeros((x_train_row, x_train_col))
# print(x_train.shape)
# print(x_train[0:len(data_train2)][:].shape)
for i in range(x_train_row):
    if i < len(data_train2):
        x_train[i][:] = data_train2[i][4:data_train2.shape[1]]
    if i >= len(data_train2):
        x_train[i][:] = data_train4[i - len(data_train2)][4:data_train4.shape[1]]

print(x_train.shape)
# print(x_train)

# get y_train
y_train_row = len(data_train2) + len(data_train4)
y_train_col = 1
y_train = np.zeros((y_train_row, y_train_col))
for i in range(y_train_row):
    if i < len(data_train2):
        y_train[i][0] = data_train2[i][0]
    if i >= len(data_test2):
        y_train[i][0] = data_train4[i - len(data_train2)][0]
print(y_train.shape)

# get train data
train_data = np.zeros((x_train_row, x_train_col + 1))
print(train_data.shape)
for i in range(x_train_row):
    train_data[i][0] = y_train[i]
    train_data[i][1:x_train_col + 1] = x_train[i]

# print(train_data[0])
# print(train_data[0][0])
# print(train_data[0][1])
# print(train_data[0][23])

# get x_test
x_test_row = len(data_test2) + len(data_test4)
x_test_col = data_test2.shape[1] - 4
x_test = np.zeros((x_test_row, x_test_col))
for i in range(x_test_row):
    if i < len(data_test2):
        x_test[i][:] = data_test2[i][4:data_test2.shape[1]]
    if i >= len(data_test2):
        x_test[i][:] = data_test4[i - len(data_test2)][4:data_test4.shape[1]]
print(x_test.shape)

# get y_test
y_test_row = len(data_test2) + len(data_test4)
y_test_col = 1
y_test = np.zeros((y_test_row, y_test_col))
for i in range(y_test_row):
    if i < len(data_test2):
        y_test[i][0] = data_test2[i][0]
    if i >= len(data_test2):
        y_test[i][0] = data_test4[i - len(data_test2)][0]
print(y_test.shape)

########################################################################################################################

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Hyper-parameters
# sequence_length = 28
# input_size = 28
sequence_length = 1
input_size = 23
# hidden_size = 128
hidden_size = 20
# num_layers = 2
num_layers = 1
num_classes = 2
# batch_size = 100
batch_size = 50
num_epochs = 10
learning_rate = 0.0001
# learning_rate = 0.01

# MNIST dataset
# train_dataset = torchvision.datasets.MNIST(root='../../data/',
#                                            train=True,
#                                            transform=transforms.ToTensor(),
#                                            download=True)
# # print(train_dataset[0])
#
# test_dataset = torchvision.datasets.MNIST(root='../../data/',
#                                           train=False,
#                                           transform=transforms.ToTensor())

# Data loader
# train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
#                                            batch_size=batch_size,
#                                            shuffle=True)
#
# test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
#                                           batch_size=batch_size,
#                                           shuffle=False)
train_loader_x = torch.utils.data.DataLoader(dataset=x_train,
                                             batch_size=batch_size,
                                             shuffle=False)
print(train_loader_x)
print(len(train_loader_x))
#
# images = next(iter(train_loader_x))
# print(images.shape)
# print(images)
# data = next(iter(train_loader))
# print(data.shape)
# print(len(data))
# print(data)
# print('   ')
# label = torch.gather(data, 1, 0)
# label = np.zeros((len(data), 1))
# for i in range(len(data)):
#     label[i] = data[i][0]
# print(label)
# image = data[:][1:23]
# print(image)
# print('   ')
# print(labels.shape)
# print(labels.size()[0])

train_loader_y = torch.utils.data.DataLoader(dataset=y_train,
                                             batch_size=batch_size,
                                             shuffle=False)
print(train_loader_y)
print(len(train_loader_y))

# labels = next(iter(train_loader_y))
# print(labels.shape)
# print(labels)
# # print(labels.shape)
# # print(labels.size()[0])
#
# A = [images, labels]
# print(A)
# train_loader = torch.cat([images, labels], dim=1)
# print(train_loader.shape)
# print(train_loader)
# print('  ')

# train_loader = [train_loader_x, train_loader_y]
# print('     ')
# print(train_loader)
# train_loader_c, train_loader_d = train_loader
# print('     ')
# print(train_loader_c)
# print(train_loader_d)
# print('     ')
# print(len(train_loader))

test_loader_x = torch.utils.data.DataLoader(dataset=x_test,
                                          batch_size=batch_size,
                                          shuffle=False)

test_loader_y = torch.utils.data.DataLoader(dataset=y_test,
                                          batch_size=batch_size,
                                          shuffle=False)


# Recurrent neural network (many-to-one)
class RNN(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_classes):
        super(RNN, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        # Set initial hidden and cell states
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(device)

        # Forward propagate LSTM
        x = x.to(torch.float32)
        out, _ = self.lstm(x, (h0, c0))  # out: tensor of shape (batch_size, seq_length, hidden_size)

        # Decode the hidden state of the last time step
        out = self.fc(out[:, -1, :])
        return out


model = RNN(input_size, hidden_size, num_layers, num_classes).to(device)

# Loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

# names = ['bob', 'john', 'lisa']
# ages = [19, 20, 32]
# for name, age in zip(names, ages):
#     print("%s's age is %s" % (name, age))

# Train the model
total_step = len(train_loader_x)
print(total_step)
num = 0
for epoch in range(num_epochs):
    for images, labels in zip(train_loader_x, train_loader_y):
        # print(num)
        labels = labels.long()

        images = images.reshape(-1, sequence_length, input_size).to(device)
        labels = labels.to(device)
        labels = torch.squeeze(labels, dim=1)
        # print(labels.shape)
        # print(labels)

        # Forward pass
        outputs = model(images)
        # print(outputs.shape)
        # print(outputs)
        # print(labels.shape)
        loss = criterion(outputs, labels)
        # print('    ')

        # Backward and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if (num + 1) % 4 == 0:
            print('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}'
                  .format(epoch + 1, num_epochs, num + 1, total_step, loss.item()))
        num = num + 1
    num = 0

# # Test the model
model.eval()
with torch.no_grad():
    correct = 0
    total = 0
    for images, labels in zip(test_loader_x, test_loader_y):
        labels = labels.long()
        # print(labels.size(0))

        images = images.reshape(-1, sequence_length, input_size).to(device)
        labels = labels.to(device)
        labels = torch.squeeze(labels, dim=1)
        # print(labels)
        # print(labels.shape)
        outputs = model(images)
        print(outputs.data)
        print(outputs.shape)

        _, predicted = torch.max(outputs.data, 1)
        print(predicted.shape)
        print(predicted)

        total += labels.size(0)
        print(total)
        correct += (predicted == labels).sum().item()
        print(correct)
        print('   ')


    print('Test Accuracy of the model on the 10000 test images: {} %'.format(100 * correct / total))

# # Save the model checkpoint
# torch.save(model.state_dict(), 'model.ckpt')
