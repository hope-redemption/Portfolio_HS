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
import torch.utils.data as Data
import torchvision
import torchvision.transforms as transforms

########################################################################################################################

# read data
#############################################################################
# data_train1
data_train1 = pd.read_csv("DataLabel1_train.csv")
# data_train1 = pd.read_csv("fe_DataLabel1_train.csv")
print(data_train1.shape)
# print(data)

# get numpy matrix which only contains data (do not contain the title)
data_train1 = np.array(data_train1)
print(data_train1.shape)
print(len(data_train1))
print(data_train1.shape[1])

# data_test1
print('  ')
data_test1 = pd.read_csv("DataLabel1_test.csv")
# data_test1 = pd.read_csv("fe_DataLabel1_test.csv")
print(data_test1.shape)
# print(data)

# get numpy matrix which only contains data (do not contain the title)
data_test1 = np.array(data_test1)
print(data_test1.shape)
print(len(data_test1))
print(data_test1.shape[1])
#############################################################################

# data_train2
data_train2 = pd.read_csv("DataLabel2_train.csv")
# data_train2 = pd.read_csv("fe_DataLabel2_train.csv")
print(data_train2.shape)
# print(data)

# get numpy matrix which only contains data (do not contain the title)
data_train2 = np.array(data_train2)
print(data_train2.shape)
print(len(data_train2))
print(data_train2.shape[1])

# data_test2
print('  ')
data_test2 = pd.read_csv("DataLabel2_test.csv")
# data_test2 = pd.read_csv("fe_DataLabel2_test.csv")
print(data_test2.shape)
# print(data)

# get numpy matrix which only contains data (do not contain the title)
data_test2 = np.array(data_test2)
print(data_test2.shape)
print(len(data_test2))
print(data_test2.shape[1])

#############################################################################

# data_train3
data_train3 = pd.read_csv("DataLabel3_train.csv")
# data_train3 = pd.read_csv("fe_DataLabel3_train.csv")
print(data_train3.shape)
# print(data)

# get numpy matrix which only contains data (do not contain the title)
data_train3 = np.array(data_train3)
print(data_train3.shape)
print(len(data_train3))
print(data_train3.shape[1])

# data_test3
print('  ')
data_test3 = pd.read_csv("DataLabel3_test.csv")
# data_test3 = pd.read_csv("fe_DataLabel3_test.csv")
print(data_test3.shape)
# print(data)

# get numpy matrix which only contains data (do not contain the title)
data_test3 = np.array(data_test3)
print(data_test3.shape)
print(len(data_test3))
print(data_test3.shape[1])

#############################################################################

# data_train4
print('  ')
data_train4 = pd.read_csv("DataLabel4_train.csv")
# data_train4 = pd.read_csv("fe_DataLabel4_train.csv")
print(data_train4.shape)
# print(data)

# get numpy matrix which only contains data (do not contain the title)
data_train4 = np.array(data_train4)
print(data_train4.shape)
print(len(data_train4))
print(data_train4.shape[1])

# data_test4
print('  ')
data_test4 = pd.read_csv("DataLabel4_test.csv")
# data_test4 = pd.read_csv("fe_DataLabel4_test.csv")
print(data_test4.shape)
# print(data)

# get numpy matrix which only contains data (do not contain the title)
data_test4 = np.array(data_test4)
print(data_test4.shape)
print(len(data_test4))
print(data_test4.shape[1])

#############################################################################

# data_train5
print('  ')
data_train5 = pd.read_csv("DataLabel10_train.csv")
# data_train5 = pd.read_csv("fe_DataLabel5_train.csv")
print(data_train5.shape)
# print(data)

# get numpy matrix which only contains data (do not contain the title)
data_train5 = np.array(data_train5)
print(data_train5.shape)
print(len(data_train5))
print(data_train5.shape[1])

# data_test5
print('  ')
data_test5 = pd.read_csv("DataLabel10_test.csv")
# data_test5 = pd.read_csv("fe_DataLabel5_test.csv")
print(data_test5.shape)
# print(data)

# get numpy matrix which only contains data (do not contain the title)
data_test5 = np.array(data_test5)
print(data_test5.shape)
print(len(data_test5))
print(data_test5.shape[1])

########################################################################################################################
# get x_train
x_train_row = len(data_train1) + len(data_train2) + len(data_train3) + len(data_train4) + len(data_train5)
x_train_col = data_train2.shape[1] - 4
# print(x_train_row)
# print(x_train_col)
x_train = np.zeros((x_train_row, x_train_col))
# print(x_train.shape)
# print(x_train[0:len(data_train2)][:].shape)
for i in range(x_train_row):
    if i < len(data_train1):
        x_train[i][:] = data_train1[i][4:data_train1.shape[1]]
    if len(data_train2) <= i < (len(data_train1) + len(data_train2)):
        x_train[i][:] = data_train2[i - len(data_train1)][4:data_train2.shape[1]]
    if (len(data_train1) + len(data_train2)) <= i < (len(data_train1) + len(data_train2) + len(data_train3)):
        x_train[i][:] = data_train3[i - len(data_train1) - len(data_train2)][4:data_train3.shape[1]]
    if (len(data_train1) + len(data_train2) + len(data_train3)) <= i < (
            len(data_train1) + len(data_train2) + len(data_train3) + len(data_train4)):
        x_train[i][:] = data_train4[i - len(data_train1) - len(data_train2) - len(data_train3)][4:data_train4.shape[1]]
    if (len(data_train1) + len(data_train2) + len(data_train3) + len(data_train4)) <= i:
        x_train[i][:] = data_train5[i - len(data_train1) - len(data_train2) - len(data_train3) - len(data_train4)][
                        4:data_train5.shape[1]]

print(x_train.shape)
# print(x_train)

# get y_train
y_train_row = len(data_train1) + len(data_train2) + len(data_train3) + len(data_train4) + len(data_train5)
y_train_col = 1
y_train = np.zeros((y_train_row, y_train_col))
for i in range(y_train_row):
    if i < len(data_train1):
        y_train[i][0] = data_train1[i][0]
    if len(data_train2) <= i < (len(data_train1) + len(data_train2)):
        y_train[i][0] = data_train2[i - len(data_train1)][0]
    if (len(data_train1) + len(data_train2)) <= i < (len(data_train1) + len(data_train2) + len(data_train3)):
        y_train[i][0] = data_train3[i - len(data_train1) - len(data_train2)][0]
    if (len(data_train1) + len(data_train2) + len(data_train3)) <= i < (
            len(data_train1) + len(data_train2) + len(data_train3) + len(data_train4)):
        y_train[i][0] = data_train4[i - len(data_train1) - len(data_train2) - len(data_train3)][0]
    if (len(data_train1) + len(data_train2) + len(data_train3) + len(data_train4)) <= i:
        y_train[i][0] = data_train5[i - len(data_train1) - len(data_train2) - len(data_train3) - len(data_train4)][0]

print(y_train.shape)

# get x_test
x_test_row = len(data_test1) + len(data_test2) + len(data_test3) + len(data_test4) + len(data_test5)
x_test_col = data_test2.shape[1] - 4
x_test = np.zeros((x_test_row, x_test_col))
for i in range(x_test_row):
    if i < len(data_test1):
        x_test[i][:] = data_test1[i][4:data_test1.shape[1]]
    if len(data_test2) <= i < (len(data_test1) + len(data_test2)):
        x_test[i][:] = data_test2[i - len(data_test1)][4:data_test2.shape[1]]
    if (len(data_test1) + len(data_test2)) <= i < (len(data_test1) + len(data_test2) + len(data_test3)):
        x_test[i][:] = data_test3[i - (len(data_test1) + len(data_test2))][4:data_test3.shape[1]]
    if (len(data_test1) + len(data_test2) + len(data_test3)) <= i < (
            len(data_test1) + len(data_test2) + len(data_test3) + len(data_test4)):
        x_test[i][:] = data_test4[i - (len(data_test1) + len(data_test2) + len(data_test3))][4:data_test4.shape[1]]
    if (len(data_test1) + len(data_test2) + len(data_test3) + len(data_test4)) <= i:
        x_test[i][:] = data_test5[i - (len(data_test1) + len(data_test2) + len(data_test3) + len(data_test4))][
                       4:data_test5.shape[1]]

print(x_test.shape)

# get y_test
y_test_row = len(data_test1) + len(data_test2) + len(data_test3) + len(data_test4) + len(data_test5)
y_test_col = 1
y_test = np.zeros((y_test_row, y_test_col))
for i in range(y_test_row):
    if i < len(data_test1):
        y_test[i][0] = data_test1[i][0]
    if len(data_test2) <= i < (len(data_test1) + len(data_test2)):
        y_test[i][0] = data_test2[i - len(data_test1)][0]
    if (len(data_test1) + len(data_test2)) <= i < (len(data_test1) + len(data_test2) + len(data_test3)):
        y_test[i][0] = data_test3[i - (len(data_test1) + len(data_test2))][0]
    if (len(data_test1) + len(data_test2) + len(data_test3)) <= i < (
            len(data_test1) + len(data_test2) + len(data_test3) + len(data_test4)):
        y_test[i][0] = data_test4[i - (len(data_test1) + len(data_test2) + len(data_test3))][0]
    if (len(data_test1) + len(data_test2) + len(data_test3) + len(data_test4)) <= i:
        y_test[i][0] = data_test5[i - (len(data_test1) + len(data_test2) + len(data_test3) + len(data_test4))][0]

print(y_test.shape)

########################################################################################################################


########################################################################################################################

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Hyper-parameters
# sequence_length = 28
# input_size = 28
sequence_length = 31
input_size = 3
# hidden_size = 128
hidden_size = 20
# num_layers = 2
num_layers = 2
num_classes = 5
# batch_size = 100
batch_size = 50
num_epochs = 2
learning_rate = 0.01
# learning_rate = 0.01

########################################################################################################################

x_train = torch.tensor(x_train, dtype=torch.float32)
x_test = torch.tensor(x_test, dtype=torch.float32)
y_train = torch.tensor(y_train, dtype=torch.long)
y_test = torch.tensor(y_test, dtype=torch.long)

train_dataset = Data.TensorDataset(x_train, y_train)
test_dataset = Data.TensorDataset(x_test, y_test)

train_loader = Data.DataLoader(
    dataset=train_dataset,  # torch TensorDataset format
    batch_size=batch_size,  # mini batch size
    shuffle=True,  # random shuffle for training
    # num_workers=2,  # subprocesses for loading data
)

test_loader = Data.DataLoader(
    dataset=test_dataset,  # torch TensorDataset format
    batch_size=batch_size,  # mini batch size
    shuffle=False,  # random shuffle for training
    # num_workers=2,  # subprocesses for loading data
)


# Recurrent neural network (many-to-one)
class BiRNN(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_classes):
        super(BiRNN, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, bidirectional=True)
        self.fc = nn.Linear(hidden_size*2, num_classes)  # 2 for bidirection

    def forward(self, x):
        # Set initial hidden and cell states
        h0 = torch.zeros(self.num_layers*2, x.size(0), self.hidden_size).to(device)  # 2 for bidirection
        c0 = torch.zeros(self.num_layers*2, x.size(0), self.hidden_size).to(device)

        # Forward propagate LSTM
        x = x.to(torch.float32)
        out, _ = self.lstm(x, (h0, c0))  # out: tensor of shape (batch_size, seq_length, hidden_size*2)
        # print(out.shape)
        # print(out)

        # Decode the hidden state of the last time step
        out = self.fc(out[:, -1, :])
        return out


model = BiRNN(input_size, hidden_size, num_layers, num_classes).to(device)

# Loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

# names = ['bob', 'john', 'lisa']
# ages = [19, 20, 32]
# for name, age in zip(names, ages):
#     print("%s's age is %s" % (name, age))

# Train the model
total_step = len(train_loader)
print(total_step)
num = 0
for epoch in range(num_epochs):
    for i, (images, labels) in enumerate(train_loader):
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

        if (num + 1) % 50 == 0:
            print('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}'
                  .format(epoch + 1, num_epochs, num + 1, total_step, loss.item()))
        num = num + 1
    num = 0

# # Test the model
model.eval()

time_start = time.time()  # 记录开始时间
with torch.no_grad():
    correct = 0
    total = 0
    for i, (images, labels) in enumerate(test_loader):
        labels = labels.long()
        # print(labels.size(0))

        images = images.reshape(-1, sequence_length, input_size).to(device)
        labels = labels.to(device)
        labels = torch.squeeze(labels, dim=1)
        # print(labels)
        # print(labels.shape)
        outputs = model(images)
        print(outputs.data)
        # print(outputs.shape)

        _, predicted = torch.max(outputs.data, 1)
        print(predicted.shape)
        print(predicted)

        total += labels.size(0)
        # print(total)
        correct += (predicted == labels).sum().item()
        # print(correct)
        # print('   ')

    print('Test Accuracy of the model on the 5097 test images: {} %'.format(100 * correct / total))

# # Save the model checkpoint
# torch.save(model.state_dict(), 'model.ckpt')

time_end = time.time()  # 记录结束时间
time_sum = time_end - time_start  # 计算的时间差为程序的执行时间，单位为秒/s
print(time_sum)
