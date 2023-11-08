#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Apr 22 16:39:06 2023

Trains Neural Net models to mimic the Leslie model
*Samples randomly from Leslie model; trains neural net on samples
    *NN uses custom loss function to focus on origin
*Outputs trained models if they are sufficiently accurate 
    * Accuracy assessed on local test data, also sampeled from leslie model

@author: cameronthieme
"""
sample_sizes = [10000, 100000, 1000000]
test_size = 1000000
epochs = 10000
weight_cap = 100


# setting up training for Leslie Model Neural Net
import torch
import datetime
import time


# define Leslie map, torch version
def leslie(x):
    '''
    input: 
        torch.array, with torch.size(1,2)
    output:
        Same data type as input
            image of that point under leslie map
    '''
    th1 = 19.6
    th2 = 23.68
    image0 = (th1 * x[:,0] + th2 * x[:,1]) * torch.exp (-0.1 * (x[:,0] + x[:,1]))
    image1 = 0.7 * x[:,0]
    image = torch.hstack([image0.resize(len(x),1),image1.resize(len(x),1)])
    return image


# Define the problem bounds
lower_bounds = [-0.001, -0.001]
upper_bounds = [90.0, 70.0]

# function to create training samples for Leslie Map
# appends origin
def sample_gen_leslie(sample_size = 1000, lower_bounds = [-0.001, -0.001], upper_bounds = [90.0, 70.0]):
    # Generate uniform points in the rectangle defined by lower_bounds, upper_bounds
    x_train = torch.hstack([
        torch.rand(sample_size, 1) * (upper_bounds[0] - lower_bounds[0]) + lower_bounds[0],
        torch.rand(sample_size, 1) * (upper_bounds[1] - lower_bounds[1]) + lower_bounds[1]
        ])
    x_train = torch.cat((x_train, torch.tensor([[0,0]], dtype = torch.float32)), dim = 0)
    y_train = leslie(x_train)
    return x_train, y_train[:,0].resize(sample_size + 1,1), y_train[:,1].resize(sample_size + 1,1)

# test sample generator
# no origin included
def test_samp_gen(sample_size = 1000, lower_bounds = [-0.001, -0.001], upper_bounds = [90.0, 70.0]):
    # Generate uniform points in the rectangle defined by lower_bounds, upper_bounds
    x_train = torch.hstack([
        torch.rand(sample_size, 1) * (upper_bounds[0] - lower_bounds[0]) + lower_bounds[0],
        torch.rand(sample_size, 1) * (upper_bounds[1] - lower_bounds[1]) + lower_bounds[1]
        ])
    y_train = leslie(x_train)
    return x_train, y_train[:,0].resize(sample_size,1), y_train[:,1].resize(sample_size,1)

# Define the neural net model with 3 hidden layers
# Hyperparameters haven't really been tuned at all
class Net(torch.nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.fcInput = torch.nn.Linear(2, 32)
        self.fc1 = torch.nn.Linear(32, 32)
        self.fc2 = torch.nn.Linear(32, 32)
        self.fc3 = torch.nn.Linear(32, 32)
        self.fc4 = torch.nn.Linear(32, 32)
        self.fc5 = torch.nn.Linear(32, 32)
        self.fc6 = torch.nn.Linear(32, 32)
        self.fc7 = torch.nn.Linear(32, 32)
        self.fc8 = torch.nn.Linear(32, 32)
        self.fcOutput = torch.nn.Linear(32, 1)
    def forward(self, x):
        x = torch.nn.functional.relu(self.fcInput(x))
        x = torch.nn.functional.relu(self.fc1(x))
        x = torch.nn.functional.relu(self.fc2(x))
        x = torch.nn.functional.relu(self.fc3(x))
        x = torch.nn.functional.relu(self.fc4(x))
        x = torch.nn.functional.relu(self.fc5(x))
        x = torch.nn.functional.relu(self.fc6(x))
        x = torch.nn.functional.relu(self.fc7(x))
        x = torch.nn.functional.relu(self.fc8(x))
        x = self.fcOutput(x)
        return x
    
    
# defining a custom loss function
# weights more heavily closer to the origin
# step function to make weights
def weight_stepper(two_tensor):
    if torch.linalg.norm(two_tensor) <= 1/weight_cap:
        return torch.tensor(weight_cap, dtype = torch.float32)
    elif torch.linalg.norm(two_tensor) < 1:
        return 1/torch.linalg.norm(two_tensor)
    else:
        return torch.tensor(1, dtype = torch.float32)

def my_loss_fn(train_input, output, target):
    weightList = torch.tensor(
        [weight_stepper(train_input[index]) for index in range(len(train_input))]
        ).resize(len(train_input),1)
    return torch.sum(weightList * (output - target) ** 2)/len(train_input)

# trainer with custom loss function
def trainer(net, x_train, y_train, optimizer, epochs = 1000, batch_size = 64, loss_fn = my_loss_fn):
    for epoch in range(epochs):
        # Randomly shuffle the training data
        perm = torch.randperm(len(x_train))
        x_train = x_train[perm]
        y_train = y_train[perm] 
        for i in range(0, len(x_train), batch_size):
            x_batch = x_train[i:i+batch_size]
            y_batch = y_train[i:i+batch_size]
            optimizer.zero_grad()
            outputs = net(x_batch)
            loss = loss_fn(x_batch,outputs, y_batch)
            loss.backward()
            optimizer.step()
    return net


for sample_size in sample_sizes:
    # seeing how long training takes
    latestTime = time.time()    
    
    # gentting training samples
    x_train, y_train0, y_train1 = sample_gen_leslie(sample_size = sample_size)
    
    # x_train, y_train0, y_train1 = comp_samp_gen(sample_size = sample_size)
    
    # initializing models
    net0 = Net()
    net1 = Net()
    
    # optimizer and learning rate
    learning_rate = 0.0001
    optimizer0 = torch.optim.SGD(net0.parameters(), lr=learning_rate)
    optimizer1 = torch.optim.SGD(net1.parameters(), lr=learning_rate)
    
    # train neural net
    trainer(net0, x_train = x_train, y_train = y_train0, optimizer = optimizer0, epochs = epochs)
    trainer(net1, x_train = x_train, y_train = y_train1, optimizer = optimizer1, epochs = epochs)
    
    # Generate some test data
    x_test, y_test0, y_test1 = test_samp_gen(sample_size = test_size)
    
    predicted0 = net0(x_test)
    predicted1 = net1(x_test)
    diffTensor0 = abs(y_test0 - predicted0)
    diffTensor1 = abs(y_test1 - predicted1)
    max0 = torch.max(diffTensor0)
    max1 = torch.max(diffTensor1)
    
    
    # Print the results
    print('Training Size:', sample_size)
    print("max0:", torch.max(diffTensor0))
    print("max1:", torch.max(diffTensor1))
    
    # print computation time
    executionTime = time.time() - latestTime
    print('Training time for this model in seconds:', executionTime)
    
    # save result if max is good
    if max0 < 1:
        print('**** SMALL ERROR ****') 
        print('Training Size:', sample_size)
        print("max0:", max0)
        print("max1:", max1)
        curDate = datetime.datetime.now()
        torch.save(net0.state_dict(), 'zNN_leslie0_size' + str(sample_size) +'_' + curDate.strftime("%b") + '_' + curDate.strftime('%d') + 'at'+ curDate.strftime('%I') + '_' + curDate.strftime('%M') + '_' + curDate.strftime('%S'))
        torch.save(net1.state_dict(), 'zNN_leslie1_size' + str(sample_size) +'_' + curDate.strftime("%b") + '_' + curDate.strftime('%d') + 'at'+ curDate.strftime('%I') + '_' + curDate.strftime('%M') + '_' + curDate.strftime('%S'))
        print('files saved as:')
        print('zNN_leslie0_size' + str(sample_size) +'_' + curDate.strftime("%b") + '_' + curDate.strftime('%d') + 'at'+ curDate.strftime('%I') + '_' + curDate.strftime('%M') + '_' + curDate.strftime('%S'))
        print('zNN_leslie1_size' + str(sample_size) +'_' + curDate.strftime("%b") + '_' + curDate.strftime('%d') + 'at'+ curDate.strftime('%I') + '_' + curDate.strftime('%M') + '_' + curDate.strftime('%S'))
        
        
print('learning rate:', learning_rate)
print('Epochs:', epochs)