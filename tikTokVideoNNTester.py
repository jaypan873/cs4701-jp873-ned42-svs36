import json
import numpy as np
import os
import math
import pickle

import vectorizer
import tikTokVideoNN

import torch
import torch.nn as nn
from torch.nn import init
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, SubsetRandomSampler

def testing_data_loader(testing_data, batch_size=1):
    testing_dataset = tikTokVideoNN.TikTokDataset(testing_data)

    testing_indices = [i for i in range(len(testing_data))]

    testing_shuffler = SubsetRandomSampler(testing_indices)
    testing_data_loader = DataLoader(testing_dataset, batch_size=batch_size, sampler=testing_shuffler)

    return testing_data_loader

def test(net, data_loader):
    net.eval()
    running_loss = 0.0
    correct_instances = [0 for i in range(9)]
    total_instances = 0

    for i, data in enumerate(data_loader, 0):
        inputs, expected_outputs = data
        outputs = net(inputs)
        
        magnitude_expected = torch.norm(expected_outputs.type(torch.FloatTensor)).item()

        loss = net.loss(outputs, expected_outputs)
        sqrt_mse = torch.sqrt(loss)

        index = 0
        for error_multiplier in np.arange(0.1, 0.51, 0.05):
            error_multiplier_threshold = math.trunc(100*error_multiplier)/100
            acceptable_loss_upper_bound = magnitude_expected*error_multiplier_threshold
            if sqrt_mse.item() <= acceptable_loss_upper_bound:
                correct_instances[index] += 1
            index += 1
            
        total_instances += 1
            
        running_loss += loss.item()

    average_loss = math.sqrt(running_loss)/total_instances
    print("Average Loss: " + str(average_loss))
    
    percent_accuracy = []
    for i in range(0, 9):
        percent_accuracy.append((100*correct_instances[i]/total_instances))

    return percent_accuracy

file = open("trending.json")
data = json.load(file)
file.close()

testing_data = vectorizer.getTestingDataAsTorchVector(data)
testing_loader = testing_data_loader(testing_data)

size_in = 9
size_hidden = 256
size_out = 4

Model = tikTokVideoNN.Net(size_in, size_hidden, size_out)
Model.load("neuralNet.pth")

percent_accuracies = test(Model, testing_loader)
print(percent_accuracies)

error_to_acc_dict = dict()
index = 0
for error in np.arange(0.1, 0.51, 0.05):
    error = math.trunc(100*error)/100
    error_to_acc_dict[error] = percent_accuracies[index]
    index += 1

with open('error_threshold_to_testing_acc.pickle', 'wb') as handle:
    pickle.dump(error_to_acc_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)

