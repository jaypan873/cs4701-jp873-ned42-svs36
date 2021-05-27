import json
import numpy as np
import os
import math
import pickle

import vectorizer

import torch
import torch.nn as nn
from torch.nn import init
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, SubsetRandomSampler

class TikTokDataset(Dataset):
    
    def __init__(self, data):
        self.videoPosts = []
        self.interaction = []

        for video_properties, interaction in data:
            self.videoPosts.append(video_properties)
            self.interaction.append(interaction)

        self.videoPosts = torch.stack(self.videoPosts)
        self.interaction = torch.stack(self.interaction)

        self.numInstances = len(self.videoPosts)

    def __len__(self):
        return self.numInstances

    def __getitem__(self, index):
        return self.videoPosts[index], self.interaction[index]

class Net(nn.Module):
    def __init__(self, size_in, size_hidden, size_out):
        super().__init__()
        self.linLayer1 = nn.Linear(size_in, size_hidden)
        self.linLayer2 = nn.Linear(size_hidden, size_hidden)
        self.linLayer3 = nn.Linear(size_hidden, size_hidden)
        self.linLayer4 = nn.Linear(size_hidden, size_hidden)
        self.linLayer5 = nn.Linear(size_hidden, size_out)
        self.dropLayer = nn.Dropout(0.25)
        self.activationFunction = nn.ReLU()
        self.lossFunction = nn.MSELoss()

    def forward(self, v):
        x = self.linLayer1(v)
        x = self.dropLayer(x)
        x = self.linLayer2(x)
        x = self.dropLayer(x)
        x = self.linLayer3(x)
        x = self.dropLayer(x)
        x = self.linLayer4(x)
        x = self.dropLayer(x)
        x = self.linLayer5(x)
        return x

    def loss(self, out, gt):
        return self.lossFunction(out, gt)

    def load(self, path):
        self.load_state_dict(torch.load(path))

    def save(self, path):
        torch.save(self.state_dict(), path)

def train_model(epoch_count, net, train_data_loader, validation_data_loader):
    optimizer = optim.Adam(net.parameters(), lr = 0.001)
    training_acc_by_error = []
    percent_accuracies = []
    for epoch in range(epoch_count):
        print("Epoch " + str(epoch) + " out of " + str(epoch_count))
        # training_acc_by_error = one_training_epoch(net, train_data_loader, optimizer)
        percent_accuracies.append(one_training_epoch(net, train_data_loader, optimizer))
        validation(net, validation_data_loader, optimizer)

    acc_by_epoch_dict = dict()
    for epoch in range(epoch_count):
        acc_by_epoch_dict[epoch+1] = percent_accuracies[epoch]

    with open('training_acc_by_epoch.pickle', 'wb') as handle:
        pickle.dump(acc_by_epoch_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)

    """
    error_to_acc_dict = dict()
    index = 0
    for error in np.arange(0.1, 0.51, 0.05):
        error = math.trunc(100*error)/100
        error_to_acc_dict[error] = training_acc_by_error[index]
        index += 1

    with open('error_threshold_to_acc.pickle', 'wb') as handle:
        pickle.dump(error_to_acc_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)
    """
    
    print("Training Complete")

def one_training_epoch(net, data_loader, optimizer):
    net.train()
    running_loss = 0.0
    # correct_instances = [0 for i in range(9)]
    correct_instances = 0
    total_instances = 0
    error_multiplier = 0.5
    
    for i, data in enumerate(data_loader, 0):
        inputs, expected_outputs = data
        optimizer.zero_grad()
        outputs = net(inputs)
        loss = net.loss(outputs, expected_outputs)
        sqrt_mse = torch.sqrt(loss)
        loss.backward()
        optimizer.step()

        magnitude_expected = torch.norm(expected_outputs.type(torch.FloatTensor)).item()
        acceptable_loss_upper_bound = magnitude_expected*error_multiplier
        if sqrt_mse.item() <= acceptable_loss_upper_bound:
            correct_instances += 1
        
        index = 0
    
        """
        for error_multiplier in np.arange(0.1, 0.51, 0.05):
            error_multiplier = math.trunc(100*error_multiplier)/100
            acceptable_loss_upper_bound = magnitude_expected*error_multiplier
            
            if sqrt_mse.item() <= acceptable_loss_upper_bound:
                correct_instances[index] += 1
            index += 1
        """
                
        total_instances += 1
            
        running_loss += loss.item()
    
    print("Epoch completed")

    """
    training_accs = []
    for i in range(0, 9):
        prediction_accuracy = 100*correct_instances[i]/total_instances
        print("Prediction Accuracy - " + str(i) + ": " + str(prediction_accuracy))
        training_accs.append(prediction_accuracy)
    """
        
    print("Running Loss: " + str(math.sqrt(running_loss)))
    # return training_accs
    print("Percent Accuracy: " + str(100*correct_instances/total_instances))
    return (100*correct_instances/total_instances)

def validation(net, data_loader, optimizer):
    net.eval()
    running_loss = 0.0
    correct_instances = 0
    total_instances = 0
    error_multiplier = 0.5

    for i, data in enumerate(data_loader, 0):
        inputs, expected_outputs = data
        outputs = net(inputs)
        
        magnitude_expected = torch.norm(expected_outputs.type(torch.FloatTensor)).item()
        acceptable_loss_upper_bound = magnitude_expected*error_multiplier

        loss = net.loss(outputs, expected_outputs)
        sqrt_mse = torch.sqrt(loss)
        
        if sqrt_mse.item() <= acceptable_loss_upper_bound:
            correct_instances += 1
        total_instances += 1
            
        running_loss += loss.item()

    running_loss /= total_instances
    percent_accuracy = 100*correct_instances/total_instances

    print("Average Loss: " + str(running_loss))
    print("Percent Accuracy: " + str(percent_accuracy))

    return percent_accuracy

def data_loaders(training_data, validation_data, batch_size=1):

    dataset = TikTokDataset(training_data + validation_data)

    training_indices = [i for i in range(len(training_data))]
    validation_indices = [i for i in range(len(training_data), len(training_data) + len(validation_data))]

    training_shuffler = SubsetRandomSampler(training_indices)
    training_data_loader = DataLoader(dataset, batch_size=batch_size, sampler=training_shuffler)
    
    validation_shuffler = SubsetRandomSampler(validation_indices)
    validation_data_loader = DataLoader(dataset, batch_size=batch_size, sampler=validation_shuffler)

    return training_data_loader, validation_data_loader
         
def main():
    file = open("trending.json")
    data = json.load(file)
    file.close()

    training_data, validation_data = vectorizer.getTrainingAndValidationDataAsTorchTuples(data)

    training_loader, validation_loader = data_loaders(training_data, validation_data)

    size_in = 9 # Combining all features from caption, author, audio
    size_hidden = 256
    size_out = 4

    num_epochs = 25

    Model = Net(size_in, size_hidden, size_out).to("cpu")
    training_acc = train_model(num_epochs, Model, training_loader, validation_loader)
    Model.save("neuralNet.pth")

if __name__ == "__main__":
    main()
