import math
import pickle
import matplotlib.pyplot as plt
import numpy as np

with open('training_acc_by_epoch.pickle', 'rb') as handle:
    epoch_acc_dict = pickle.load(handle)

epoch_count = 25

epoch_nums = []
training_accuracies = []
for epoch in range(epoch_count):
    epoch_nums.append(epoch + 1)
    training_accuracies.append(epoch_acc_dict[epoch+1])

plt.plot(epoch_nums, training_accuracies)
plt.title("Training Accuracies After Each Epoch")
plt.xlabel("Epoch Number")
plt.ylabel("Training Accuracy (%)")
plt.show()
