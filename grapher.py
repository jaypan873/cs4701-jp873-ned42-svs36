import math
import pickle
import matplotlib.pyplot as plt
import numpy as np

with open('error_threshold_to_acc.pickle', 'rb') as handle:
    training_acc_dict = pickle.load(handle)

with open('error_threshold_to_testing_acc.pickle', 'rb') as handle:
    testing_acc_dict = pickle.load(handle)

error_thresholds = []
training_accuracies = []
testing_accuracies = []
for error in np.arange(0.1, 0.51, 0.05):
    error_threshold = math.trunc(100*error)/100
    error_thresholds.append(100*error_threshold)
    training_accuracies.append(training_acc_dict[error_threshold])
    testing_accuracies.append(testing_acc_dict[error_threshold])

fig, (ax1, ax2) = plt.subplots(1, 2)
fig.suptitle('Training and Testing Accuracies vs. Error Threshold for \"Correct Instances\"')
fig.text(0.5, 0.02, 'Error Thresholds (%)', ha='center')
fig.text(0.02, 0.5, 'Prediction Accuracies (%)', va='center', rotation='vertical')

ax1.plot(error_thresholds, training_accuracies)
ax2.plot(error_thresholds, testing_accuracies)
plt.show()

