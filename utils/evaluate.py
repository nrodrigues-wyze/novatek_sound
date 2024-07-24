# Copyright 2020 Wyze Labs Inc. All Rights Reserved.
# ==============================================================================
import numpy as np
from utils.pytorch_utils import forward
from sklearn import metrics

def calculate_accuracy(y_true, y_score):
    n = y_true.shape[0]
    accuracy = np.sum(np.argmax(y_true, axis = -1) == np.argmax(y_score, axis = -1)) / n
    return accuracy


class Evaluator(object):
    def __init__(self, model, classes_num):
        self.model = model
        self.classes_num = classes_num

    def evaluate(self, data_loader):
        # Forward
        output_dict = forward(
            model = self.model,
            generator = data_loader,
            return_target = True)
        #import pdb; pdb.set_trace()
        #output_dict = {'clipwise_output': output[0]}
        clipwise_output = output_dict['clipwise_output']  # (audios_num, classes_num)
        target = output_dict['target']  # (audios_num, classes_num)

        cm = metrics.confusion_matrix(np.argmax(target, axis = -1), np.argmax(clipwise_output, axis = -1),
                                      labels = None)
        accuracy = calculate_accuracy(target, clipwise_output)

        statistics = {'accuracy': accuracy, 'cm': cm}

        return statistics




