import numpy as np
from copy import deepcopy

class Logger:

    def __init__(self, param):
        self.parameters = param
        # contains the list of MSE calculation over time
        self.mse = []
        # contains the list of labels of the dataset used for the MSE calculation
        self.mse_memory_label = []
        # how many samples from which greenhouse in the memory?
        self.memory_index_proportions = []
        # every time there is a switch between greenhouses, store the current iteration index
        self.switch_time = []

        self.count_of_changed_memory_elements = []
        
        self.input_variances= []
        self.output_variances= []
        self.learning_progress = []

    def store_log(self, mse = [], gh_index = [], mem_idx_prop = [], input_var = [], output_var = [], count_of_changed_memory_elements = [], learning_progress = []):
        self.mse.append(mse)
        self.mse_memory_label.append(gh_index)
        self.memory_index_proportions.append(mem_idx_prop)
        self.input_variances.append(input_var)
        self.output_variances.append(output_var)
        self.count_of_changed_memory_elements.append(count_of_changed_memory_elements)
        self.learning_progress.append(deepcopy(learning_progress))
        #print (str(self.learning_progress))

    def switch_dataset(self):
        self.switch_time.append(len(self.mse))

    def get_iteration_count(self):
        return len(self.mse)