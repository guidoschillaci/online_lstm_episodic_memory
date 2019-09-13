import numpy as np
import random
from parameters import Parameters, MemUpdateStrategy

class Memory:

    def __init__(self, param):
        self.parameters = param

        # compute the MSE every these steps
        self.mse_calculation_step = self.parameters.get('batch_size') * self.parameters.get('batchs_to_update_online')

        # content of the memory
        self.input_variables = []
        self.output_variables = []
        self.greenhouse_index = []
        self.sample_confidence_interval = []
        self.prediction_errors = [] # for each sample in the memory, store the prediction errors calculated at the last fits at times t and t-1
        self.learning_progress = [] # derivative of the prediction errors (for the moment, just simply pe(t) - pe(t-1)


    def update(self, input_window, output_window, gh_index):

        counter_of_changed_elements = 0
        # if the size of the stored samples has not reached the full size of the memory, then just append the samples
        if len(self.input_variables) < self.parameters.get('memory_size'):
            self.input_variables.append(input_window)
            self.output_variables.append(output_window)
            self.greenhouse_index.append(gh_index)
            self.prediction_errors.append([])
            self.learning_progress.append([])
        else:
            if self.parameters.get('memory_update_strategy') == MemUpdateStrategy.RANDOM.value:
                # iterate the memory and decide whether to assign the current sample to an element or not, with probability p
                #for i in range(len(self.input_variables)):
                i = random.randrange(0, len(self.input_variables))
                ran = random.random()
                if ran < self.parameters.get('memory_update_probability'):
                    self.input_variables[i] = input_window
                    self.output_variables[i] = output_window
                    self.greenhouse_index[i] = gh_index
                    self.prediction_errors[i] = []
                    self.learning_progress[i] = []
                        #counter_of_changed_elements = counter_of_changed_elements + 1
            elif self.parameters.get('memory_update_strategy') == MemUpdateStrategy.LOW_LEARNING_PROGRESS.value:
                # select the element with the highest or lowest learning progress (which of the two is the best?
                # and substitute it with the new sample - with probability p
                #for i in range(len(self.input_variables)):
                ran = random.random()
                if ran < self.parameters.get('memory_update_probability'):
                    index = self.learning_progress.index( np.min (self.learning_progress)) # gives high plasticity?
                    self.input_variables[index] = input_window
                    self.output_variables[index] = output_window
                    self.greenhouse_index[index] = gh_index
                    self.prediction_errors[index] = []
                    self.learning_progress[index] = []
                        #counter_of_changed_elements = counter_of_changed_elements +1
            elif self.parameters.get('memory_update_strategy') == MemUpdateStrategy.HIGH_LEARNING_PROGRESS.value:
                # select the element with the highest or lowest learning progress (which of the two is the best?
                # and substitute it with the new sample - with probability p
                #for i in range(len(self.input_variables)):
                ran = random.random()
                if ran < self.parameters.get('memory_update_probability'):
                    index = self.learning_progress.index( np.max (self.learning_progress)) # gives low plasticity?
                    self.input_variables[index] = input_window
                    self.output_variables[index] = output_window
                    self.greenhouse_index[index] = gh_index
                    self.prediction_errors[index] = []
                    self.learning_progress[index] = []
                        #counter_of_changed_elements = counter_of_changed_elements +1
            else:
                print ('Wrong parameter memory_update_strategy')
            counter_of_changed_elements = counter_of_changed_elements + 1
        return counter_of_changed_elements

    def get_greenhouse_index_proportion(self):
        gh1 = sum(x == 1 for x in self.greenhouse_index)
        gh2 = sum(x == 2 for x in self.greenhouse_index)
        gh3 = sum(x == 3 for x in self.greenhouse_index)
        return [gh1, gh2, gh3]

    def get_variance(self):
        input_var = np.var(self.input_variables)
        print ('input var  ' + str(input_var))
        output_var = np.var(self.output_variables)
        print ('output var  ' + str(output_var))
        return input_var, output_var

    def get_learning_progress(self):
        return self.learning_progress

    def update_learning_progress(self, model):
        predictions = model.predict( np.asarray(self.input_variables) )
        for i in range (len  (self.output_variables) ):
            prediction_error = (np.linalg.norm(predictions[i] - self.output_variables[i]) ** 2)
            self.prediction_errors[i].append(deepcopy(prediction_error))
            if len ( self.prediction_errors[i]) >= 2:
                self.learning_progress[i] = np.fabs(self.prediction_errors[i][-1] - self.prediction_errors[i][-2])