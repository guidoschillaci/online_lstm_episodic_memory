import numpy as np
import random
import os
from copy import deepcopy

from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense, Input, Flatten
from tensorflow.python.keras.layers import LSTM
from tensorflow.python.keras.layers import BatchNormalization
from tensorflow.python.keras.activations import sigmoid
from tensorflow.python.keras.activations import hard_sigmoid
from tensorflow.python.keras.callbacks import EarlyStopping
from tensorflow.python.keras.callbacks import TensorBoard
#from tensorflow.python.keras.utils import plot_model
from tensorflow.python.keras.utils.vis_utils import plot_model


import datetime
import logger
import memory
from parameters import Parameters, MemUpdateStrategy

class Model:


    def make_recurrent_nn(self, plot =False):
        print ('Model is a recurrent LSTM network')
        model = Sequential()
        # model.add( LSTM( int(time_series_length/2), input_shape=(time_series_length,features), unroll=True, return_sequences=True, dropout=0.0, recurrent_dropout=0.0 ) )
        model.add(LSTM(units=8, input_shape=(self.parameters.get_window_size(), len(self.parameters.get('inp_sensors'))), unroll=True,dropout=0.0, recurrent_dropout=0.0))
        model.add(BatchNormalization())
        model.add(Dense(16, activation=hard_sigmoid))
        model.add(BatchNormalization())
        model.add(Dense(8, activation=sigmoid))
        model.add(BatchNormalization())
        model.add(Dense(len(self.parameters.get('out_sensors'))))

        model.compile(loss=self.parameters.get('loss'), optimizer=self.parameters.get('optimizer'))
        model.summary()

        if plot:
            plot_model(model, to_file='model_plot.png', show_shapes=True, show_layer_names=True)
            print('Model plot saved')

        return model

    def make_mlp_nn(self, plot=False):
        print ('Model is a Multi-Layer Perceptron')
        model = Sequential()
        model.add( Input( shape=(self.parameters.get_window_size(), len(self.parameters.get('inp_sensors') ) ) ) )
        #model.add(
        #    LSTM(units=8, input_shape=(self.parameters.get_window_size(), len(self.parameters.get('inp_sensors'))),
        #         unroll=True, dropout=0.0, recurrent_dropout=0.0))
        model.add(Flatten())
        model.add(BatchNormalization())
        model.add(Dense(16, activation=hard_sigmoid))
        model.add(BatchNormalization())
        model.add(Dense(8, activation=sigmoid))
        model.add(BatchNormalization())
        model.add(Dense(len(self.parameters.get('out_sensors'))))

        model.compile(loss=self.parameters.get('loss'), optimizer=self.parameters.get('optimizer'))
        model.summary()

        if plot:
            plot_model(model, to_file='model_plot.png', show_shapes=True, show_layer_names=True)
            print('Model plot saved')

        return model

    def __init__(self, param):

        self.time_start = datetime.datetime.now()

        self.parameters = param
        self.model = []
        if param.get('model_type') == 'recurrent':
            self.model = self.make_recurrent_nn()
        else:
            self.model = self.make_mlp_nn()

        self.memory_size = self.parameters.get('memory_size')  # how many windows to keep in memory?
        self.prob_update = self.parameters.get('memory_update_probability') # probability of substituting an element of the memory with the current observations

        print ('Memory update strategy is ', self.parameters.get('memory_update_strategy'))
        self.memory = memory.Memory(param = self.parameters)
        self.logger = logger.Logger(param = self.parameters)

    def compute_mse(self, test_dataset):
        input = test_dataset['window_inputs']
        output = test_dataset['window_outputs']

        predictions = self.model.predict(input)
        mse = (np.linalg.norm(predictions - output) ** 2) / len(output)

        #print ('Current mse: ', mse)
        return mse

    def online_fit_on(self, train_dataset, test_dataset, gh_index):

        input_batch = []
        output_batch = []
      #  for i in range(2):
        for i in range(len(train_dataset['window_inputs'])):

            input_batch.append(train_dataset['window_inputs'][i])
            output_batch.append(train_dataset['window_outputs'][i])

            # update the memory with the current observations
            count_of_changed_memory_elements = self.memory.update(train_dataset['window_inputs'][i],
                               train_dataset['window_outputs'][i],
                               gh_index=gh_index)

            #print ('var ')
            #print (str(np.asarray(train_dataset['window_inputs']).shape))
            #print (str(np.asarray(train_dataset['window_outputs']).shape))

            # print 'memory size ', len( memory_input)
            # if the lists have reached the batch size, then fit the model
            if len(input_batch) == self.parameters.get('batch_size'):
                # fit the model with the current batch of observations and the memory!
                # create then temporary input and output tensors containig batch and memory
                full_input = []
                full_output = []
                if self.parameters.get('memory_size') > 0 :
                    full_input = np.vstack((np.asarray(input_batch), np.asarray(self.memory.input_variables)))
                    full_output = np.vstack((np.asarray(output_batch), np.asarray(self.memory.output_variables)))
                else:
                    full_input = deepcopy(np.asarray(input_batch))
                    full_output = deepcopy(np.asarray(output_batch))

                print('Processed ', i + 1, ' samples of ', len(train_dataset['window_inputs']))

                print('fitting with ', len(full_input), ' samples')
                self.model.fit(full_input, full_output, epochs=1,  # validation_split=0.25,
                                   # no validation data! ## check this!
                                   # validation_data=(online_valx,online_valy), # what about the validation data? Keep the one as in the offline test?
                                   batch_size=self.parameters.get('batch_size'))  # , callbacks=[ earlystop, tbCallBack ] )

                if self.parameters.get('memory_update_strategy') == MemUpdateStrategy.HIGH_LEARNING_PROGRESS.value or self.parameters.get('memory_update_strategy') == MemUpdateStrategy.LOW_LEARNING_PROGRESS.value:
                    print ('updating learning progress for each memory element')
                    self.memory.update_learning_progress(self.model)

                # print 'current memory output' # to not print the full windows, just print the output and check if things are slowly changing
                # print memory_input
                # print memory_output
                # restore batch arrays
                input_batch = []
                output_batch = []

            if i % (self.parameters.get('mse_calculation_step') * self.parameters.get('batch_size')) == 0:

                if self.parameters.get('memory_size') != 0:
                    print('Mem_upd_str: ', self.parameters.get('memory_update_strategy'), ' repet.nr.:',
                          self.parameters.get('experiment_repetition'))
                else:
                    print('No_memory exp, repet.nr.:', self.parameters.get('experiment_repetition'))

                mse_all =[]
                for td in range(len(test_dataset)): # compute mse for all the test datasets of each greenhouse
                    mse_all.append(self.compute_mse(test_dataset[td]))
                learn_progress = self.memory.get_learning_progress() # returns the current learning progress for each sample storedin the memory
                mem_idx_prop = self.memory.get_greenhouse_index_proportion()
                input_var, output_var = self.memory.get_variance()
                # compute the mse1
                self.logger.store_log(mse=mse_all, gh_index=gh_index, mem_idx_prop= mem_idx_prop, input_var = input_var,
                                      output_var = output_var,
                                      count_of_changed_memory_elements = count_of_changed_memory_elements, learning_progress=learn_progress)
                for td in range(len(test_dataset)):  # compute mse for all the test datasets of each greenhouse
                    print ('mse(gh'+str(td)+'):' +str(mse_all[td]) + ' idx_prop '+ str(mem_idx_prop))


        self.logger.switch_dataset() # store the len of the mse vector
        del input_batch
        del output_batch

    def save(self, directory):
        np.save(os.path.join(directory, 'mse'), self.logger.mse)
        np.save(os.path.join(directory, 'mse_memory_label'), self.logger.mse_memory_label)
        np.save(os.path.join(directory, 'memory_index_proportions'), self.logger.memory_index_proportions)
        np.save(os.path.join(directory, 'switch_time'), self.logger.switch_time)
        np.save(os.path.join(directory, 'input_variances'), self.logger.input_variances)
        np.save(os.path.join(directory, 'output_variances'), self.logger.output_variances)
        np.save(os.path.join(directory, 'learning_progress'), self.logger.learning_progress)
        self.parameters.save()

        # saving times
        self.time_end = datetime.datetime.now()
        time_delta = self.time_end - self.time_start
        time_data = [ self.time_start, self.time_end, time_delta]
        np.save(os.path.join(directory, 'time_data'), time_data)
        print ('data saved')