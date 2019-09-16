import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from parameters import MemUpdateStrategy
import pickle

if __name__ == "__main__":

    iterations = 5

    main_path = 'results/'

    for iter in range(iterations):
        directory = main_path + 'exp_0_iter_'+ str(iter) + '/'


        parameters = pickle.load(open(directory + 'parameters.pkl', 'rb'))

        mse = np.load(directory + 'mse.npy')
        output_var = np.load(directory + 'output_variances.npy')
        input_var = np.load(directory + 'input_variances.npy')
        mem_index = np.load(directory + 'memory_index_proportions.npy')
        mem_label = np.load(directory + 'mse_memory_label.npy')
        switch_time = np.load(directory + 'switch_time.npy')
        #learning_progress = np.load(directory + 'learning_progress.npy', allow_pickle=True)

        #print (learning_progress)
        #print (str((learning_progress.shape)))
        #learning_progress = np.array(learning_progress, dtype=float)
        #print(str(type(learning_progress)))
        #print (str(np.asarray(learning_progress[:,:])))
        #print (str(np.asarray(learning_progress).shape))

        # learning progress
        #fig, ax = plt.subplots(1, 1, figsize=(7, 7))
        # x = np.arange(len(learning_progress))
        # print learning_progress.shape()
        # plt.scatter(x, y1, y2, y3, labels=['GH1 Data', 'GH2 Data', 'GH3 Data'])
        #plt.imshow(learning_progress.T, aspect='auto')
        #plt.colorbar()
        #plt.savefig(directory + 'learning_progress.png')
        #plt.show()

        fig, ax = plt.subplots( 1, 1, figsize=(10,7) )
        ax.plot(np.arange(len(mse)),mse)
        ax.axvline(x=switch_time[0],color='r', linestyle='dashed')
        ax.axvline(x=switch_time[1],color='b', linestyle='dashed')
        plt.ylim(0, 0.5)
        plt.title('Mean Squared Error - MemStrategy: '+ str(parameters['memory_update_strategy']) + ' iteration '+ str(iter))
        plt.savefig(directory + 'learning_progress.png')
        plt.show()

        fig, ax = plt.subplots( 1, 1, figsize=(10,7) )
        ax.plot(np.arange(len(input_var)),input_var)
        ax.axvline(x=switch_time[0],color='r', linestyle='dashed')
        ax.axvline(x=switch_time[1],color='b', linestyle='dashed')
        plt.title('Input variances - MemStrategy: ' )#+ str(parameters['memory_update_strategy']) )
        plt.savefig(directory + 'input_variances.png')
        #plt.show()

        fig, ax = plt.subplots( 1, 1, figsize=(10,7) )
        ax.plot(np.arange(len(output_var)),output_var)
        ax.axvline(x=switch_time[0],color='r', linestyle='dashed')
        ax.axvline(x=switch_time[1],color='b', linestyle='dashed')
        plt.title('Output variances - MemStrategy: ' )#+ str(parameters['memory_update_strategy']) )
        plt.savefig(directory + 'output_variances.png')
        #plt.show()


        fig, ax = plt.subplots( 1, 1, figsize=(10,7) )
        ax.plot(np.arange(len(mem_label)),mem_label)
        ax.axvline(x=switch_time[0],color='r', linestyle='dashed')
        ax.axvline(x=switch_time[1],color='b', linestyle='dashed')
        plt.title('Which greenhouse data is used - MemStrategy: ')# + str(parameters['memory_update_strategy']) )
        plt.savefig(directory + 'memory_label.png')
        #plt.show()


        # memory index proportion
        fig, ax = plt.subplots(1, 1, figsize=(7, 7))
        x = np.arange(len(mem_index))
        y1 = mem_index[:, 0]
        y2 = mem_index[:, 1]
        y3 = mem_index[:, 2]
        # Basic stacked area chart.
        plt.stackplot(x, y1, y2, y3, labels=['GH1 Data', 'GH2 Data', 'GH3 Data'])
        ax.xaxis.grid()  # vertical lines
        ax.axvline(x=switch_time[0], color='r', linestyle='dashed')
        ax.axvline(x=switch_time[1], color='b', linestyle='dashed')
        plt.legend(loc='upper left')
        plt.savefig(directory + 'memory_index_proportion.png')
        #plt.show()

