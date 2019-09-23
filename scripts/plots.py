import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from parameters import MemUpdateStrategy
import pickle
import os


def do_plots(directory, no_memory=False):
    parameters = pickle.load(open(directory + 'parameters.pkl', 'rb'))

    if os.path.isfile(directory + 'mse.npy'):
        mse = np.load(directory + 'mse.npy')
    else:
        print ('File '+directory + 'mse.npy'+ 'does not exists!')
    if os.path.isfile(directory + 'output_variances.npy'):
        output_var = np.load(directory + 'output_variances.npy')
    else:
        print('File ' + directory + 'output_variances.npy' + 'does not exists!')
    if os.path.isfile(directory + 'input_variances.npy'):
        input_var = np.load(directory + 'input_variances.npy')
    else:
        print('File ' + directory + 'input_variances.npy' + 'does not exists!')
    if os.path.isfile(directory + 'memory_index_proportions.npy'):
        mem_index = np.load(directory + 'memory_index_proportions.npy')
    else:
        print('File ' + directory + 'memory_index_proportions.npy' + 'does not exists!')
    if os.path.isfile(directory + 'mse_memory_label.npy'):
        mem_label = np.load(directory + 'mse_memory_label.npy')
    else:
        print('File ' + directory + 'mse_memory_label.npy' + 'does not exists!')
    if os.path.isfile(directory + 'switch_time.npy'):
        switch_time = np.load(directory + 'switch_time.npy')
    else:
        print('File ' + directory + 'switch_time.npy' + 'does not exists!')
    # learning_progress = np.load(directory + 'learning_progress.npy', allow_pickle=True)

    # print (learning_progress)
    # print (str((learning_progress.shape)))
    # learning_progress = np.array(learning_progress, dtype=float)
    # print(str(type(learning_progress)))
    # print (str(np.asarray(learning_progress[:,:])))
    # print (str(np.asarray(learning_progress).shape))

    # learning progress
    # fig, ax = plt.subplots(1, 1, figsize=(7, 7))
    # x = np.arange(len(learning_progress))
    # print learning_progress.shape()
    # plt.scatter(x, y1, y2, y3, labels=['GH1 Data', 'GH2 Data', 'GH3 Data'])
    # plt.imshow(learning_progress.T, aspect='auto')
    # plt.colorbar()
    # plt.savefig(directory + 'learning_progress.png')
    # plt.show()

    fig, ax = plt.subplots(1, 1, figsize=(10, 7))
    ax.plot(np.arange(len(mse)), mse[:, 0], color='r', label='gh1')
    ax.plot(np.arange(len(mse)), mse[:, 1], color='b', label='gh2')
    ax.plot(np.arange(len(mse)), mse[:, 2], color='g', label='gh3a')
    ax.plot(np.arange(len(mse)), mse[:, 3], color='y', label='gh3b')
    plt.legend(loc='upper left')
    ax.axvline(x=switch_time[0], color='r', linestyle='dashed')
    ax.axvline(x=switch_time[1], color='b', linestyle='dashed')
    plt.ylim(0, 0.5)
  #  plt.xlim(0, 200)
    plt.title(
        'Mean Squared Error - ' + ('No Memory ' if no_memory else ('MemStrategy: ' + str(int(parameters['memory_update_strategy'])) ) ) + ' iteration ' + str(
            iter))
    plt.savefig(directory + 'learning_progress.png')
    plt.show()

    fig, ax = plt.subplots(1, 1, figsize=(10, 7))
    ax.plot(np.arange(len(input_var)), input_var)
    ax.axvline(x=switch_time[0], color='r', linestyle='dashed')
    ax.axvline(x=switch_time[1], color='b', linestyle='dashed')
    plt.title(
        'Input variances - ' + ('No Memory ' if no_memory else ('MemStrategy: ' + str(int(parameters['memory_update_strategy'])) ) ) + ' iteration ' + str(
            iter))
  #  plt.xlim(0, 200)
    plt.savefig(directory + 'input_variances.png')
    plt.show()

    fig, ax = plt.subplots(1, 1, figsize=(10, 7))
    ax.plot(np.arange(len(output_var)), output_var)
    ax.axvline(x=switch_time[0], color='r', linestyle='dashed')
    ax.axvline(x=switch_time[1], color='b', linestyle='dashed')
  #  plt.xlim(0, 200)
    plt.title(
        'Output variances - ' + ('No Memory ' if no_memory else ('MemStrategy: ' + str(int(parameters['memory_update_strategy'])) ) ) + ' iteration ' + str(
            iter))
    plt.savefig(directory + 'output_variances.png')
    plt.show()

    fig, ax = plt.subplots(1, 1, figsize=(10, 7))
    ax.plot(np.arange(len(mem_label)), mem_label)
    ax.axvline(x=switch_time[0], color='r', linestyle='dashed')
    ax.axvline(x=switch_time[1], color='b', linestyle='dashed')
   # plt.xlim(0, 200)
    plt.title('Which greenhouse data is used - ' + ('No Memory ' if no_memory else ('MemStrategy: ' + str(int(parameters['memory_update_strategy'])) ) ) + ' iteration ' + str(
            iter))
    plt.savefig(directory + 'memory_label.png')
    plt.show()

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
   # plt.xlim(0, 200)
    plt.legend(loc='upper left')
    plt.title(
        'Memory Index - ' + ('No Memory ' if no_memory else ('MemStrategy: ' + str(int(parameters['memory_update_strategy'])) ) ) + ' iteration ' + str(
            iter))
    plt.savefig(directory + 'memory_index_proportion.png')

    plt.show()

if __name__ == "__main__":

    main_path = 'results/'
    is_nomemory_exp_available = True
    iterations = 1

    # have you carried out experiments also without using memory? (memory size == 0)
    if is_nomemory_exp_available:
        for iter in range(iterations):
            directory = main_path + 'exp_nomemory_iter_' + str(iter) + '/'
            if os.path.isdir(directory):
                do_plots(directory, no_memory=is_nomemory_exp_available)
            else:
                print ('Directory: ' + directory + ' does not exist')
                print ('Maybe you have not carried out an experiment without memory?')

    # plot all the rest of the experiments
    experiments = 3
    for exp in range(experiments):
        for iter in range(iterations):
            directory = main_path + 'exp_'+str(exp)+'_iter_'+ str(iter) + '/'
            if os.path.isdir(directory):
                do_plots(directory)
            else:
                print ('Directory: ' + directory + ' does not exist')


    print ('All the plots are saved')
