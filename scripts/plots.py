import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.dates as mdates
from parameters import MemUpdateStrategy
import pickle
import os
from sklearn.manifold import TSNE
from sklearn import decomposition
import time
import seaborn as sns
import load_datasets
import parameters

outvar_mean_all=[]
outvar_std_all=[]

def get_strategy_name(strategy_id):
    if strategy_id == 0:
        return 'Discard High LP'
    if strategy_id == 1:
        return 'Discard Low LP'
    if strategy_id == 2:
        return 'Discard Random'
    return 'Unknown'

def tsne(data):
    time_start = time.time()
    tsne = TSNE(n_components=3, verbose=1, perplexity=40, n_iter=300)
    data_ = data.reshape(len(data), len(data[0])*len(data[0,0]))
    print ('data_ shape', data.shape)
    tsne_results = tsne.fit_transform(data_)

    print('t-SNE done! Time elapsed: ', (time.time()-time_start), ' seconds')

    df_subset['tsne-2d-one'] = tsne_results[:,0]
    df_subset['tsne-2d-two'] = tsne_results[:,1]
    plt.figure(figsize=(16,10))
    sns.scatterplot(
        x="tsne-2d-one", y="tsne-2d-two",
        hue="y",
        palette=sns.color_palette("hls", 10),
        data=df_subset,
        legend="full",
        alpha=0.3
    )
    plt.savefig('t-sne.png')

def plot_pca(data1, data2, data3, data4, out1, out2, out3, out4):
    time_start = time.time()
    pca = decomposition.PCA(n_components=4)
    print('fit...')
    print ('shape ', data1.shape)
    print ('shape out ', out1.shape)
    shape1_ =data1.shape
    shape2_ =data2.shape
    shape3_ =data3.shape
    shape4_ =data4.shape
    data1_reshaped = np.hstack( (np.reshape(data1, (len(data1)* shape1_[1], shape1_[2])) ,  np.asarray(out1)) )
    data2_reshaped = np.hstack( (np.reshape(data2, (len(data2)* shape2_[1], shape2_[2])) ,  np.asarray(out2)) )
    data3_reshaped = np.hstack( (np.reshape(data3, (len(data3)* shape3_[1], shape3_[2])) ,  np.asarray(out3)) )
    data4_reshaped = np.hstack( (np.reshape(data4, (len(data4)* shape4_[1], shape4_[2])) ,  np.asarray(out4)) )
    print ('shape reshaped', data1_reshaped.shape)
    pca.fit(np.vstack((data1_reshaped, data2_reshaped, data3_reshaped, data4_reshaped)))
    print('transform...')
    X1 = pca.transform(data1_reshaped[::1])
    X2 = pca.transform(data2_reshaped[::1])
    X3 = pca.transform(data3_reshaped[::1])
    X4 = pca.transform(data4_reshaped[::1])


    #github.com/mirandal-gh/pca
    print ('explained variance ratio ' , pca.explained_variance_ratio_)
    # pca.explain_variance_ratio_.cumsum()

    print ('shape x1 ', np.asarray(X1).shape)
    area =4
    # Plot
    fig, ax = plt.subplots(1, 1, figsize=(10, 10))
    plt.scatter(X1[:,0], X1[:,1], s=area, c='g', alpha = 0.2, label='gh1')
    plt.scatter(X2[:,0], X2[:,1], s=area, c='red', alpha = 0.2, label='gh2(2015)')
    plt.scatter(X3[:,0], X3[:,1], s=area, c='purple', alpha = 0.2, label='gh2(>2016)')
    plt.scatter(X4[:,0], X4[:,1], s=area, c='b', alpha = 0.2, label='gh3')
    plt.legend(loc='upper right')
    plt.title('PCA - First two components')
    plt.xlabel('PC1. EVR: %1.2f' % pca.explained_variance_ratio_[0])
    plt.ylabel('PC2. EVR: %1.2f' % pca.explained_variance_ratio_[1])
    plt.savefig('pca.png')
    plt.show()


    fig = plt.figure()
    ax = Axes3D(fig)

    ax.scatter(X1[:,0], X1[:,1], X1[:,2], s=area, c='g', alpha = 0.2, label='gh1')
    ax.scatter(X2[:,0], X2[:,1], X2[:,2], s=area, c='red', alpha = 0.2, label='gh2(2015)')
    ax.scatter(X3[:,0], X3[:,1], X3[:,2], s=area, c='purple', alpha = 0.2, label='gh2(>2016)')
    ax.scatter(X4[:,0], X4[:,1], X4[:,2], s=area, c='b', alpha = 0.2, label='gh3')
    plt.legend(loc='upper right')
    plt.title('PCA - First three components')
    ax.set_xlabel('PC1. EVR: %1.2f' % pca.explained_variance_ratio_[0])
    ax.set_ylabel('PC2. EVR: %1.2f' % pca.explained_variance_ratio_[1])
    ax.set_zlabel('PC3. EVR: %1.2f' % pca.explained_variance_ratio_[2])
    plt.savefig('pca3D.png')
    plt.show()


def do_plots(directory, greenhouses = 4, iterations =1, no_memory=False, show_intermediate=False):

    mse_all = []
    input_var_all = []
    output_var_all = []
    switch_time = []
    for gh in range(greenhouses):
        mse_all.append([])
        #input_var_all.append([])
        #output_var_all.append([])

    for iterat in range(iterations):
        subdirectory = directory + 'iter_'+str(iterat)+'/'
        parameters = pickle.load(open(subdirectory + 'parameters.pkl', 'rb'))
        if os.path.isfile(subdirectory + 'mse.npy'):
            mse=np.load(subdirectory + 'mse.npy')
            for gh in range(greenhouses):
                mse_all[gh].append(mse[:,gh])
        else:
            print ('File '+subdirectory + 'mse.npy'+ 'does not exists!')

        if os.path.isfile(subdirectory + 'output_variances.npy'):
            output_var = np.load(subdirectory + 'output_variances.npy')
           # print ('shape read ', np.asarray(output_var).shape)
            #for gh in range(greenhouses):
            output_var_all.append(output_var[:])
        else:
            print('File ' + subdirectory + 'output_variances.npy' + 'does not exists!')

        if os.path.isfile(subdirectory + 'input_variances.npy'):
            input_var =np.load(subdirectory + 'input_variances.npy')
           # for exp in range(experiments):
           #     input_var_all[exp].append(input_var[:,exp])
        else:
            print('File ' + subdirectory + 'input_variances.npy' + 'does not exists!')

        if os.path.isfile(subdirectory + 'memory_index_proportions.npy'):
            mem_index = np.load(subdirectory + 'memory_index_proportions.npy')
        else:
            print('File ' + subdirectory + 'memory_index_proportions.npy' + 'does not exists!')
        if os.path.isfile(subdirectory + 'mse_memory_label.npy'):
            mem_label = np.load(subdirectory + 'mse_memory_label.npy')
        else:
            print('File ' + subdirectory + 'mse_memory_label.npy' + 'does not exists!')
        if os.path.isfile(subdirectory + 'switch_time.npy'):
            switch_time = np.load(subdirectory + 'switch_time.npy')
        else:
            print('File ' + subdirectory + 'switch_time.npy' + 'does not exists!')
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
        length = len(mse)
        fig, ax = plt.subplots(1, 1, figsize=(10, 7))
        ax.plot(np.arange(len(mse)), mse[:, 0], marker='.', color='green', label='gh1')
        ax.plot(np.arange(len(mse)), mse[:, 1], marker='.', color='red', label='gh2')
        ax.plot(np.arange(len(mse)), mse[:, 2], marker='.', color='purple', label='gh3a')
        ax.plot(np.arange(len(mse)), mse[:, 3], marker='.', color='blue', label='gh3b')
        plt.legend(loc='upper left')
        ax.axvline(x=switch_time[0], color='r', linestyle='dashed')
        ax.axvline(x=switch_time[1], color='purple', linestyle='dashed')
        ax.axvline(x=switch_time[2], color='blue', linestyle='dashed')
        plt.ylim(0, 0.5)
      #  plt.xlim(0, 200)
        plt.title(
            'Mean Squared Error - ' + ('No Memory ' if no_memory else ('MemStrategy: ' + get_strategy_name(int(parameters['memory_update_strategy'])) ) ) + ' iteration ' + str(
                iter))
        plt.savefig(subdirectory + 'learning_progress.png')
        if show_intermediate:
            plt.show()
        plt.close()

        fig, ax = plt.subplots(1, 1, figsize=(10, 7))
        ax.plot(np.arange(len(input_var)), input_var)
        ax.axvline(x=switch_time[0], color='r', linestyle='dashed')
        ax.axvline(x=switch_time[1], color='purple', linestyle='dashed')
        ax.axvline(x=switch_time[2], color='blue', linestyle='dashed')
        plt.title(
            'Input variances - ' + ('No Memory ' if no_memory else ('MemStrategy: ' + get_strategy_name(int(parameters['memory_update_strategy'])) ) ) + ' iteration ' + str(
                iter))
      #  plt.xlim(0, 200)
        plt.savefig(subdirectory + 'input_variances.png')
        if show_intermediate:
            plt.show()
        plt.close()

        fig, ax = plt.subplots(1, 1, figsize=(10, 7))
        ax.plot(np.arange(len(output_var)), output_var)
        ax.axvline(x=switch_time[0], color='r', linestyle='dashed')
        ax.axvline(x=switch_time[1], color='purple', linestyle='dashed')
        ax.axvline(x=switch_time[2], color='b', linestyle='dashed')
      #  plt.xlim(0, 200)
        plt.title(
            'Output variances - ' + ('No Memory ' if no_memory else ('MemStrategy: ' + get_strategy_name(int(parameters['memory_update_strategy'])) ) ) + ' iteration ' + str(
                iter))
        plt.savefig(subdirectory + 'output_variances.png')
        if show_intermediate:
            plt.show()
        plt.close()

        fig, ax = plt.subplots(1, 1, figsize=(10, 7))
        ax.plot(np.arange(len(mem_label)), mem_label)
        ax.axvline(x=switch_time[0], color='r', linestyle='dashed')
        ax.axvline(x=switch_time[1], color='purple', linestyle='dashed')
        ax.axvline(x=switch_time[2], color='b', linestyle='dashed')
       # plt.xlim(0, 200)
        plt.title('Which greenhouse data is used - ' + ('No Memory ' if no_memory else ('MemStrategy: ' + get_strategy_name(int(parameters['memory_update_strategy'])) ) ))
        plt.savefig(subdirectory + 'memory_label.png')
        if show_intermediate:
            plt.show()
        plt.close()

        # memory index proportion
        fig, ax = plt.subplots(1, 1, figsize=(7, 7))
        x = np.arange(len(mem_index))
        y1 = mem_index[:, 0]
        y2 = mem_index[:, 1]
        y3 = mem_index[:, 2]
        # Basic stacked area chart.
        pal = ['g', 'r', 'b']
        plt.stackplot(x, y1, y2, y3, colors=pal, labels=['GH1 Data', 'GH2 Data', 'GH3 Data'])
        ax.xaxis.grid()  # vertical lines
        ax.axvline(x=switch_time[0], color='r', linestyle='dashed')
        ax.axvline(x=switch_time[1], color='purple', linestyle='dashed')
        ax.axvline(x=switch_time[2], color='b', linestyle='dashed')
       # plt.xlim(0, 200)
        plt.legend(loc='upper left')
        plt.title(
            'Memory Index - ' + ('No Memory ' if no_memory else ('MemStrategy: ' + get_strategy_name(int(parameters['memory_update_strategy'])) ) ) )
        plt.savefig(subdirectory + 'memory_index_proportion.png')
        if show_intermediate:
            plt.show()

        plt.close()  # close figures
    plt.clf() # clear figures
    # do the plots with error bars
    #print ('shape mse ', np.asarray(mse_all).shape)
    #print ('shape mse ', np.asarray(mse_all[0]).shape)
    mse_mean = []
    mse_std_dev = []
    for gh in range(greenhouses):
        mse_mean.append(np.mean(mse_all[gh], axis=0))
        mse_std_dev.append(np.std(mse_all[gh], axis=0))
    #print (mse_mean)
    #print ('len(mse_all) ', length)
    #print ('mse_mean[0] ' , mse_mean[0])

    fig, ax = plt.subplots(1, 1, figsize=(10, 7))
    plt.rcParams.update({'font.size': 14})
    plt.tick_params(labelsize=14)

    ax.plot(np.arange(length), mse_mean[0], color='green', label='gh1')
    ax.plot(np.arange(length), mse_mean[1], color='red', label='gh2(2015)')
    ax.plot(np.arange(length), mse_mean[2], color='purple', label='gh2(>2016)')
    ax.plot(np.arange(length), mse_mean[3], color='blue', label='gh3')

    plt.fill_between(np.arange(length), mse_mean[0] - mse_std_dev[0], mse_mean[0] + mse_std_dev[0], color='green', alpha=0.3)
    plt.fill_between(np.arange(length), mse_mean[1] - mse_std_dev[1], mse_mean[1] + mse_std_dev[1], color='red', alpha=0.3)
    plt.fill_between(np.arange(length), mse_mean[2] - mse_std_dev[2], mse_mean[2] + mse_std_dev[2], color='purple', alpha=0.3)
    plt.fill_between(np.arange(length), mse_mean[3] - mse_std_dev[3], mse_mean[3] + mse_std_dev[3], color='b', alpha=0.3)

    plt.legend(loc='upper left')
    ax.axvline(x=switch_time[0], color='r', linestyle='dashed')
    ax.axvline(x=switch_time[1], color='purple', linestyle='dashed')
    ax.axvline(x=switch_time[2], color='b', linestyle='dashed')
    plt.ylim( (pow(10,-3),pow(10,1)) )
    plt.yscale('log')
    plt.rcParams['grid.alpha'] = 0.4
    plt.grid(True, which='both')
    #  plt.xlim(0, 200)
    plt.title(
        'Mean Squared Error - ' + ('No Memory ' if no_memory else (
                    'MemStrategy: ' + get_strategy_name(int(parameters['memory_update_strategy'])))) )
    plt.savefig(directory + 'learning_progress.png')
    plt.show()


    if not no_memory:
        #print ('shape ' , np.asarray(output_var_all).shape)
        outvar_mean_all.append(np.mean(output_var_all, axis=0))
        outvar_std_all.append(np.std(output_var_all, axis=0))


    '''
    fig, ax = plt.subplots(1, 1, figsize=(10, 7))
    ax.plot(np.arange(len(input_var)), input_var)
    ax.axvline(x=switch_time[0], color='r', linestyle='dashed')
    ax.axvline(x=switch_time[1], color='b', linestyle='dashed')
    plt.title(
        'Input variances - ' + ('No Memory ' if no_memory else (
                    'MemStrategy: ' + str(int(parameters['memory_update_strategy'])))) + ' iteration ' + str(
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
        'Output variances - ' + ('No Memory ' if no_memory else (
                    'MemStrategy: ' + str(int(parameters['memory_update_strategy'])))) + ' iteration ' + str(
            iter))
    plt.savefig(directory + 'output_variances.png')
    plt.show()
    '''
    plt.close()
    return switch_time




def do_plots_singletest(directory, greenhouses = 4, iterations =1, no_memory=False, show_intermediate=False):

    mse_all = []
    input_var_all = []
    output_var_all = []
    switch_time = []
    for gh in range(greenhouses):
        mse_all.append([])
        #input_var_all.append([])
        #output_var_all.append([])

    for iterat in range(iterations):
        subdirectory = directory + 'iter_'+str(iterat)+'/'
        parameters = pickle.load(open(subdirectory + 'parameters.pkl', 'rb'))
        if os.path.isfile(subdirectory + 'mse.npy'):
            mse=np.load(subdirectory + 'mse.npy')
            for gh in range(greenhouses):
                mse_all[gh].append(mse[:,gh])
        else:
            print ('File '+subdirectory + 'mse.npy'+ 'does not exists!')

        if os.path.isfile(subdirectory + 'output_variances.npy'):
            output_var = np.load(subdirectory + 'output_variances.npy')
           # print ('shape read ', np.asarray(output_var).shape)
            #for gh in range(greenhouses):
            output_var_all.append(output_var[:])
        else:
            print('File ' + subdirectory + 'output_variances.npy' + 'does not exists!')

        if os.path.isfile(subdirectory + 'input_variances.npy'):
            input_var =np.load(subdirectory + 'input_variances.npy')
           # for exp in range(experiments):
           #     input_var_all[exp].append(input_var[:,exp])
        else:
            print('File ' + subdirectory + 'input_variances.npy' + 'does not exists!')

        if os.path.isfile(subdirectory + 'memory_index_proportions.npy'):
            mem_index = np.load(subdirectory + 'memory_index_proportions.npy')
        else:
            print('File ' + subdirectory + 'memory_index_proportions.npy' + 'does not exists!')
        if os.path.isfile(subdirectory + 'mse_memory_label.npy'):
            mem_label = np.load(subdirectory + 'mse_memory_label.npy')
        else:
            print('File ' + subdirectory + 'mse_memory_label.npy' + 'does not exists!')
        if os.path.isfile(subdirectory + 'switch_time.npy'):
            switch_time = np.load(subdirectory + 'switch_time.npy')
        else:
            print('File ' + subdirectory + 'switch_time.npy' + 'does not exists!')
        # learning_progress = np.load(directory + 'learning_progress.npy', allow_pickle=True)

        

    # do the plots with error bars
    mse_mean = []
    mse_std_dev = []
    for gh in range(greenhouses):
        mse_mean.append(np.mean(mse_all[gh], axis=0))
        mse_std_dev.append(np.std(mse_all[gh], axis=0))
    #print (mse_mean)
    #print ('len(mse_all) ', length)
    #print ('mse_mean[0] ' , mse_mean[0])

    fig, ax = plt.subplots(1, 1, figsize=(10, 7))
    ax.plot(np.arange(length), mse_mean[0], color='green', label='gh1')
    ax.plot(np.arange(length), mse_mean[1], color='red', label='gh2(2015)')
    ax.plot(np.arange(length), mse_mean[2], color='purple', label='gh2(>2016)')
    ax.plot(np.arange(length), mse_mean[3], color='blue', label='gh3')

    plt.fill_between(np.arange(length), mse_mean[0] - mse_std_dev[0], mse_mean[0] + mse_std_dev[0], color='green', alpha=0.3)
    plt.fill_between(np.arange(length), mse_mean[1] - mse_std_dev[1], mse_mean[1] + mse_std_dev[1], color='red', alpha=0.3)
    plt.fill_between(np.arange(length), mse_mean[2] - mse_std_dev[2], mse_mean[2] + mse_std_dev[2], color='purple', alpha=0.3)
    plt.fill_between(np.arange(length), mse_mean[3] - mse_std_dev[3], mse_mean[3] + mse_std_dev[3], color='b', alpha=0.3)

    plt.legend(loc='upper left')
    ax.axvline(x=switch_time[0], color='r', linestyle='dashed')
    ax.axvline(x=switch_time[1], color='purple', linestyle='dashed')
    ax.axvline(x=switch_time[2], color='b', linestyle='dashed')
    plt.ylim(0, 0.5)
    #  plt.xlim(0, 200)
    plt.title(
        'Mean Squared Error - ' + ('No Memory ' if no_memory else (
                    'MemStrategy: ' + get_strategy_name(int(parameters['memory_update_strategy'])))) )
    plt.savefig(directory + 'learning_progress.png')
    plt.show()


    plt.close()
    return switch_time







def plot_var(directory, switch, experiments=3 ):

    colors = ['r', 'b', 'g']
    fig, ax = plt.subplots(1, 1, figsize=(10, 7))
    for i in range(experiments):
        ax.plot(np.arange(len(outvar_mean_all[i])), outvar_mean_all[i], color=colors[i], label= get_strategy_name(i))
        plt.fill_between(np.arange(len(outvar_mean_all[i])), outvar_mean_all[i] - outvar_std_all[i], outvar_mean_all[i] + outvar_std_all[i], color=colors[i],
                         alpha=0.3)
        plt.rcParams.update({'font.size': 14})
        plt.tick_params(labelsize=14)
        plt.legend(loc='upper left')
        ax.axvline(x=switch[0], color='r', linestyle='dashed')
        ax.axvline(x=switch[1], color='purple', linestyle='dashed')
        ax.axvline(x=switch[2], color='b', linestyle='dashed')
        plt.ylim(0, 0.08)
    #  plt.xlim(0, 200)
    plt.title('Variance of the output features of the memory samples')
    plt.savefig(directory + 'output_variance.png')
    plt.show()


if __name__ == "__main__":

    do_mse_plots = True
    do_pca = False

    if do_pca:
        param = parameters.Parameters()
        param.set('days_in_window', 1)
        param.set('day_size', 1)
        param.set('step', 1)

        train_datasets, test_datasets = load_datasets.Loader(model_name='results', param=param).get_train_test_datasets()

        plot_pca(np.asarray(train_datasets[0]['window_inputs']), \
                 np.asarray(train_datasets[1]['window_inputs']), \
                 np.asarray(train_datasets[2]['window_inputs']), \
                 np.asarray(train_datasets[3]['window_inputs']), \
                 np.asarray(train_datasets[0]['window_outputs']), \
                 np.asarray(train_datasets[1]['window_outputs']), \
                 np.asarray(train_datasets[2]['window_outputs']), \
                 np.asarray(train_datasets[3]['window_outputs']))
        '''
        param = parameters.Parameters()
        param.set('days_in_window', 1)
    
        train_datasets, test_datasets = load_datasets.Loader(model_name='results',
                                                             param=param).get_train_test_datasets()
    
        data = train_datasets[0]['window_inputs']
        print ('data shape ', np.asarray(data.shape))
        tsne(np.asarray(data))
    
        print ('tsne done')
        '''



    if do_mse_plots:
        #main_path = 'results_good_5days/'
        #main_path = 'results/'
        main_path = 'results_good_2_days/'
        is_nomemory_exp_available = True
        iterations = 5

        # have you carried out experiments also without using memory? (memory size == 0)
        if is_nomemory_exp_available:
            directory = main_path + 'exp_nomemory_'#iter_' + str(iter) + '/'
            do_plots(directory, iterations=iterations,  no_memory=is_nomemory_exp_available)

        # plot all the rest of the experiments
        experiments = 3
        switch_t = []
        for exp in range(experiments):
            directory = main_path + 'exp_'+str(exp)+'_'  # iter_' + str(iter) + '/'
            switch_t = do_plots(directory, iterations=iterations, no_memory=False)
        print (np.asarray(switch_t).shape)
        plot_var(main_path, switch = switch_t, experiments=3)

        print ('All the plots are saved')
