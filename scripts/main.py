import load_datasets
from parameters import Parameters, MemUpdateStrategy
import numpy as np
import model
import time
import os
import sys


from doepy import build, read_write # pip install doepy - it may require also diversipy

import tensorflow as tf


#GPU_FRACTION = 1.0

#if tf.__version__ < "1.8.0":
#    config = tf.ConfigProto()
#    config.gpu_options.per_process_gpu_memory_fraction = GPU_FRACTION
#    session = tf.Session(config=config)
#else:
#    config = tf.compat.v1.ConfigProto()
#    config.gpu_options.per_process_gpu_memory_fraction = GPU_FRACTION
#    session = tf.compat.v1.Session(config=config)

def get_doe_mem_strategy_string_to_float(mem_upd_strategy_string):
    if mem_upd_strategy_string == 'High_LP':
        return 0.0
    elif mem_upd_strategy_string == 'Low_LP':
        return 1.0
    elif mem_upd_strategy_string == 'Random':
        return 2.0
    print ('error main.get_doe_mem_strategy_string_to_float: wrong mem_upd_strategy_string')
    sys.exit(0)

def get_doe_mem_strategy_float_to_string( mem_upd_strategy_float):
    print ('mem_upd_strategy_float ', mem_upd_strategy_float)
    if mem_upd_strategy_float == 0.0 :
        return 'High_LP'
    elif mem_upd_strategy_float == 1.0:
        return 'Low_LP'
    elif mem_upd_strategy_float == 2.0:
        return 'Random'
    print ('error main.get_doe_mem_strategy_float_to_string: wrong mem_upd_strategy_float')
    sys.exit(0)


if __name__ == "__main__":

    do_no_memory_experiment = True
    experiment_repetitions = 10
    days_in_win = 2
    if not os.path.isfile('results/design_of_experiments.csv'):

        # consider other methods than full factiorial, if having too many parameters
        # https://doepy.readthedocs.io/en/latest/
        doe = build.full_fact(
            {
             'days_in_window' : [days_in_win], #[1, 7],
             'memory_size': [1000], # was [500],# fix this to day_size * x?
             'memory_update_probability': [0.05], #[0.0001, 0.001],
            # make sure that the following has same orderas memupdatestrategy Enum (in parameters.py). TODO: make this better!
            # 'memory_update_strategy': [ MemUpdateStrategy.RANDOM]
            #'memory_update_strategy': [  MemUpdateStrategy.HIGH_LEARNING_PROGRESS, MemUpdateStrategy.LOW_LEARNING_PROGRESS, MemUpdateStrategy.RANDOM]
             'memory_update_strategy': [get_doe_mem_strategy_string_to_float('High_LP'),
                                        get_doe_mem_strategy_string_to_float('Low_LP'),
                                        get_doe_mem_strategy_string_to_float('Random')]
                #  'memory_update_strategy': [MemUpdateStrategy.LOW_LEARNING_PROGRESS, MemUpdateStrategy.RANDOM]

                #     #            it was [HIGH_LEARNING_PROGRESS, MemUpdateStrategy.LOW_LEARNING_PROGRESS  MemUpdateStrategy.RANDOM ]

            }
        )

        # write down to a file the experiments
        read_write.write_csv(doe,filename='results/design_of_experiments.csv')
    else:
        data_in = read_write.read_variables_csv('results/design_of_experiments.csv')
        doe = build.full_fact(data_in)


    #debug
    #print (doe.shape[0]) # number of experiments
    #print (doe.loc[:, :]) # print all the experiments
    #print (doe.loc[0, 'days_in_window']) # how to access a parameter

    # first experiment without memory
    if do_no_memory_experiment:

        models = []
        exp = 'nomemory'
        print('************************************************')
        print('************************************************')
        print('************************************************')
        print('************************************************')
        # create parameters object
        paramet = Parameters()
        #paramet.set('days_in_window', 2)
        paramet.set('days_in_window', days_in_win)
        #paramet.set('memory_update_probability', doe.loc[exp, 'memory_update_probability'] )
        #paramet.set('memory_update_strategy', doe.loc[exp, 'memory_update_strategy'] )
        paramet.set('memory_size', 0)

        # perform 5 repetitions of the same experiment
        for repeat in range(experiment_repetitions):
            model_name = 'exp_' + (exp) + '_iter_' + str(repeat)
            paramet.set('experiment_repetition', repeat)
            paramet.set('directory', model_name)
            # print (os.path.isdir('../../results/' + paramet.get('directory') ))
            if not os.path.isdir('results/' + paramet.get('directory')):

                print('************************************************')
                print('************************************************')

                # load the datasets
                train_datasets, test_datasets = load_datasets.Loader(model_name=model_name,
                                                                     param=paramet).get_train_test_datasets()

                print('Results folder: ' + paramet.get('directory'))

                # set random seed
                ## this should change, when running multiple tests and statistics!
                # np.random.seed(42)
                ## reset the seed
                np.random.seed(int(time.time()))

                # models.append(model.Model(paramet))
                models = model.Model(paramet)
                len_train_ds = len(train_datasets)
                #test = test_datasets  # [d] # all datasets
                for d in range(len_train_ds):
                #for d in range(1):
                    train = train_datasets[d]
                    greenhouse_index = train_datasets[d]['greenhouse_index']

                    # do online learning
                    # models[-1].online_fit_on(train, test, greenhouse_index)
                    # models[-1].save( paramet.get('directory') )
                    models.online_fit_on(train, test_datasets, greenhouse_index)
                    models.save(paramet.get('directory'))
                    print('Finished GH Dataset ' + str(greenhouse_index) + ' of exp ' + (exp) + ' repetition ' + str(
                        repeat))

                print(
                    'finished repetition ' + str(repeat) + ' of exp ' + (exp) + ' exp name ' + paramet.get('directory'))

                # cleaning up memory....
                # clear tensorflow session
                print('Clearing TF session')
                if tf.__version__ < "1.8.0":
                    tf.reset_default_graph()
                else:
                    tf.compat.v1.reset_default_graph()
                print('TF session cleared. Freeing memory')
                del train_datasets
                del test_datasets
                del greenhouse_index
                del models
                print('Memory freed')

    print('Running ' + str(doe.shape[0]) + ' tests, each repeated ' + str(experiment_repetitions) + ' times')
    print('doe '+ doe)
    # run every experiment defined by the parameters set in doe (design of experiment) object
    for exp in range(doe.shape[0]):
        print ('************************************************')
        print ('************************************************')
        print ('************************************************')
        print ('************************************************')
        # create parameters object
        paramet = Parameters()
        paramet.set('days_in_window', doe.loc[exp, 'days_in_window'] )
        paramet.set('memory_size', doe.loc[exp, 'memory_size'] )
        paramet.set('memory_update_probability', doe.loc[exp, 'memory_update_probability'] )
        paramet.set('memory_update_strategy',  get_doe_mem_strategy_float_to_string(doe.loc[exp, 'memory_update_strategy']) )

        # perform N repetitions of the same experiment
        for repeat in range(experiment_repetitions):
            model_name = 'exp_' + str(exp) + '_iter_' + str(repeat)
            paramet.set('experiment_repetition', repeat)
            paramet.set('directory', model_name)
            #print (os.path.isdir('../../results/' + paramet.get('directory') ))
            if not os.path.isdir('results/' + paramet.get('directory') ):

                print ('************************************************')
                print ('************************************************')
                
                # load the datasets
                train_datasets, test_datasets = load_datasets.Loader(model_name=model_name, param=paramet).get_train_test_datasets()
    
                print ('Results folder: ' + paramet.get('directory'))
    
                # set random seed
                ## this should change, when running multiple tests and statistics!
                #np.random.seed(42)
                ## reset the seed
                np.random.seed(int(time.time()))
    
                #models.append(model.Model(paramet))
                models = model.Model(paramet)
                len_train_ds = len(train_datasets)
                for d in range(len_train_ds):
                    train = train_datasets[d]
                    test = test_datasets#[d] # all datasets
                    greenhouse_index = train_datasets[d]['greenhouse_index']
    
                    # do online learning
                    #models[-1].online_fit_on(train, test, greenhouse_index)
                    #models[-1].save( paramet.get('directory') )
                    models.online_fit_on(train, test, greenhouse_index)
                    models.save( paramet.get('directory') )
                    print ('Finished GH Dataset '+ str(greenhouse_index) + ' of exp ' + str(exp) + ' repetition ' + str(repeat))

                print ('finished repetition '+str(repeat) + ' of exp '+ str(exp) + ' exp name ' + paramet.get('directory'))

                # cleaning up memory....
                # clear tensorflow session
                print ('Clearing TF session')
                if tf.__version__ < "1.8.0":
                    tf.reset_default_graph()
                else:
                    tf.compat.v1.reset_default_graph()
                print('TF session cleared. Freeing memory')
                del train_datasets
                del test_datasets
                del greenhouse_index
                del models
                print ('Memory freed')

        print ('finished experiment '+ str(exp))

