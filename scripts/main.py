import load_datasets
from parameters import Parameters, MemUpdateStrategy
import numpy as np
import model
import time
import os

from doepy import build, read_write # pip install doepy - it may require also diversipy

import tensorflow as tf

if tf.__version__ < "1.8.0":
    config = tf.ConfigProto()
    config.gpu_options.per_process_gpu_memory_fraction = 0.75
    session = tf.Session(config=config)
else:
    config = tf.compat.v1.ConfigProto()
    config.gpu_options.per_process_gpu_memory_fraction = 0.75
    session = tf.compat.v1.Session(config=config)

if __name__ == "__main__":
    do_no_memory_experiment = True
    experiment_repetitions = 1

    if not os.path.isfile('results/design_of_experiments.csv'):

        # consider other methods than full factiorial, if having too many parameters
        # https://doepy.readthedocs.io/en/latest/
        doe = build.full_fact(
            {
             'days_in_window' : [1], #[1, 7],
             'memory_size': [200], # fix this to day_size * x?
             'memory_update_probability': [0.1], #[0.0001, 0.001],
            # make sure that the following has same orderas memupdatestrategy Enum (in parameters.py). TODO: make this better!
             'memory_update_strategy': [ MemUpdateStrategy.RANDOM, MemUpdateStrategy.HIGH_LEARNING_PROGRESS, MemUpdateStrategy.LOW_LEARNING_PROGRESS]
    #            it was [HIGH_LEARNING_PROGRESS, MemUpdateStrategy.LOW_LEARNING_PROGRESS  MemUpdateStrategy.RANDOM ]

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
        exp = doe.shape[0] + 1
        print('************************************************')
        print('************************************************')
        print('************************************************')
        print('************************************************')
        # create parameters object
        paramet = Parameters()
        paramet.set('days_in_window', 1)
        paramet.set('memory_size', 0)

        # perform 5 repetitions of the same experiment
        for repeat in range(experiment_repetitions):
            model_name = 'exp_' + str(exp) + '_iter_' + str(repeat)

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
                test = test_datasets  # [d] # all datasets
                for d in range(len_train_ds):
                    train = train_datasets[d]
                    greenhouse_index = train_datasets[d]['greenhouse_index']

                    # do online learning
                    # models[-1].online_fit_on(train, test, greenhouse_index)
                    # models[-1].save( paramet.get('directory') )
                    models.online_fit_on(train, test, greenhouse_index)
                    models.save(paramet.get('directory'))
                    print('Finished GH Dataset ' + str(greenhouse_index) + ' of exp ' + str(exp) + ' repetition ' + str(
                        repeat))

                print(
                    'finished repetition ' + str(repeat) + ' of exp ' + str(exp) + ' exp name ' + paramet.get('directory'))

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
        paramet.set('memory_update_strategy', doe.loc[exp, 'memory_update_strategy'] )

        # perform 5 repetitions of the same experiment
        for repeat in range(experiment_repetitions):
            model_name = 'exp_' + str(exp) + '_iter_' + str(repeat)

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

