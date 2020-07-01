

import os
import numpy as np
import pandas as pd
import time
from copy import deepcopy
import parameters

from sklearn.preprocessing import MinMaxScaler
from skimage.util.shape import view_as_windows


class Loader:

    def __init__(self, model_name, param):

        self.directory = 'results/' + model_name
        if not os.path.exists( self.directory ):
            os.makedirs( self.directory )

        param.set('directory', self.directory)
        self.parameters = deepcopy(param)
        print ('days_in_window ', self.parameters.get('days_in_window'))
        print ('Verbosity level '+ str(self.parameters.get('verbosity_level')))

        print('Loading original datasets...')
        self.load_original_datasets()

        print('Normalising original datasets...')
        self.normalise_datasets()

        print ('Creating train and test datasets')
        self.create_train_and_test_datasets()

    # load the datasets of the greenhouses. This need to be windowed, later
    def load_original_datasets(self):
        # import data
        df_col = pd.read_csv('Data/zineg_11_16/col_11-16.clean2.csv', parse_dates=['Timestamp'])
        if self.parameters.get('verbosity_level') > 1:
            print('df_col ')
            print(df_col[100:110])

        df_ref = pd.read_csv( 'Data/zineg_11_16/ref_11-16.clean2.csv', parse_dates=['Timestamp'] )
        if self.parameters.get('verbosity_level')>1:
            print('df_ref ')
            print(df_ref[100:110])

        df_neber1 = pd.read_csv('Data/zineg_11_16/neber-18.test2a.clean.csv', parse_dates=['Timestamp'])
        if self.parameters.get('verbosity_level') > 1:
            print('df_neber1 ')
            print(df_ref[100:110])

        df_neber2 = pd.read_csv( 'Data/zineg_11_16/neber-18.test2b.clean.csv', parse_dates=['Timestamp'] )
        if self.parameters.get('verbosity_level')>1:
            print('df_neber2 ')
            print(df_neber2[100:110])


        self.unwindowed_dataset_names = ['GH1', 'GH2', 'GH3a', 'GH3b']
        # put everything into a dictionary
        self.unwindowed_datasets = {
            self.unwindowed_dataset_names[0] : df_col[['Timestamp', 'co2', 'temp', 'rh', 'leaft1', 'leaft2', 'trans', 'photo', 'rad']].copy(),
            self.unwindowed_dataset_names[1] : df_ref[['Timestamp', 'co2', 'temp', 'rh', 'leaft1', 'leaft2', 'trans', 'photo', 'rad']].copy(),
            self.unwindowed_dataset_names[2]: df_neber1[['Timestamp', 'co2', 'temp', 'rh', 'leaft1', 'leaft2', 'trans', 'photo', 'rad']].copy(),
            self.unwindowed_dataset_names[3]: df_neber2[['Timestamp', 'co2', 'temp', 'rh', 'leaft1', 'leaft2', 'trans', 'photo', 'rad']].copy()
        }

        self.sensor_names = ['temp', 'rh', 'co2', 'rad', 'leaft1', 'leaft2', 'trans', 'photo']
        # reorder columns
        for name in self.unwindowed_dataset_names:
            self.unwindowed_datasets[name] = self.unwindowed_datasets[name][['Timestamp'] + self.sensor_names]

        if self.parameters.get('verbosity_level')>0:
            self.unwindowed_datasets[ self.unwindowed_dataset_names[0]].head()

    ## check this! Is it correct that we fit the scalers with data only from GH1?
    def normalise_datasets(self):

        # Scaler for the outputs, to make it easier to de-scale afterwards
        self.scaler_trans = MinMaxScaler(copy=True, feature_range=(
        self.parameters.get('normalization_limits')[0], self.parameters.get('normalization_limits')[1]))
        self.scaler_trans.fit(self.unwindowed_datasets['GH1']['trans'].values.reshape(-1, 1))

        self.scaler_photo = MinMaxScaler(copy=True, feature_range=(
        self.parameters.get('normalization_limits')[0], self.parameters.get('normalization_limits')[1]))
        self.scaler_photo.fit(self.unwindowed_datasets['GH1']['photo'].values.reshape(-1, 1))

        # Scaler for the complete data tables
        self.scaler = MinMaxScaler(copy=True, feature_range=(
        self.parameters.get('normalization_limits')[0], self.parameters.get('normalization_limits')[1]))
        self.scaler.fit(self.unwindowed_datasets['GH1'][self.sensor_names])

        # normalise
        for name in self.unwindowed_dataset_names:
            self.unwindowed_datasets[name].loc[:, self.sensor_names] = self.scaler.transform(self.unwindowed_datasets[name][self.sensor_names])
            if self.parameters.get('verbosity_level') > 0:
                print ('Normalised '+ str(name))
                print (self.unwindowed_datasets[name][100:110])




    def create_train_and_test_datasets(self):
        # Train + Val data go from pd.Timestamp( '01-jan-2011 00:00') to pd.Timestamp('01-jan-2015 00:00')
        # Online fit GH2 goes from pd.Timestamp( '01-jan-2015 00:00') to pd.Timestamp('15-jan-2016 00:00')
        # Test GH2 goes from pd.Timestamp( '15-jan-2016 00:00') to pd.Timestamp('01-jan-2017 00:00')

        # train datasets
        self.tt_datasets_names = ['train_GH1', 'online_GH2', 'test_GH2', 'test_GH3']
        self.tt_datasets_index = [1, 2, 2, 3]
        self.tt_datasets = {
            self.tt_datasets_names[0] : self.unwindowed_datasets['GH1']  [ (self.unwindowed_datasets['GH1']['Timestamp'] < pd.Timestamp('01-jan-2015 00:00'))],
            self.tt_datasets_names[1] : self.unwindowed_datasets['GH2']  [(self.unwindowed_datasets['GH2']['Timestamp'] >= pd.Timestamp('01-jan-2015 00:00')) &
                                                                          (self.unwindowed_datasets['GH2']['Timestamp'] <= pd.Timestamp('15-jan-2016 00:00'))],
            self.tt_datasets_names[2] : self.unwindowed_datasets['GH2']  [ (self.unwindowed_datasets['GH2']['Timestamp'] >= pd.Timestamp('01-jan-2016 00:00'))],
            self.tt_datasets_names[3] : self.unwindowed_datasets['GH3b']
        }

        for name in self.tt_datasets_names:
            print ('shape '+ name+ ' '+ str(self.tt_datasets[name].shape))

        # prepare for creating windows
        ## win_size: how big each window-sample is, from those who enter the LSTM
        ## len(sensors): how many variables there are, inputs + outputs together
        self.window_shape = (self.parameters.get_window_size(), len(self.sensor_names))

        # split into train and test
        ##### set always the same seed, for selecting the same test indexes, in case you want to do stats on multiple runs
        ##### remember to reset it back to pseudo-random after this (np.random.seed(time.time()))
        np.random.seed(55)

        self.train_datasets = []
        self.test_datasets  = []
        for i in range(len(self.tt_datasets_names)):
            name = self.tt_datasets_names[i]
            greenhouse_index = self.tt_datasets_index[i]
            print ('extracting windows from ' + name)
            windows = self.extract_windows(self.tt_datasets[name], self.parameters.get('inp_sensors'),
                                      self.parameters.get('out_sensors'), ['Timestamp'],
                                      self.parameters.get_window_size(), self.parameters.get('step'),
                                      verbose=True)

            #'name'     : name,
            #'windows_x': windows[0],
            #'windows_y': windows[1],
            #'Timestamp': windows[2]

            len_ds = len(windows[0])
            test_indexes = np.random.choice(range(len_ds), int(len_ds / self.parameters.get('mse_test_dataset_fraction')))

            #print ('test idx' + str(test_indexes))

            train_indexes = np.ones(len_ds, np.bool)
            train_indexes[test_indexes] = 0

            self.train_datasets.append(
                {
                    'name'      : name,
                    'greenhouse_index' : greenhouse_index,
                    'window_inputs' : windows[0][train_indexes],
                    'window_outputs' : windows[1][train_indexes],
                    'Timestamp' : windows[2][train_indexes]
                }
            )

            self.test_datasets.append(
                {
                    'name'      : name,
                    'greenhouse_index': greenhouse_index,
                    'window_inputs' : windows[0][test_indexes],
                    'window_outputs' : windows[1][test_indexes],
                    'Timestamp' : windows[2][test_indexes]
                }
            )

            if self.parameters.get('verbosity_level') > 0:
                print ('dataset '+name + ' train size '+ str(len(windows[0][train_indexes])) +
                       ' test size '+ str(len(windows[0][test_indexes])) )

        ## reset the seed
        np.random.seed(int(time.time()))


    def extract_windows(self, dataset, input_columns, output_columns, tracking_columns, windows_size, step, verbose=True ):

        '''
        Returns 3D numpy arrays with series of chunks from dataset, each of size windows_size

        Returns three arrays, with data from input_columns, output_columns and tracking_columns

        For output_columns it returns only the last value in the window

        tracking_columns is intended to make it possible to trace back the position of the individual windows,
        because some might be deleted if they contain NaN-values
        '''

        assert type( input_columns ) == list
        assert type( output_columns ) == list
        assert type( tracking_columns ) == list
        assert type( dataset ) == pd.DataFrame
        assert all( [ c in dataset.columns for c in input_columns ] )
        assert all( [ c in dataset.columns for c in output_columns ] )
        assert all( [ c in dataset.columns for c in tracking_columns ] )

        columns_to_use = input_columns + output_columns + tracking_columns
        columns_size = len( columns_to_use ) # how many variables there are, inputs + outputs together
        window_shape = ( windows_size, columns_size ) # window_size, dataset_columns_size

        dataset = dataset[ columns_to_use ] # pd.DataFrame

        if verbose:
            print( '*'*50 )

        # convert the interesting columns into a numpy array
        original_dataset = dataset.values

        if verbose:
            print ('Shape of original dataset: ', original_dataset.shape )

        # Create the windows!
        windowed_dataset = view_as_windows( original_dataset, window_shape, step=step )
        if verbose:
            print ('Shape of windowed dataset :',  windowed_dataset.shape )

        # for some reason, view_as_windows adds a dimension. Reshape to remove this.
        windowed_dataset = windowed_dataset.reshape( len( windowed_dataset ),
                                                     window_shape[0],
                                                     window_shape[1] )
        if verbose:
            print ('Shape of windowed_reshaped windowed dataset: ', windowed_dataset.shape )

        # these lists will contain the input/output samples for the neural network
        input_samples = []
        output_samples = []
        tracking_samples = []
        input_indexes = [ dataset.columns.get_loc( c ) for c in input_columns ]
        output_indexes = [ dataset.columns.get_loc( c ) for c in output_columns ]
        tracking_indexes = [ dataset.columns.get_loc( c ) for c in tracking_columns ]

        # drop samples containing nan
        for sample in windowed_dataset:
            # keep the current sample (shape: windows_size, columns_size) only if it does not have any NaN
            if ~np.isnan( sample[ :, input_indexes+output_indexes ].astype(np.float64) ).any():
                # No-ARX: inps -> last point:
                input_samples.append( sample[ 0:windows_size, input_indexes ] )
                # takes only the last element in the window! -> LSTM to predict a single value!
                output_samples.append( sample[ -1, output_indexes ] )
                tracking_samples.append( sample[ -1, tracking_indexes ] )
                # ARX: window -> last point:
                #            input_samples.append( sample[ 0:windows_size-1, input_indexes ] )
                #            # takes only the last element in the window! -> LSTM to predict a single value!
                #            output_samples.append( sample[ -1, output_indexes ] )
                #            tracking_samples.append( sample[ -1, tracking_indexes ] )

        # change from list to numpy array
        input_samples = np.asarray( input_samples ).astype(np.float64)
        output_samples = np.asarray( output_samples ).astype(np.float64)
        tracking_samples = np.asarray( tracking_samples )

        assert np.isnan( input_samples ).any() == False
        assert np.isnan( output_samples ).any() == False

        if verbose:
            print ('Input samples shape: ', input_samples.shape )
            print ('Output samples shape: ', output_samples.shape )
            print ('Tracking samples shape: ', tracking_samples.shape )
            print( '*'*50 )

        return input_samples, output_samples, tracking_samples

    def get_train_test_datasets(self):
        return self.train_datasets, self.test_datasets
