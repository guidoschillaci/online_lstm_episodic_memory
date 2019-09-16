import pickle
import os

def convert_pickle_from_python3_to_python2():
    iterations = 5

    main_path = 'results/'

    for iter in range(iterations):
        directory = main_path + 'exp_0_iter_'+ str(iter) + '/'
        parameters = pickle.load(open(directory + 'parameters.pkl', 'rb'))

        pickle.dump(parameters, open(os.path.join('directory', 'parameters.pkl'), 'wb'), protocol=2)  # protcolo2 for compatibility with python2

if __name__ == "__main__":
    convert_pickle_from_python3_to_python2()