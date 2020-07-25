import numpy as np
import pickle

class Util:
    def saveObj(obj : ... , filename : str):

        '''
        This function takes as inputs an object and a filename name. Saves the provided
        object in the Temp directory.

        Parameters:
        obj (...) : The dictionary object to save.
        filename (str) : The file name for the dictionary object to save.
        '''

        with open('Temp/{}.pickle'.format(filename), 'wb') as file:
            pickle.dump(obj, file, protocol = pickle.HIGHEST_PROTOCOL)

    def loadObj(filename : str):

        '''
        This function takes as input a filename name. Loads the object from file and
        returns it.

        Parameters:
        filename (str) : The file name for the object to load.

        Returns:
        obj (...) : The object to return

        '''

        with open('Temp/{}.pickle'.format(filename), 'rb') as file:
            obj = pickle.load(file)
        return obj
