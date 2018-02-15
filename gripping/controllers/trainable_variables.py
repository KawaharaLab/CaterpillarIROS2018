import numpy as np
import pickle
import os


np.seterr(all="raise")


class TrainableVariables:
    def __init__(self):
        self.__trainble_vars = {}

    def variable_names(self) -> list:
        return self.__trainble_vars.keys()

    def append_varaible(self, name: str, var: np.array):
        assert name not in self.__trainble_vars.keys()
        self.__trainble_vars[name] = var

    def update_variable(self, name: str, var: np.array):
        assert name in self.__trainble_vars.keys() and self.__trainble_vars[name].shape == var.shape
        self.__trainble_vars[name] = var

    def get_variable(self, name: str) -> np.array:
        assert name in self.__trainble_vars.keys()
        return self.__trainble_vars[name]

    def save(self, save_file: str):
        with open(save_file, 'wb') as f:
            pickle.dump(self.__trainble_vars, f)

    def restore(self, data_file: str):
        """
            Compare restored variables and variables in __trainble_vars
        """
        assert os.path.exists(data_file)
        with open(data_file, 'rb') as f:
            restored_vars = pickle.load(f)

        assert type(restored_vars) is dict
        for k, v in self.__trainble_vars.items():
            if k not in restored_vars.keys():
                raise Exception("variable {} not found in specified file".format(k))
            if restored_vars[k].shape != v.shape:
                raise Exception("shape of variable {} doesn't match".format(k))
            self.__trainble_vars[k] = restored_vars[k]
