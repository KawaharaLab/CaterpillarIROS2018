from itertools import accumulate
import os
import numpy as np

from controllers.trainable_variables import TrainableVariables

np.seterr(all="raise")


class BaseActor:
    def __init__(self, actor_model_file=None):
        self.__trainble_variables = TrainableVariables()
        self.__infer_func = self._build_network()

        if actor_model_file is not None:
            self.__trainble_variables.restore(actor_model_file)

    def __str__(self):
        raise NotImplementedError   # Force actors to return their class name

    def new_trainable_variable(self, name: str, initial_value: np.array) -> np.array:
        # Append to trainable variables
        assert name not in self.__trainble_variables.variable_names()
        self.__trainble_variables.append_varaible(name, initial_value)
        return self.__trainble_variables.get_variable(name)

    def var(self, name: str) -> np.array:
        assert name in self.__trainble_variables.variable_names()
        return self.__trainble_variables.get_variable(name)

    def params_shapes(self) -> list:
        # List of shapes, ex. ((2,3), (4,5), ...)
        return [v.shape for v in self.current_params()]

    def params_num(self) -> int:
        # Amount of the all trainable variables
        return sum([list(accumulate(sh, lambda x, y: x * y))[-1] for sh in self.params_shapes()])

    def current_params(self) -> list:
        # Returns [np.array(params_0), np.array(params_1), ...]
        return [self.__trainble_variables.get_variable(v_name)
                for v_name in self.__trainble_variables.variable_names()]

    def set_params(self, params: np.array):
        # params: [params_00, params_01, ..., params_10, ...]
        shapes = self.params_shapes()
        splitted_params = np.split(params, np.cumsum([np.prod(s) for s in shapes])[:-1])
        reshaped_params = [np.reshape(s_params, shape)
                           for s_params, shape in zip(splitted_params, shapes)]
        for v_name, new_value in zip(self.__trainble_variables.variable_names(), reshaped_params):
            self.__trainble_variables.update_variable(v_name, new_value)

    def _build_network(self):
        """Define a function which receives a state vector and returns an action."""
        raise NotImplementedError

    def get_action(self, state: np.array) -> np.array:
        return self.__infer_func(state)

    def save(self, save_file_path, step=None):
        if step is not None:
            save_file_path = "{}_step{}".format(save_file_path, step)

        # Make sure that save_file_path can be created
        dir_ = os.path.dirname(save_file_path)
        if not os.path.exists(dir_):
            os.makedirs(dir_)

        self.__trainble_variables.save(save_file_path)
