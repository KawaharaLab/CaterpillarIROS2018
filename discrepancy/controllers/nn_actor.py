from .config import config
from .base_actor import BaseActor
import numpy as np

np.seterr(all="raise")


FEEDBACK_MAX = 1.


class NeuralNetworkActor(BaseActor):
    def __str__(self):
        return "SimpleNeuralNetworkActor"

    def _build_network(self):
        """
            \dot{\phi}_i = \omega + f_i(F) \cos \phi_i
            Define a function which returns [action_0, action_1, ...]
        """
        hidden_layer_num = 10
        self.new_trainable_variable("w_sin_0", np.zeros((config.params["somites"], hidden_layer_num), dtype=np.float32))
        self.new_trainable_variable("b_sin_0", np.zeros(hidden_layer_num, dtype=np.float32))
        self.new_trainable_variable("w_sin_1", np.zeros((hidden_layer_num, config.params["oscillators"]), dtype=np.float32))
        self.new_trainable_variable("b_sin_1", np.zeros(config.params["oscillators"], dtype=np.float32))

        self.new_trainable_variable("w_cos_0", np.zeros((config.params["somites"], hidden_layer_num), dtype=np.float32))
        self.new_trainable_variable("b_cos_0", np.zeros(hidden_layer_num, dtype=np.float32))
        self.new_trainable_variable("w_cos_1", np.zeros((hidden_layer_num, config.params["oscillators"]), dtype=np.float32))
        self.new_trainable_variable("b_cos_1", np.zeros(config.params["oscillators"], dtype=np.float32))

        def action_infer(state: np.array) -> np.array:
            """
                state: [f_0, f_1, ..., phi_0, phi_1, ...]
            """
            forces = state[:config.params["somites"]]
            phis = state[config.params["somites"]:]

            # Correction (since somite2 is comparatively smaller than end somites)
            # forces[config.params["somites"] // 2] *= 4

            f_sin, f_cos = self._calc_fs(forces)
            return f_sin * np.sin(phis) + f_cos * np.cos(phis)

        return action_infer

    def _calc_fs(self, forces: np.array) -> tuple:
        # return: f_sin, f_cos
        assert forces.shape == (config.params["somites"],)

        def sigmoid(x: np.array) -> np.array:
            return 1. / (1. + np.exp(-x))

        def tanh(x: np.array) -> np.array:
            return (1. - np.exp(-2 * x)) / (1. + np.exp(-2 * x))

        # sin
        h_s_0 = sigmoid(np.matmul(forces, self.var("w_sin_0")) + self.var("b_sin_0"))
        z_s_1 = np.matmul(h_s_0, self.var("w_sin_1")) + self.var("b_sin_1")
        f_sin = tanh(z_s_1) * FEEDBACK_MAX   # - FEEDBACK_MAX ~ FEEDBACK_MAX
        # cos
        h_c_0 = sigmoid(np.matmul(forces, self.var("w_cos_0")) + self.var("b_cos_0"))
        z_c_1 = np.matmul(h_c_0, self.var("w_cos_1")) + self.var("b_cos_1")
        f_cos = tanh(z_c_1) * FEEDBACK_MAX   # - FEEDBACK_MAX ~ FEEDBACK_MAX
        return f_sin, f_cos
