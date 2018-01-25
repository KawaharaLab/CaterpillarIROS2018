import numpy as np

from . import config
from . import base_actor


np.seterr(all="raise")


class Actor(base_actor.BaseActor):
    def __str__(self):
        return "SimpleNeuralNetworkActor"

    def _build_network(self):
        r"""
        Define linear function for feedback model.

        \dot{\phi}_i = \omega + f_i(F) \cos \phi_i
        return: [action_0, action_1, ...]
        """
        self.new_trainable_variable("w_sin", np.zeros(
            (config.somites, config.oscillators), dtype=np.float32))
        self.new_trainable_variable("b_sin", np.zeros(config.oscillators, dtype=np.float32))
        self.new_trainable_variable("w_cos", np.zeros(
            (config.somites, config.oscillators), dtype=np.float32))
        self.new_trainable_variable("b_cos", np.zeros(config.oscillators, dtype=np.float32))

        def action_infer(state: np.array) -> np.array:
            """
            Get state and return feedback.

            state: [f_0, f_1, ..., phi_0, phi_1, ...]
            """
            forces = state[:config.somites]
            phis = state[config.somites:]

            f_sin, f_cos = self._calc_fs(forces)
            return f_sin * np.sin(phis) + f_cos * np.cos(phis)

        return action_infer

    def _calc_fs(self, forces: np.array) -> tuple:
        assert forces.shape == (config.somites,)
        w_sin, w_cos, b_sin, b_cos = self.get_params()
        return np.matmul(forces, w_sin) + b_sin, np.matmul(forces, w_cos) + b_cos

    def get_amplitudes_and_phases(self, forces: np.array) -> tuple:
        f_sin, f_cos = self._calc_fs(forces)
        return np.sqrt(np.power(f_sin, 2) + np.power(f_cos, 2)), np.arctan2(f_cos, -f_sin)

    def get_params(self) -> list:
        def tanh(x: np.array) -> np.array:
            return (1. - np.exp(-2 * x)) / (1. + np.exp(-2 * x))

        sigma = 1. / config.oscillators
        adjust = sigma
        return adjust * tanh(self.var("w_sin")), adjust * tanh(self.var("w_cos")),\
            adjust * tanh(self.var("b_sin")), adjust * tanh(self.var("b_cos"))

    def get_raw_params(self) -> list:
        return self.var("w_sin"), self.var("w_cos"), self.var("b_sin"), self.var("b_cos")
