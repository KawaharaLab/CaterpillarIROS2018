import numpy as np
from . import config
from . import base_actor

np.seterr(all="raise")
F_OUTPUT_BOUND = 1.

def simple_bound(x: np.array, lower_bound: float, upper_bound: float) -> float:
    return np.maximum(np.minimum(x, upper_bound), lower_bound)

class Actor(base_actor.BaseActor):
    def __str__(self):
        return "LinearActor"

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

            state: [f_0, f_1, ..., phi_0, phi_1, ..., t_0, t_1, ...]
            """
            forces = state[:config.somites]
            phis = state[config.somites:config.somites + config.oscillators]

            f_sin, f_cos = self._calc_fs(forces)
            return f_sin * np.sin(phis) + f_cos * np.cos(phis)

        return action_infer

    def _calc_fs(self, forces: np.array) -> tuple:
        assert forces.shape == (config.somites,)
        return simple_bound(np.matmul(forces, self.var("w_sin")) + self.var("b_sin"), -F_OUTPUT_BOUND, F_OUTPUT_BOUND),\
            simple_bound(np.matmul(forces, self.var("w_cos")) + self.var("b_cos"), -F_OUTPUT_BOUND, F_OUTPUT_BOUND)

    def get_raw_params(self) -> list:
        return self.var("w_sin"), self.var("w_cos"), self.var("b_sin"), self.var("b_cos")
