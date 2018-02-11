import numpy as np
from . import config
from . import base_actor

np.seterr(all="raise")
F_OUTPUT_BOUND = 1.
HIDDEN_LAYER_UNITS = 5


def simple_bound(x: np.array, lower_bound: float, upper_bound: float) -> float:
    return np.maximum(np.minimum(x, upper_bound), lower_bound)


def sigmoid(x: np.array) -> np.array:
    x = np.clip(x, -500, 500)
    return 1. / (1 + np.exp(-x))

def limit_angle_range(angles: np.array) -> np.array:
    return np.maximum(0, np.minimum(config.caterpillar_params["realtime_tunable_ts_rom"], angles))


class Actor(base_actor.BaseActor):
    def __str__(self):
        return "LinearActor"

    def _build_network(self):
        r"""
        Define linear function for feedback model.

        \dot{\phi}_i = \omega + f_i(F) \cos \phi_i
        return: [action_0, action_1, ...]
        """
        self.new_trainable_variable("w0_sin", np.zeros(
            (config.somites * 2 - 2, HIDDEN_LAYER_UNITS), dtype=np.float64))
        self.new_trainable_variable("b0_sin", np.zeros(HIDDEN_LAYER_UNITS, dtype=np.float64))
        self.new_trainable_variable("w1_sin", np.zeros(
            (HIDDEN_LAYER_UNITS, config.oscillators), dtype=np.float64))
        self.new_trainable_variable("b1_sin", np.zeros(config.oscillators, dtype=np.float64))

        self.new_trainable_variable("w0_cos", np.zeros(
            (config.somites * 2 - 2, HIDDEN_LAYER_UNITS), dtype=np.float64))
        self.new_trainable_variable("b0_cos", np.zeros(HIDDEN_LAYER_UNITS, dtype=np.float64))
        self.new_trainable_variable("w1_cos", np.zeros(
            (HIDDEN_LAYER_UNITS, config.oscillators), dtype=np.float64))
        self.new_trainable_variable("b1_cos", np.zeros(config.oscillators, dtype=np.float64))

        self.new_trainable_variable("angle_ranges", np.ones(config.oscillators, dtype=np.float64) * config.caterpillar_params["realtime_tunable_ts_rom"])


        def action_infer(state: np.array) -> np.array:
            """
            Get state and return feedback.

            state: [f_0, f_1, ..., phi_0, phi_1, ..., t_0, t_1, ...]
            return: [phase_feedback0, phase_feedback1, ..., angle_range0, angle_range1, ...]
            """
            forces = state[:config.somites]
            phis = state[config.somites:config.somites + config.oscillators]
            tensions = state[config.somites + config.oscillators:]

            f_sin, f_cos = self._calc_fs(np.concatenate((forces, tensions)))
            return f_sin * np.sin(phis) + f_cos * np.cos(phis), limit_angle_range(self.var("angle_ranges"))

        return action_infer

    def _calc_fs(self, state: np.array) -> tuple:
        assert state.shape == (config.somites * 2 - 2,), "state shape should be {}, got {}".format((config.somites * 2 - 2,), state.shape)
        f_sin = sigmoid(
                    np.matmul(sigmoid(
                        np.matmul(state, self.var("w0_sin")) + self.var("b0_sin")
                    ), self.var("w1_sin")) + self.var("b1_sin")
                )
        f_cos = sigmoid(
                    np.matmul(sigmoid(
                        np.matmul(state, self.var("w0_cos")) + self.var("b0_cos")
                    ), self.var("w1_cos")) + self.var("b1_cos")
                )
        return f_sin, f_cos

    def get_raw_params(self) -> list:
        return self.var("w0_sin"), self.var("w1_sin"), self.var("w0_cos"), self.var("w1_cos"),\
            self.var("b0_sin"), self.var("b1_sin"), self.var("b0_cos"), self.var("b1_cos")
