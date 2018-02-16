import numpy as np
from . import config
from . import base_actor

np.seterr(all="raise")
F_OUTPUT_BOUND = 1.
HIDDEN_LAYER_UNITS = 5


def sigmoid(x: np.array) -> np.array:
    x = np.clip(x, -500, 500)
    return 1. / (1 + np.exp(-x))


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
            (config.somites, HIDDEN_LAYER_UNITS), dtype=np.float64))
        self.new_trainable_variable("b0_sin", np.zeros(HIDDEN_LAYER_UNITS, dtype=np.float64))
        self.new_trainable_variable("w1_sin", np.zeros(
            (HIDDEN_LAYER_UNITS, config.oscillators + config.grippers), dtype=np.float64))
        self.new_trainable_variable("b1_sin", np.zeros(config.oscillators + config.grippers, dtype=np.float64))

        self.new_trainable_variable("w0_cos", np.zeros(
            (config.somites, HIDDEN_LAYER_UNITS), dtype=np.float64))
        self.new_trainable_variable("b0_cos", np.zeros(HIDDEN_LAYER_UNITS, dtype=np.float64))
        self.new_trainable_variable("w1_cos", np.zeros(
            (HIDDEN_LAYER_UNITS, config.oscillators + config.grippers), dtype=np.float64))
        self.new_trainable_variable("b1_cos", np.zeros(config.oscillators + config.grippers, dtype=np.float64))

        # self.new_trainable_variable("gripping_phase_threshold", np.ones(config.grippers, dtype=np.float64) * config.caterpillar_params["gripping_phase_threshold"])

        def action_infer(state: np.array) -> np.array:
            """
            Get state and return feedback.

            state: [f_0, f_1, ..., phi_0, phi_1, ..., t_0, t_1, ...]
            return: [phase_feedback0, phase_feedback1, ..., angle_range0, angle_range1, ...]
            """
            frictions = state[:config.somites]
            phis = state[config.somites:config.somites + config.oscillators + config.grippers]

            f_sin, f_cos = self._calc_fs(frictions)
            return f_sin * np.sin(phis) + f_cos * np.cos(phis),\
                np.ones(config.grippers) * config.caterpillar_params["gripping_phase_threshold"]
                # np.clip(self.var("gripping_phase_threshold"), -1., 1)

        return action_infer

    def _calc_fs(self, state: np.array) -> tuple:
        assert state.shape == (config.somites,), "state shape should be {}, got {}".format((config.somites * 2 - 2,), state.shape)
        f_sin = F_OUTPUT_BOUND * sigmoid(
                    np.matmul(sigmoid(
                        np.matmul(state, self.var("w0_sin")) + self.var("b0_sin")
                    ), self.var("w1_sin")) + self.var("b1_sin")
                )
        f_cos = F_OUTPUT_BOUND * sigmoid(
                    np.matmul(sigmoid(
                        np.matmul(state, self.var("w0_cos")) + self.var("b0_cos")
                    ), self.var("w1_cos")) + self.var("b1_cos")
                )
        return f_sin, f_cos

    def get_raw_params(self) -> list:
        return self.var("w0_sin"), self.var("w1_sin"), self.var("w0_cos"), self.var("w1_cos"),\
            self.var("b0_sin"), self.var("b1_sin"), self.var("b0_cos"), self.var("b1_cos")

    @staticmethod
    def dump_config() -> dict:
        return {
            "actor name": "neural network actor",
            "F_OUTPUT_BOUND": F_OUTPUT_BOUND,
            "HIDDEN_LAYER_UNITS": HIDDEN_LAYER_UNITS,
        }
