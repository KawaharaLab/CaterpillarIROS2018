import numpy as np
from . import config
from . import base_actor

np.seterr(all="raise")
DISCREPANCY_COEF_MAX = 1.0
DISCREPANCY_COEF = .5
F_OUTPUT_BOUND = 1.
HIDDEN_LAYER_UNITS = 5


def sigmoid(x: np.array) -> np.array:
    x = np.clip(x, -500, 500)
    return 1. / (1 + np.exp(-x))


class Actor(base_actor.BaseActor):
    def __str__(self):
        return "DiscrepancyActor"

    def _build_network(self):
        r"""
        Define linear function for feedback model.

        \dot{\phi_i} = \omega + f_i(F) \cos \phi_i - \alpha \frac{\partial I}{\partial \phi_i}
        \frac{\partial I}{\partial} = \alpha k \bar{L} A T \sin \phi_i

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

        def action_infer(state: np.array) -> np.array:
            """
            Get state and return feedback.

            state: [f_0, f_1, ..., phi_0, phi_1, ..., t_0, t_1, ...]
            return: [phase_feedback0, phase_feedback1, ..., angle_range0, angle_range1, ...]

            Discrepancy for torsion spring = alpha / 2 * k * range * T * sin(phi_i)
            """
            forces = state[:config.somites]
            phis = state[config.somites:config.somites + config.oscillators]
            tensions = state[config.somites + config.oscillators:]

            f_sin, f_cos = self._calc_fs(np.concatenate((forces, tensions)))
            discrepancies = -0.5 * config.caterpillar_params["vertical_ts_k"] * config.caterpillar_params["realtime_tunable_ts_rom"] * tensions * np.sin(phis)
            return f_sin * np.sin(phis) + f_cos * np.cos(phis) - self.get_discrep_coeffs() * discrepancies, np.ones(config.oscillators) * config.caterpillar_params["realtime_tunable_ts_rom"]

        return action_infer

    def _calc_fs(self, state: np.array) -> tuple:
        assert state.shape == (config.somites * 2 - 2,), "state shape should be {}, got {}".format((config.somites * 2 - 2,), state.shape)
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

    def get_discrep_coeffs(self) -> float:
        return DISCREPANCY_COEF

    def get_raw_params(self) -> list:
        return self.var("w0_sin"), self.var("w1_sin"), self.var("w0_cos"), self.var("w1_cos"),\
            self.var("b0_sin"), self.var("b1_sin"), self.var("b0_cos"), self.var("b1_cos")

    @staticmethod
    def dump_config() -> dict:
        return {
            "actor name": "discrepancy_actor",
            "DISCREPANCY_COEF": DISCREPANCY_COEF,
            "F_OUTPUT_BOUND": F_OUTPUT_BOUND,
            "HIDDEN_LAYER_UNITS": HIDDEN_LAYER_UNITS,
        }
