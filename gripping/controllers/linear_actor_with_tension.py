import numpy as np
from . import config
from . import base_actor

np.seterr(all="raise")
F_OUTPUT_BOUND = 1.


class Actor(base_actor.BaseActor):
    def __str__(self):
        return "LinearActorWithTension"

    def _build_network(self):
        r"""
        Define linear function for feedback model.

        \dot{\phi}_i = \omega + f_i(F) \cos \phi_i
        return: [action_0, action_1, ...]
        """
        self.new_trainable_variable("w_sin", np.zeros(
            (config.somites * 2 - 2, config.oscillators + config.grippers), dtype=np.float64))
        self.new_trainable_variable("b_sin", np.zeros(config.oscillators + config.grippers, dtype=np.float64))

        self.new_trainable_variable("w_cos", np.zeros(
            (config.somites * 2 - 2, config.oscillators + config.grippers), dtype=np.float64))
        self.new_trainable_variable("b_cos", np.zeros(config.oscillators + config.grippers, dtype=np.float64))


        def action_infer(state: np.array) -> np.array:
            """
            Get state and return feedback.

            state: [f_0, f_1, ..., phi_0, phi_1, ..., t_0, t_1, ...]
            return: [phase_feedback0, phase_feedback1, ..., angle_range0, angle_range1, ...]
            """
            forces = state[:config.somites]
            phis = state[config.somites:config.somites + config.oscillators + config.grippers]
            tensions = state[config.somites + config.oscillators + config.grippers:]

            f_sin, f_cos = self._calc_fs(np.concatenate((forces, tensions)))
            return f_sin * np.sin(phis) + f_cos * np.cos(phis),\
                np.ones(config.grippers) * config.caterpillar_params["gripping_phase_threshold"]


        return action_infer

    def _calc_fs(self, state: np.array) -> tuple:
        assert state.shape == (config.somites * 2 - 2,), "state shape should be {}, got {}".format((config.somites * 2 - 2,), state.shape)
        f_sin = np.clip(np.matmul(state, self.var("w_sin")) + self.var("b_sin"), -F_OUTPUT_BOUND, F_OUTPUT_BOUND)
        f_cos = np.clip(np.matmul(state, self.var("w_cos")) + self.var("b_cos"), -F_OUTPUT_BOUND, F_OUTPUT_BOUND)
        return f_sin, f_cos

    def get_raw_params(self) -> list:
        return self.var("w_sin"), self.var("w_cos"), self.var("b_sin"), self.var("b_cos")

    @staticmethod
    def dump_config() -> dict:
        return {
            "actor name": "linear actor with tensions",
            "F_OUTPUT_BOUND": F_OUTPUT_BOUND,
        }
